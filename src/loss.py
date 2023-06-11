import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PseudoLabelFocalCE(torch.nn.Module,):
    def __init__(self, beta, num_classes, focal_p=3.0, conf_discount=True, confidence_upper_bound=0.75, confidence_lower_bound=0.2):
        super(PseudoLabelFocalCE, self).__init__()
        self.running_conf = torch.ones(num_classes).float()/num_classes
        self.beta = beta
        self.confidence_upper_bound = confidence_upper_bound
        self.confidence_lower_bound = confidence_lower_bound
        self.conf_discount = conf_discount
        self.focal_p = focal_p
        
    def _update_running_conf(self, probs, tolerance=1e-8, momentum=0.99):
        """Maintain the moving class prior"""
        B,C,H,W = probs.size()
        probs_avg = probs.mean(0).view(C,-1).mean(-1)

        # updating the new records: copy the value
        update_index = probs_avg > tolerance
        new_index = update_index & (self.running_conf == self.beta)
        self.running_conf[new_index] = probs_avg[new_index]

        # use the moving average for the rest
        self.running_conf *= momentum
        self.running_conf += (1 - momentum) * probs_avg

    def _focal_ce_conf(self, logits, pseudo_gt, teacher_probs):
        focal_weight = (1 - self.running_conf.clamp(0.)) ** self.focal_p
        loss_ce = F.cross_entropy(logits, pseudo_gt, weight=focal_weight.detach(), ignore_index=255, reduction="none")

        with torch.no_grad():
            C = logits.size(1)
            B,H,W = loss_ce.size()
            loss_per_class = torch.zeros_like(logits)
            loss_idx = pseudo_gt.clone()
            loss_idx[pseudo_gt == 255] = 0
            loss_per_class.scatter_(1, loss_idx[:,None], loss_ce[:,None]) # B,C,H,W
            loss_per_class = loss_per_class.view(B, C, -1).mean(-1).mean(0)

        #teacher_norm = teacher_probs.sum() + 1e-3
        loss = (loss_ce * teacher_probs) # / teacher_norm
        return loss, loss_per_class

    def _threshold_discount(self):
        return 1. - torch.exp(- self.running_conf / self.beta)
    
    def _pseudo_labels_probs(self, probs, discount = True): #ignore_augm
        """Consider top % pixel w.r.t. each image"""

        B,C,H,W = probs.size()
        max_conf, max_idx = probs.max(1, keepdim=True) # B,1,H,W

        probs_peaks = torch.zeros_like(probs)
        probs_peaks.scatter_(1, max_idx, max_conf) # B,C,H,W
        top_peaks, _ = probs_peaks.view(B,C,-1).max(-1) # B,C

        # > used for ablation
        #top_peaks.fill_(1.)

        top_peaks *= self.confidence_upper_bound

        if discount:
            # discount threshold for long-tail classes
            top_peaks *= self._threshold_discount().view(1, C)

        top_peaks.clamp_(self.confidence_lower_bound) # in-place
        probs_peaks.gt_(top_peaks.view(B,C,1,1))

        # ignore if lower than the discounted peaks
        ignore = probs_peaks.sum(1, keepdim=True) != 1

        # thresholding the most confident pixels
        pseudo_labels = max_idx.clone()
        pseudo_labels[ignore] = 0.0

        pseudo_labels = pseudo_labels.squeeze(1)
        #pseudo_labels[ignore_augm] = 0.0

        return pseudo_labels, max_conf, max_idx
    
    def forward(self, logits, target):
        device = logits.device
        if self.running_conf.device != device:
            self.running_conf = self.running_conf.to(device)
        pseudo_labels, teacher_conf, teacher_maxi = self._pseudo_labels_probs(target, self.conf_discount)
        loss_ce, loss_per_class = self._focal_ce_conf(logits, pseudo_labels, teacher_conf)
        return loss_ce, loss_per_class


class DINOLoss(nn.Module):
    def __init__(self, out_channels, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_channels))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, input, target, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_output = input
        teacher_output = target
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center.unsqueeze(-1).unsqueeze(-1)) / temp, dim=1)
        teacher_out = teacher_out.detach()

        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=1), dim=1) # should be same shape as img
        self.update_center(teacher_output)
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        
        new_center = teacher_output.mean([2,3])
        # ema update
        self.center = self.center * self.center_momentum + new_center * (1 - self.center_momentum)


def get_projection_loss(params):
    if params['projection_loss'] == 'DINO':
        loss_fn = DINOLoss(out_channels=params['embedding_dim'], warmup_teacher_temp=0.04, teacher_temp=0.07,
                    warmup_teacher_temp_epochs=5000, nepochs=300000).to(params['device'])        
    elif params['projection_loss'] == 'MSE':
        def loss_fn(input, target, epoch):
            return getattr(F, 'mse_loss')(input, target, reduction='none')
    elif params['projection_loss'] == 'COSINE':
        def loss_fn(input, target, epoch):
            return (F.cosine_similarity(
                input,target,dim=1)-1).abs().mean()
    elif params['projection_loss'] == 'L1':
        def loss_fn(input, target, epoch):
            return getattr(F, 'l1_loss')(input, target, reduction='none')
    return loss_fn