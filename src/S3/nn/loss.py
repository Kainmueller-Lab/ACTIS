import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class QuantileLoss(nn.Module):
    def __init__(self, params, nclasses=3):
        super().__init__()
        self.params = params
        self.quantile = 0
        self.quantile_start = self.params["quantile"]
        self.quantile_end = self.params["quantile_end"]
        self.quantile_warmup_steps = self.params["quantile_warmup_steps"]
        self.loss_fn = self.params["loss_fn_QuantLoss"]
        self.nclasses = nclasses
        self.quantile_ema = {
            i:{"positive": 0.2} for i in range(nclasses)
        }
    
    def quantile_ema_update(self, quantile_ema, quantile):
        """EMA update for quantile
        Args:
            quantile_ema (float): current ema value
            quantile (float): quantile value
            step (int): current step
        """
        return quantile_ema * 0.99 + quantile * 0.01

    def quantile_scheduler(self, step):
        """Linear scheduler for quantile
        Args:
            step (int): current step
        """
        return np.min([(
                    self.quantile_start
                    + ((self.quantile_end - self.quantile_start) * step)
                    / self.quantile_warmup_steps
                ), self.quantile_end])

    def class_wise_confidence_selection(self, pseudo_labels):
        """Selects the pixels with the lowest loss for each class and for positive and negative
        Args:
            pseudo_labels: pseudo labels for each pixel
        Returns:
            masks for each class and for positive and negative pixels with the lowest loss
        """
        selected_masks = []
        pseudo_labels = pseudo_labels.softmax(1)
        quantized_pseudo_l = pseudo_labels.argmax(axis=1)
        # select pixels for each class in gt
        for i in range(self.nclasses):
            single_class_mask = torch.where(quantized_pseudo_l == i,True,False)
            # get the softmax scores for single class
            single_class_scores = pseudo_labels[:,i,...][single_class_mask]
            # get the quantile for the single class
            if single_class_scores.nelement() != 0:
                quantile_positive = torch.quantile(single_class_scores, self.quantile)
                self.quantile_ema[i]["positive"] = self.quantile_ema_update(
                    self.quantile_ema[i]["positive"], quantile_positive
                )
            # threshold pseudo_label
            positive_mask = torch.where(
                (pseudo_labels[:,i,...] >= self.quantile_ema[i]["positive"]), 1, 0
            ) * single_class_mask
            # append the masks to the selected_masks
            selected_masks.append(positive_mask)
        return selected_masks, quantized_pseudo_l

    def forward(self, input, target, step):
        """Calculates the loss for the prediction and the pseudo labels
        Args:
            prediction: prediction of the model
            pseudo_labels: pseudo labels for each pixel
        Returns:
            loss: loss for the prediction and the pseudo labels
        """
        # get the masks for each class and for positive and negative pixels with the lowest loss
        self.step = step
        self.quantile = self.quantile_scheduler(self.step)
        selected_masks, quantized_gt = self.class_wise_confidence_selection(target)
        selected_masks = torch.stack(selected_masks).amax(axis=0)
        # get the loss for each pixel
        loss = self.loss_fn(input, quantized_gt)
        # get the loss for each class and for positive and negative pixels
        loss = loss * selected_masks
        self.selected_masks = selected_masks
        return loss.unsqueeze(1)