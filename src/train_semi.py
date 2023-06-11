import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torch.utils.data import DataLoader
import random
import h5py
import torch.nn.functional as F
from data_utils import *
import numpy as np
from unet import *
import copy
from data_utils import save_model
from torch.utils.tensorboard import SummaryWriter
import time
import segmentation_models_pytorch as smp
import toml
from loss import get_projection_loss
from augmentations import test_time_aug, prep_intensity_aug_fn, prep_spatial_aug_fn
from torchvision import datapoints as DP
from projector import prep_projection_head

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

params = {
    #'data': 'data/DSB2018_n0/train/train_data.npz', # DSB data, 10 samples 10 19 38 76 152 
    #'data': 'data/Mouse_n0/train/train_data.npz', # Mouse data, 5 samples 5 10 19 38 76
    'data': 'data/Flywing_n0/train/train_data.npz', # Flywing data, 5 samples 5 10 19 38 76
    'experiment' : 'exp_0_flywing_seed1_samples10_DINO',
    'batch_size_labeled': 1,
    'batch_size_unlabeled': 2,
    'num_workers': 1,
    'training_steps':300000,
    'in_channels': 1,
    'num_fmaps': 32,
    'fmap_inc_factors': 2,
    'downsample_factors': [ [ 2, 2,], [ 2, 2,], [ 2, 2,], [ 2, 2,],],
    'num_fmaps_out': 3,
    'constant_upsample': False,
    'padding': 'same',
    'activation': 'ReLU',
    'learning_rate': 1.0e-2,
    'num_annotated' : 10,
    'seed': 1,
    'checkpoint_path': 'exp_0_flywing_seed1_samples10_pretrained/best_model',
    'projection_loss': True,
    'proj_loss_weight': 0.1,
    'pretrained_model': True,
    'projection_softmax': 'none', # 'pre' projection, 'post' projection or none
    'projection_loss': 'L1', # 'MSE' or 'COSINE' or 'L1' or DINO
    'projection_head': 'MLP', # 'DINO', 'MLP' or 'SHALLOW'
    'trainable_projector': True,
    'embedding_dim': 512,
    'load_student_weights': True, # if False it initiliazes a fresh student model instead of loading one
    'warmup_steps': 0, # during warmup_steps the teacher model doesn't get updated
    'batchsize_unlabeled': 10,
    'batchsize_labeled': 2,
    'fast_update_slow_model': True,
    'aug_params': {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1,0.1), 'scale': (0.5,1.5), 'shear': 0.2,}, "p": 0.25},
        'ElasticTransform': {"kwargs": {'alpha': [120.0,120.0], 'sigma': 8.0}, "p": 0.25},
    }
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params['device'] = device

X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks = prepare_data(params) # BHW

loss_weight = torch.Tensor([1.0,1.0,4.0]).to(device)

if 'Flywing' in params['data']:
    print('Set interior to zero')
    Y_labeled[Y_labeled==1] = 0
    Y_val_masks[Y_val_masks==1] = 0
    loss_weight = torch.Tensor([0.0,1.0,4.0]).to(device)
    
X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks = [
    d[:,np.newaxis,...] for d in [X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks]
]

Y_labeled = convert_to_oneHot(Y_labeled)
Y_labeled = Y_labeled.argmax(-1)

X_val = X_val
Y_val = convert_to_oneHot(Y_val_masks)
Y_val = Y_val.argmax(-1).squeeze(1)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# TODO put data into TensorDataset and put whole dataset on device
# implement normalization in torch instead of numpy
labeled_dataset = SliceDataset(raw=X_labeled, labels=Y_labeled)
labeled_dataloader = DataLoader(labeled_dataset,
                    batch_size=params['batch_size_labeled'],
                    prefetch_factor=32 if params['num_workers'] > 1 else None,
                    sampler=torch.utils.data.RandomSampler(
                        [1.0]*len(labeled_dataset),
                        num_samples=params['num_annotated'],
                        replacement=True
                    ),
                    worker_init_fn=worker_init_fn,
                    num_workers=params['num_workers']//2)

unlabeled_dataset = SliceDataset(raw=X_unlabeled, labels=None)
unlabeled_dataloader = DataLoader(unlabeled_dataset,
                    batch_size=params['batch_size_unlabeled'],
                    prefetch_factor=32 if params['num_workers'] > 1 else None,
                    sampler=torch.utils.data.RandomSampler(
                        [1.0]*len(unlabeled_dataset),
                        num_samples=params['training_steps'],
                        replacement=True
                    ),
                    worker_init_fn=worker_init_fn,
                    num_workers=params['num_workers']//2)

validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
validation_dataloader = DataLoader(validation_dataset,
                    batch_size=10,
                    shuffle=True,
                    prefetch_factor=32 if params['num_workers'] > 1 else None,
                    num_workers=params['num_workers']//2)


if params['pretrained_model']:
    model = smp.Unet(
        encoder_name= "timm-efficientnet-b5", # "timm-efficientnet-b5", # choose encoder
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=params['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=params['num_fmaps_out'],          # model output channels (number of classes in your dataset)
        ).to(params['device'])
else:
    model = UNet(
        in_channels = params['in_channels'],
        num_fmaps = params['num_fmaps'],
        fmap_inc_factor = params['fmap_inc_factors'],
        downsample_factors = params['downsample_factors'],
        activation = params['activation'],
        padding = params['padding'],
        num_fmaps_out = params['num_fmaps_out'],
        constant_upsample = params['constant_upsample']
    ).to(params['device'])
if params['load_student_weights']:
    model.load_state_dict(torch.load(params['checkpoint_path'])['model_state_dict'])
    model = model.train()
    slow_model = copy.deepcopy(model)
    slow_model = slow_model.train()
else:
    model = model.train()
    slow_model = copy.deepcopy(model)
    slow_model.load_state_dict(torch.load(params['checkpoint_path'])['model_state_dict'])
    slow_model = slow_model.train()

projector = prep_projection_head(params)


optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['training_steps'], eta_min=1e-5)

s_augment = prep_spatial_aug_fn(params['aug_params'])
c_augment = prep_intensity_aug_fn(128)
loss_weight = torch.Tensor([1.0,1.0,4.0]).to(device)

os.makedirs(os.path.join(params['experiment'],'train','snaps'),exist_ok=True)
writer_dir = os.path.join(params['experiment'],'train','summary',str(time.time()))
os.makedirs(writer_dir,exist_ok=True)
writer = SummaryWriter(writer_dir)

with open(os.path.join(params['experiment'], 'params.toml'), 'w') as f:
    toml.dump(params, f)

validation_loss = []
validation_loss_slow = []
step = -1

loss_fn = get_projection_loss(params)

while True:
    for (raw_labeled, gt_3c), raw_unlabeled in zip(labeled_dataloader, unlabeled_dataloader):
        if step>params['training_steps']:
            break
        step += 1
        for param in model.parameters():
            param.grad = None
        gt_3c = gt_3c.to(device) # add channel dimension
        raw_labeled = raw_labeled.to(device)
        raw_unlabeled = raw_unlabeled.to(device)
        
        pseudo_labels = test_time_aug(raw_unlabeled, slow_model)
        raw_labeled = DP.Image(raw_labeled)
        raw_unlabeled = DP.Image(raw_unlabeled)
        pseudo_labels = DP.Image(pseudo_labels)
        gt_3c = DP.Mask(gt_3c)

        raw_labeled_aug, raw_unlabeled_aug, gt_3c_aug, pseudo_labels_aug = s_augment(
            raw_labeled, raw_unlabeled, gt_3c, pseudo_labels
        )
        pseudo_labels_aug_mask = pseudo_labels_aug != 0
        raw_labeled_aug, raw_unlabeled_aug = c_augment(raw_labeled_aug, raw_unlabeled_aug)
        
        # concat labeled and unlabeled data + forward pass
        raw_aug = torch.cat([raw_labeled_aug, raw_unlabeled_aug], dim=0)
        model_out = model(raw_aug)

        # split into labeled and unlabeled data
        out_labeled, out_unlabeled = torch.split(
            model_out, [params['batch_size_labeled'], params['batch_size_unlabeled']], dim=0
        )

        if params['projection_softmax'] == 'pre':
            out_fast_proj = projector(out_unlabeled.softmax(1))
            pseudo_labels_proj = projector(pseudo_labels_aug.softmax(1))
        elif params['projection_softmax'] == 'post':
            out_fast_proj = projector(out_unlabeled).softmax(1)
            pseudo_labels_proj = projector(pseudo_labels_aug).softmax(1)
        else:
            out_fast_proj = projector(out_unlabeled)
            pseudo_labels_proj = projector(pseudo_labels_aug)

        # loss calculation
        loss_img = loss_fn(
            input = out_fast_proj,
            target = pseudo_labels_proj.detach(),
            epoch = step    
        )
        loss_img = loss_img.mean(1)
        # mask out areas that were padded during augmentation
        loss_img *= pseudo_labels_aug_mask.any(1).to(float)

        loss_img *= params['proj_loss_weight']
        con_loss = loss_img.mean() 
        writer.add_scalar('aug_con_loss', con_loss.item(), step)
        loss = con_loss
        
        loss_img = F.cross_entropy(
            input = out_labeled,
            target = gt_3c_aug.squeeze(0).long(), 
            weight = loss_weight,
            reduction = 'none'
        )
        ce_loss = loss_img.mean()
        writer.add_scalar('ce_loss', ce_loss.item(), step)
        loss += ce_loss
        print('step: ', step, 'loss: ', loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save snaps
        if step % 10000 ==0:
            # save training snap
            out_dict = {}
            out_dict['raw_labeled'] = raw_labeled_aug[0,...].cpu().detach().numpy()
            out_dict['pred_labeled'] = out_labeled[0,...].softmax(0).squeeze().cpu().detach().numpy()
            out_dict['label'] = gt_3c_aug[0,...].cpu().detach().numpy()
            out_dict['raw_unlabeled'] = raw_unlabeled_aug[0,...].cpu().detach().numpy()
            out_dict['pseudo_label'] = pseudo_labels_aug[0,...].softmax(0).cpu().detach().numpy()
            with h5py.File(os.path.join(params['experiment'],'train/snaps/snap_step_'+str(step)+'.hdf'),'w') as f:
                for key in list(out_dict.keys()):
                    f.create_dataset(key, data = out_dict[key].astype(np.float32))
        if step % 100 == 0 and step > params['warmup_steps']:
            print('Update Momentum Network')
            for param_slow, param_fast in zip(slow_model.parameters(), model.parameters()):
                param_slow = 0.99 * param_slow + 0.01 * param_fast
        if step % 200 == 0:
            val_loss = []
            val_loss_slow = []
            for raw, gt_3c in validation_dataloader:
                raw = raw.to(device)
                with torch.no_grad():
                    out = model(raw)
                    out_slow = slow_model(raw)
                b,c,h,w = out.shape
                label = gt_3c.to(device)
                loss = F.cross_entropy(input = out,
                                        target = label.long(),
                                        weight = loss_weight)
                loss_slow = F.cross_entropy(input = out_slow,
                                        target = label.long(),
                                        weight = loss_weight)
                val_loss.append(loss.item())
                val_loss_slow.append(loss_slow.item())
            val_new = np.mean(val_loss)
            val_new_slow = np.mean(val_loss_slow)
            validation_loss.append(val_new)
            validation_loss_slow.append(val_new_slow)
            writer.add_scalar('val_loss', val_new, step)
            writer.add_scalar('val_loss_slow', val_new_slow, step)
            print('Validation loss: ', val_new)
            if val_new <= np.min(validation_loss):
                print('Save best model')
                save_model(step, model, optimizer, loss, params['experiment']+"/best_model")
                if val_new<val_new_slow and params['fast_update_slow_model']:
                    slow_model = copy.deepcopy(model)
            if val_new_slow <= np.min(validation_loss_slow):
                print('Save best slow model')
                save_model(step, slow_model, optimizer, loss, params['experiment']+"/best_slow_model")