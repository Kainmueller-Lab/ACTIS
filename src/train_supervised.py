import torch
import torch.nn.functional as F
import numpy as np
from spatial_augmenter import SpatialAugmenter
import os
import h5py
from torch.utils.data import DataLoader
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from data_utils import *
import time 
import matplotlib.pyplot as plt
import random
import segmentation_models_pytorch as smp
import toml

torch.backends.cudnn.benchmark = True
torch.manual_seed(471)

aug_params= {
    'mirror': {'prob': 0.5, 'prob_x': 0.25, 'prob_y': 0.25},
    'translate': {'max_percent':0.1, 'prob': 0.2},
    'scale': {'min': 0.5, 'max':1.5, 'prob': 0.5},
    'zoom': {'min': 0.2, 'max':2.0, 'prob': 0.5},
    'rotate': {'max_degree': 90, 'prob': 0.5},
    'shear': {'max_percent': 0.2, 'prob': 0.5},
    'elastic': {'alpha': [120,120], 'sigma': 8, 'prob': 0.75}
}

params = {
    #'data': 'data/DSB2018_n0/train/train_data.npz', # DSB data, 10 samples 10 19 38 76 152 
    #'data': 'data/Mouse_n0/train/train_data.npz', # Mouse data, 5 samples 5 10 19 38 76
    'data': 'data/Flywing_n0/train/train_data.npz', # Flywing data, 5 samples 5 10 19 38 76
    'experiment' : 'exp_0_flywing_seed1_samples10_pretrained', # flywing sample 10 is missing
    'batch_size': 10,
    'training_steps':160000,
    'in_channels': 1,
    'num_fmaps': 32,
    'fmap_inc_factors': 2,
    'downsample_factors': [ [ 2, 2,], [ 2, 2,], [ 2, 2,], [ 2, 2,],],
    'num_fmaps_out': 3,
    'constant_upsample': False,
    'padding': 'same',
    'activation': 'ReLU',
    'learning_rate': 1e-3,
    'num_annotated' : 10,
    'seed': 1,
    'pretrained_model': True,
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params['device'] = device

trainval_data =  np.load(params['data'])
train_images = trainval_data['X_train'].astype(np.float32)
train_masks = trainval_data['Y_train']
val_images = trainval_data['X_val'].astype(np.float32)
val_masks = trainval_data['Y_val']
number_of_annotated_training_images = params['num_annotated']

# Seed to shuffle training data (annotated GT and raw image pairs).
seed = params['seed']

# First we shuffle the training images to remove any bias.
X_shuffled, Y_shuffled = shuffle_train_data(train_images, train_masks, random_seed=seed)

# Here we convert the number of annotated images to be used for training as percentage of available training data.
percentage_of_annotated_training_images = float((number_of_annotated_training_images/train_images.shape[0])*100.0)
assert percentage_of_annotated_training_images >= 0.0 and percentage_of_annotated_training_images <=100.0

# Here we zero out the segmentations of those training images which are not part of the selected annotated images.
X_frac, Y_frac = zero_out_train_data(X_shuffled, Y_shuffled, fraction = percentage_of_annotated_training_images)
X_labeled, Y_labeled =  X_shuffled[:number_of_annotated_training_images], Y_shuffled[:number_of_annotated_training_images]
X_unlabeled = X_shuffled[number_of_annotated_training_images:]

X_val, Y_val_masks = val_images, val_masks

loss_weight = torch.Tensor([1.0,1.0,4.0]).to(device)

if 'Flywing' in params['data']:
    print('Set interior to zero')
    Y_labeled[Y_labeled==1] = 0
    Y_val_masks[Y_val_masks==1] = 0
    loss_weight = torch.Tensor([0.0,1.0,4.0]).to(device)

# Here we add the channel dimension to our input images.
# Dimensionality for training has to be 'SYXC' (Sample, Y-Dimension, X-Dimension, Channel)
X_labeled = X_labeled[:,np.newaxis,...]
X_unlabeled = X_unlabeled[:,np.newaxis,...]

Y_labeled = convert_to_oneHot(Y_labeled)
Y_labeled = Y_labeled.argmax(-1)

X_val = X_val[:,np.newaxis,...]
Y_val = convert_to_oneHot(Y_val_masks)
Y_val = Y_val.argmax(-1)

labeled_dataset = SliceDataset(raw=X_labeled, labels=Y_labeled)
labeled_dataloader = DataLoader(labeled_dataset,
                    batch_size=10,
                    shuffle=True,
                    prefetch_factor=4,
                    num_workers=4)

validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
validation_dataloader = DataLoader(validation_dataset,
                    batch_size=20,
                    shuffle=True,
                    prefetch_factor=4,
                    num_workers=4)

unlabeled_dataset = SliceDataset(raw=X_unlabeled, labels=None)
unlabeled_dataloader = DataLoader(unlabeled_dataset,
                    batch_size=20,
                    shuffle=True,
                    prefetch_factor=4,
                    num_workers=4)

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


learning_rate = params['learning_rate']
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['training_steps'], eta_min=1e-5)

os.makedirs(os.path.join(params['experiment'],'train','snaps'),exist_ok=True)
writer_dir = os.path.join(params['experiment'],'train','summary',str(time.time()))
os.makedirs(writer_dir,exist_ok=True)
writer = SummaryWriter(writer_dir)

with open(os.path.join(params['experiment'], 'params.toml'), 'w') as f:
    toml.dump(params, f)

step = 0
s_augment = SpatialAugmenter(aug_params)
ctransform_func = color_augmentations(128)
#
validation_loss = []
while step<params['training_steps']:
    tmp_loader = iter(labeled_dataloader)
    for raw, gt_3c in tmp_loader:
        if step == params['training_steps']:
            break
        step += 1
        # augmentation pipeline
        n_views = 1
        raw = raw.to(device)
        gt_3c = gt_3c.to(device).float()
        img_caug = ctransform_func(raw)
        s_augment.interpolation='bilinear'
        img_saug = [] 
        gt_3c_saug = []
        for i in range(img_caug.shape[0]):
            img_saug_tmp, gt_3c_saug_tmp = s_augment.forward_transform(img_caug[i].unsqueeze(0), gt_3c[i].unsqueeze(0).unsqueeze(0))
            img_saug.append(img_saug_tmp)
            gt_3c_saug.append(gt_3c_saug_tmp)
        img_saug = torch.cat(img_saug, dim=0)
        gt_3c_saug = torch.cat(gt_3c_saug, dim=0)
        # calculate loss
        for param in model.parameters():
            param.grad = None
        loss = 0
        out_dict = {}
        out_fast = model(img_saug)
        loss_img = F.cross_entropy(input = out_fast,
            target = gt_3c_saug.squeeze(1).long(), # no gradient
            weight = loss_weight,
            reduction='none')
        ce_loss = loss_img.mean()
        writer.add_scalar('3 class loss', ce_loss , step)
        loss += ce_loss
        writer.add_scalar('total loss', loss , step)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 10000 == 0:
            save_model(step, model, optimizer, loss, params['experiment']+"/model_step_"+str(step))
        if step % 1000 == 0:
            out_dict['raw'] = img_saug[0,...].cpu().detach().numpy()
            out_dict['pred'] = out_fast[0,...].softmax(0).squeeze().cpu().detach().numpy()
            out_dict['label'] = gt_3c_saug[0,...].cpu().detach().numpy()
            # save training snap
            with h5py.File(os.path.join(params['experiment'],'train/snaps/snap_step_'+str(step)),'w') as f:
               for key in list(out_dict.keys()):
                   f.create_dataset(key, data = out_dict[key].astype(np.float32))
        val_loss = []
        if step % 200 == 0:
            for raw, gt_3c in validation_dataloader:
                raw = raw.to(device)
                out = model(raw)
                label = gt_3c.to(device)
                loss = F.cross_entropy(input = out,
                                        target = label.long().squeeze(1),
                                        weight = loss_weight)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            validation_loss.append(val_loss)            
            writer.add_scalar('val_loss', val_loss, step)
            print('Validation loss: ', val_loss)
            if val_loss <= np.min(validation_loss):
                save_model(step, model, optimizer, loss, params['experiment']+"/best_model")

