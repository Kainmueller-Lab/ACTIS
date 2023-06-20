import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from unet import *
from torch import nn
import random
from torchvision import transforms
from torchvision.transforms.transforms import RandomApply, GaussianBlur, ColorJitter
from skimage.segmentation import find_boundaries
from augmentations import test_time_aug

def save_model(step, model, optimizer, loss, filename):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, filename)
    
class CombinedDataLoader():
    '''
    DataLoader which combines all the other DataLoaders in this project. 
    __next__ yields a batch of size batch_size containing samples from all DataLoaders,
    composed based on their sampling_probs

    dataloaders: list
        Containing torch.utils.data.DataLoader objects with batch_size=1
    batch_size: int

    sampling_probs: list
        Probabilities for sampling from the dataloaders

    batch_size: int
        Determines len of output sample list
    
    buffer_size: int
        The number of samples to load into buffer before shuffling

    mode: str
        either 'train' or 'validation'
    '''
    def __init__(self, dataloaders, sampling_probs, batch_size, mode):
        if len(dataloaders) != len(sampling_probs):
            raise ValueError('Number of dataloaders does not match sampling_probs')
        if np.sum(sampling_probs) != 1.0:
            raise ValueError(f'Sampling_probs sum to {np.sum(sampling_probs)} != 1.0')
        
        self.dataloaders = dataloaders
        self.batch_size = batch_size
        self.sampling_probs = sampling_probs
        self.dataloader_iterables = [iter(dataloader) for dataloader in dataloaders]
        self.dataloader_queues = [[] for dataloader in dataloaders]
        self.mode = mode
        
    def get_item(self, idx):
        try:
            return next(self.dataloader_iterables[idx])
        except StopIteration:
            if self.mode == 'train':
                self.dataloader_iterables[idx] = iter(self.dataloaders[idx])
                return next(self.dataloader_iterables[idx])
            else:
                raise StopIteration
 
    def __iter__(self):
        return self
    
    def __next__(self):
        datasource = np.random.multinomial(1, self.sampling_probs, size=self.batch_size) # batchsize x sampling_probs
        indexes = np.argmax(datasource, axis=1) # batchsize
        batch = []
        for idx in indexes:
            batch.append(self.get_item(idx))
        if self.batch_size == 1:
            return batch.pop(0)
        else:
            return batch
    
class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, img):
        device = img.device
        noise = torch.randn(img.shape).to(device) * self.sigma
        return img + noise

def color_augmentations(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = torch.nn.Sequential(
        RandomApply([color_jitter,
                    GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.2,0.2))], p=0.5),
        GaussianNoise(0.03)
        )
    return data_transforms


def center_crop(t, croph, cropw):
    _,_,h,w = t.shape
    startw = w//2-(cropw//2)
    starth = h//2-(croph//2)
    return t[:,:,starth:starth+croph,startw:startw+cropw]

def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-8, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x   = x.astype(dtype, copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x, 0, 1)
    return x
        
def pad_up_to(tensor, xx, yy):
        """
        :param tensor: torch tensor
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """

        b,c,h,w = tensor.shape

        t = (xx - h) // 2
        b = xx - t - h

        l = (yy - w) // 2
        r = yy - l - w
        t,b,l,r = [np.abs(i) for i in [t,b,l,r]]
        return F.pad(tensor, (r,l,b,t,0,0,0,0))

class SliceDataset(Dataset):
    def __init__(self, raw, labels):
        super().__init__()
        self.raw = raw
        self.labels = labels
    def __len__(self):
        if isinstance(self.raw, list):
            return len(self.raw)
        else:
            return self.raw.shape[0]

    def __getitem__(self, idx):
        raw_tmp = normalize_percentile(self.raw[idx].astype(np.float32))
        if self.labels is not None:
            return raw_tmp, self.labels[idx].astype(np.float32)
        else:
            return raw_tmp

def shuffle_train_data(X_train, Y_train, random_seed):
    """
    Shuffles data with seed 1.
    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train : array(float)
        shuffled array of training images.
    Y_train : array(float)
        Shuffled array of labelled training images.
    """
    np.random.seed(random_seed)
    seed_ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[seed_ind]
    Y_train = Y_train[seed_ind]

    return X_train, Y_train

def zero_out_train_data(X_train, Y_train, fraction):
    """
    Fractionates training data according to the specified `fraction`.
    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    fraction: float (between 0 and 100)
        fraction of training images.
    Returns
    -------
    X_train : array(float)
        Fractionated array of source images.
    Y_train : float
        Fractionated array of label images.
    """
    train_frac = int(np.round((fraction / 100) * X_train.shape[0]))
    Y_train[train_frac:] *= 0

    return X_train, Y_train


def convert_to_oneHot(data, eps=1e-8):
    """
    Converts labelled images (`data`) to one-hot encoding.
    Parameters
    ----------
    data : array(int)
        Array of lablelled images.
    Returns
    -------
    data_oneHot : array(int)
        Array of one-hot encoded images.
    """
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
        if ( np.abs(np.max(data[i])) <= eps ):
            data_oneHot[i][...,0] *= 0

    return data_oneHot


def add_boundary_label(lbl, dtype=np.uint16):
    """
    Find boundary labels for a labelled image.
    Parameters
    ----------
    lbl : array(int)
         lbl is an integer label image (not binarized).
    Returns
    -------
    res : array(int)
        res is an integer label image with boundary encoded as 2.
    """

    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res

def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


def prepare_data(params):
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
    return X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks


def prepare_test_data(params):
    test_data =  np.load(params['test_data'], allow_pickle=True)
    test_images = test_data["X_test"]
    test_masks = test_data["Y_test"]
    if test_images.ndim == 1:
        test_images = np.split(test_images, test_images.shape[0])
        test_masks = np.split(test_masks, test_masks.shape[0])
        test_images = [test_img[0] for test_img in test_images]
        test_masks = [test_mask[0] for test_mask in test_masks]
        for i, (img, mask) in enumerate(zip(test_images, test_masks)):
            h, w = img.shape
            if h % 32 != 0 or w % 32 != 0:
                h_pad = 32 - (h % 32)
                w_pad = 32 - (w % 32)
                test_images[i] = np.pad(img, [[0,h_pad], [0, w_pad]])
                test_masks[i] = np.pad(mask, [[0,h_pad], [0, w_pad]])
    return test_images, test_masks


def tile_and_stitch_ov(model, raw, input_shape, device, overlap=(0,0), crop=(0,0), flip=False, rotate=False):
    B,C,H,W = raw.shape
    test_img = torch.zeros(1,C,*input_shape)
    with torch.no_grad():
        out = model(test_img.to(device))
    if isinstance(out, dict):
        keys = list(out.keys())
        output_shape = out[keys[0]].shape[-2:]
        out_dict = {}
        for key in keys:
            out_dict[key] = torch.zeros(B,out[key].shape[1],H+input_shape[0],W+input_shape[1])
        region_counter = torch.zeros_like(out_dict[key])
        # fill dict with zero tensors of the right shape
    else:
        keys=False
        output_shape = out.shape[-2:]
        out = torch.zeros(B,out.shape[1],H+input_shape[0],W+input_shape[1])
        region_counter = torch.zeros_like(out)
    #
    padh = (input_shape[0]-output_shape[0])//2
    padw = (input_shape[1]-output_shape[1])//2
    raw_padded = F.pad(raw, (padh,padh+input_shape[0]-crop[0]-1,padw,padw+input_shape[1]-crop[1]-1), mode='reflect')
    for h in range(0,H,output_shape[0]-overlap[0]):
        for w in range(0,W,output_shape[1]-overlap[0]):
            raw_tmp = raw_padded[:,:,h:h+input_shape[0],w:w+input_shape[1]].to(device)
            with torch.no_grad():
                out_tmp = test_time_aug(raw_tmp, model, flip=flip, rotate=rotate).cpu()
                out_tmp = center_crop(out_tmp, output_shape[0], output_shape[1])
            if keys:
                for key in keys:
                    out_dict[key][:,:,h:h+output_shape[0],w:w+output_shape[1]] += out_tmp[key]
            else:
                out[:,:,h:h+output_shape[0],w:w+output_shape[1]] += out_tmp
            region_counter[:,:,h:h+output_shape[0],w:w+output_shape[1]] += 1.0
    # divide by region_counter and get rid of padded area
    if keys:
        for key in keys:
            out_dict[key] /= region_counter
            out_dict[key] = out_dict[key][:,:,:H,:W]
        return out_dict
    else:
        out /= region_counter
        out = out[:,:,:H,:W]
        return out, region_counter
