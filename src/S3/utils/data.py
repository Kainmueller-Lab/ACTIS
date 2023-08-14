import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.transforms import RandomApply, GaussianBlur, ColorJitter
from unet import *


def save_model(step, model, optimizer, loss, filename):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, img):
        device = img.device
        noise = torch.randn(img.shape).to(device) * self.sigma
        return img + noise


def prep_spatial_aug_fn(aug_params):
    aug_list = []
    for key in aug_params.keys():
        if "kwargs" in aug_params[key].keys():
            aug_list.append(getattr(transforms, key)(**aug_params[key]['kwargs']))
        else:
            aug_list.append(getattr(transforms, key)(**aug_params[key]))

    aug_fn = transforms.Compose(aug_list)
    return aug_fn


def test_time_aug(input, model, flip=True, rotate=True, eval=True):
    # input: [B, C, H, W]
    if eval:
        model = model.eval()
    else:
        model = model.train()
    B, C, H, W = input.shape
    input_list = []
    if rotate:
        for k in [1, 2, 3]:
            input_list.append(
                torch.rot90(input, k, [2, 3])
            )
    if flip:
        for k in [0, 1]:
            input_list.append(
                torch.flip(input, [2 + k])
            )
    input_list.append(input)
    input_aug = torch.cat(input_list, dim=0)
    with torch.no_grad():
        output_aug = model(input_aug)
    output_aug_split = output_aug.split(B)
    i = 0
    output_list = []
    if rotate:
        for k in [-1, -2, -3]:
            output_list.append(
                torch.rot90(output_aug_split[i], k, [2, 3])
            )
            i += 1
    if flip:
        for k in [0, 1]:
            output_list.append(
                torch.flip(output_aug_split[i], [2 + k])
            )
            i += 1
    output_list.append(output_aug[-B:])
    output = torch.stack(output_list, dim=0).mean(dim=0)
    model = model.train()
    return output


def color_augmentations(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = torch.nn.Sequential(
        RandomApply([color_jitter,
                     GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.2, 0.2))], p=0.5),
        GaussianNoise(0.03)
    )
    return data_transforms


def center_crop(t, croph, cropw):
    _, _, h, w = t.shape
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[:, :, starth:starth + croph, startw:startw + cropw]


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
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

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

    b, c, h, w = tensor.shape

    t = (xx - h) // 2
    b = xx - t - h

    l = (yy - w) // 2
    r = yy - l - w
    t, b, l, r = [np.abs(i) for i in [t, b, l, r]]
    return F.pad(tensor, (r, l, b, t, 0, 0, 0, 0))


class SliceDataset(Dataset):
    def __init__(self, raw, labels, norm_axis=None):
        super().__init__()
        self.raw = raw
        self.labels = labels
        self.norm_axis = norm_axis

    def __len__(self):
        if isinstance(self.raw, list):
            return len(self.raw)
        else:
            return self.raw.shape[0]

    def process(self, raw):
        raw_tmp = normalize_percentile(raw, axis=self.norm_axis)
        return raw_tmp

    def __getitem__(self, idx):
        raw_tmp = self.process(self.raw[idx].astype(np.float32))
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
        if (np.abs(np.max(data[i])) <= eps):
            data_oneHot[i][..., 0] *= 0

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
    trainval_data = np.load(params['data'])
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
    percentage_of_annotated_training_images = float(
        (number_of_annotated_training_images / train_images.shape[0]) * 100.0)
    assert percentage_of_annotated_training_images >= 0.0 and percentage_of_annotated_training_images <= 100.0

    # Here we zero out the segmentations of those training images which are not part of the selected annotated images.
    X_frac, Y_frac = zero_out_train_data(X_shuffled, Y_shuffled, fraction=percentage_of_annotated_training_images)
    X_labeled, Y_labeled = X_shuffled[:number_of_annotated_training_images], Y_shuffled[
                                                                             :number_of_annotated_training_images]
    X_unlabeled = X_shuffled[number_of_annotated_training_images:]

    X_val, Y_val_masks = val_images, val_masks
    return X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks


def prepare_test_data(params):
    test_data = np.load(params['test_data'], allow_pickle=True)
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
                test_images[i] = np.pad(img, [[0, h_pad], [0, w_pad]])
                test_masks[i] = np.pad(mask, [[0, h_pad], [0, w_pad]])
    return test_images, test_masks


def mono_color_augmentations(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    data_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.01, 0.5)),
            GaussianNoise(0.2 * s)
        ], p=0.75),
    ])
    return data_transforms
