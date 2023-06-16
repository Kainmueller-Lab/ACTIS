from torchvision.transforms import v2 as transforms
import torch

class GaussianNoise(transforms.Transform):
    def __init__(self, var):
        super().__init__()
        self.var = var

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img + torch.randn_like(img) * self.var
        else:
            out = []
            for t in img:
                t += torch.randn_like(t) * self.var
                out.append(t)
            return out


def prep_spatial_aug_fn(aug_params):
    aug_list = []
    for key in aug_params.keys():
        if "kwargs" in aug_params[key].keys():
            aug_list.append(getattr(transforms, key)(**aug_params[key]['kwargs']))
        else:
            aug_list.append(getattr(transforms, key)(**aug_params[key]))

    aug_fn = transforms.Compose(aug_list)
    return aug_fn

def prep_intensity_aug_fn(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    data_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.2,0.2)),
            GaussianNoise(0.1*s)
        ], p=0.5),
        ])
    return data_transforms


def test_time_aug(input, model, flip=True, rotate=True):
    # input: [B, C, H, W]
    B, C, H, W = input.shape
    input_list = []
    if rotate:
        for k in [1,2,3]:
            input_list.append(
                torch.rot90(input, k, [2,3])
            )
    if flip:
        for k in [0,1]:
            input_list.append(
                torch.flip(input, [2+k])
            )
    input_list.append(input)
    input_aug = torch.cat(input_list, dim=0)
    with torch.no_grad():
        output_aug = model(input_aug)
    output_aug_split = output_aug.split(B)
    i = 0
    output_list = []
    if rotate:
        for k in [-1,-2,-3]:
            output_list.append(
                torch.rot90(output_aug_split[i], k, [2,3])
            )
            i += 1
    if flip:
        for k in [0,1]:
            output_list.append(
                torch.flip(output_aug_split[i], [2+k])
            )
            i += 1
    output_list.append(output_aug[-B:])
    output = torch.stack(output_list, dim=0).mean(dim=0)
    return output