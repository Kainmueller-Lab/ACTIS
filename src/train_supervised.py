import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import torch
from torch.utils.data import DataLoader
import random
import h5py
import torch.nn.functional as F
from data_utils import *
import numpy as np
from data_utils import save_model
from torch.utils.tensorboard import SummaryWriter
import time
import segmentation_models_pytorch as smp
from torchvision import datapoints as DP
from evaluate import calculate_scores
from torchvision.transforms import InterpolationMode
import argparse
import toml
torch.backends.cudnn.benchmark = True


def supervised_training(params):
    params['aug_params'] = {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1,0.1), 'scale': (0.5,1.5), 'shear': 0.2,}, "p": 0.25},
        'ElasticTransform': {"kwargs": {'alpha': [120.0,120.0], 'sigma': 8.0}, "p": 0.25},
    }
    params['aug_params_label'] = {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1,0.1), 'scale': (0.5,1.5), 'shear': 0.2,}, "p": 0.25,
                         "interpolation":InterpolationMode.NEAREST},
        'ElasticTransform': {"kwargs": {'alpha': [120.0,120.0], 'sigma': 8.0}, "p": 0.25,
                            "interpolation":InterpolationMode.NEAREST},
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    params['data'] = os.path.join(params['base_dir'], params['data'])

    X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks = prepare_data(params) # BHW
    loss_weight = torch.Tensor([1.0,1.0,4.0]).to(device)

    if 'Flywing' in params['data']:
        print('Set interior to zero')
        Y_labeled[Y_labeled==1] = 0
        
    X_labeled, Y_labeled, X_unlabeled, X_val, Y_val = [
        d[:,np.newaxis,...] for d in [X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks]
    ]

    Y_labeled = convert_to_oneHot(Y_labeled)
    Y_labeled = Y_labeled.argmax(-1).squeeze(1)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2*16) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    labeled_dataset = SliceDataset(
        raw=X_labeled.repeat(100, axis=0),
        labels=Y_labeled.repeat(100, axis=0),
        )
    labeled_dataloader = DataLoader(labeled_dataset,
                        batch_size=params['batch_size_labeled'],
                        prefetch_factor=32 if params['num_workers'] > 1 else None,
                        sampler=torch.utils.data.RandomSampler(
                            [1.0]*len(labeled_dataset),
                            num_samples=params['num_annotated'],
                            replacement=True
                        ),
                        worker_init_fn=worker_init_fn,
                        num_workers=params['num_workers']//2,
                        drop_last=True)

    validation_dataset = SliceDataset(
        raw=X_val,
        labels=Y_val,
    )
    validation_dataloader = DataLoader(validation_dataset,
                        batch_size=20,
                        shuffle=False,
                        prefetch_factor=32 if params['num_workers'] > 1 else None,
                        num_workers=params['num_workers']//2)


    model = smp.Unet(
        encoder_name= "timm-efficientnet-b5", 
        encoder_weights=params["encoder_weights"],
        in_channels=params['in_channels'],
        classes=params['num_fmaps_out'],
        ).to(params['device'])
    model = model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['learning_rate'], weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['training_steps'], eta_min=1e-5
    )

    s_augment = prep_spatial_aug_fn(params['aug_params'])
    s_augment_label = prep_spatial_aug_fn(params['aug_params_label'])
    c_augment = mono_color_augmentations(params['size'], params['s'])

    exp_dir = os.path.join(params['base_dir'], 'experiments', params['experiment'])
    snap_dir = os.path.join(exp_dir,'train','snaps')
    checkpoint_dir = os.path.join(exp_dir,'train','checkpoints')
    params["checkpoint_dir"] = checkpoint_dir
    writer_dir = os.path.join(exp_dir, 'train','summary',str(time.time()))
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(writer_dir,exist_ok=True)
    os.makedirs(checkpoint_dir,exist_ok=True)
    writer = SummaryWriter(writer_dir)

    with open(os.path.join(exp_dir, 'params.toml'), 'w') as f:
        toml.dump(params, f)

    val_ap50 = []
    step = 0
    while step<params['training_steps']:
        tmp_loader = iter(labeled_dataloader)
        for raw_labeled, gt_3c in tmp_loader:
            if step == params['training_steps']:
                break
            step += 1
            for param in model.parameters():
                param.grad = None
            gt_3c = gt_3c.to(device) # add channel dimension
            raw_labeled = raw_labeled.to(device)
            raw_labeled = DP.Image(raw_labeled)
            gt_3c = DP.Mask(gt_3c.to(torch.int32))
            raw_labeled_aug = []
            gt_3c_aug = []
            # 
            for raw_labeled_tmp, gt_3c_tmp in zip(raw_labeled, gt_3c):
                # apply the same augmentations on raw and gt
                state = torch.get_rng_state()
                raw_labeled_tmp = s_augment(raw_labeled_tmp)
                torch.set_rng_state(state)
                gt_3c_tmp = s_augment_label(gt_3c_tmp.unsqueeze(0))
                gt_3c_tmp = gt_3c_tmp.squeeze(0)
                raw_labeled_aug.append(raw_labeled_tmp)
                gt_3c_aug.append(gt_3c_tmp)
            raw_labeled_aug = torch.stack(raw_labeled_aug)
            gt_3c_aug = torch.stack(gt_3c_aug)
            raw_labeled_aug = c_augment(raw_labeled_aug)
            out_labeled = model(raw_labeled_aug)
            ce_loss_img = F.cross_entropy(
                input = out_labeled,
                target = gt_3c_aug.squeeze(0).long(), 
                weight = loss_weight,
                reduction = 'none',
                label_smoothing = 0.05
            )
            ce_loss = ce_loss_img.mean()
            writer.add_scalar('ce_loss', ce_loss.item(), step)
            print('step: ', step, 'ce_loss: ', ce_loss.cpu().item())
            ce_loss.backward()
            optimizer.step()
            scheduler.step()
            # save snaps
            if step % 1000 ==0:
                # save training snap
                out_dict = {}
                out_dict['raw_labeled'] = raw_labeled_aug[0,...].cpu().detach().numpy()
                out_dict['pred_labeled'] = out_labeled[0,...].softmax(0).squeeze().cpu().detach().numpy()
                out_dict['label'] = gt_3c_aug[0,...].cpu().unsqueeze(0).detach().numpy()
                out_dict['ce_loss'] = ce_loss_img[0,...].cpu().unsqueeze(0).detach().numpy()
                with h5py.File(os.path.join(snap_dir,'snap_step_'+str(step)+'.hdf'),'w') as f:
                    for key in list(out_dict.keys()):
                        f.create_dataset(key, data = out_dict[key].astype(np.float32))
            if step % 10000 == 0:
                # save model
                save_model(step, model, optimizer, checkpoint_dir,
                    os.path.join(checkpoint_dir, 'model_step_'+str(step)+'.pth')
                )
            if step % 200 == 0:
                out_list = []
                gt_list = []
                padding = 16
                for raw, gt in validation_dataloader:
                    raw = raw.to(device)
                    raw = F.pad(raw, [padding,padding,padding,padding], mode="reflect")
                    with torch.no_grad():
                        out = model(raw)
                    out = out[..., padding:-padding, padding:-padding]
                    out_list += out.cpu().split(1)
                    gt_list += gt.squeeze(1).split(1)
                metric_dict = calculate_scores(out_list, gt_list, "None")

                val_ap50.append(metric_dict["ap_50"])
                print("Validation:")
                print(metric_dict)
                for key in metric_dict.keys():
                    writer.add_scalar(key, metric_dict[key], step)
                if metric_dict["ap_50"] >= np.max(val_ap50):
                    print('Save best model')
                    save_model(step, model, optimizer, metric_dict["ap_50"],
                               os.path.join(checkpoint_dir, "best_model.pth")
                            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument('--params', type=str, default="configs/params_supervised.toml", help='Path to params.toml file')
    args = parser.parse_args()
    params = toml.load(args.params)
    supervised_training(params)
    sys.subprocess.call(
        ["python", "evaluate_experiment.py", "--experiment", params["experiment"]]
    )