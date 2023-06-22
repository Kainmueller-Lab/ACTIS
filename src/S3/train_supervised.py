import os
from types import SimpleNamespace

from S3.utils.parameter import Parameter

os.environ["OMP_NUM_THREADS"] = "1"
from torch.utils.data import DataLoader
from S3.utils.data import *
import numpy as np
from S3.nn.unet import *
from S3.utils.data import save_model
from torch.utils.tensorboard import SummaryWriter
import time
import segmentation_models_pytorch as smp
import toml
from S3.utils.augmentations import prep_intensity_aug_fn, prep_spatial_aug_fn
from torchvision import datapoints as DP
from evaluate import calculate_scores

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

params = {
    'base_dir': '/fast/AG_Kainmueller/jrumber/PhD/semi_supervised_IS',
    'data': 'data/DSB2018_n0/train/train_data.npz',  # DSB data, 10 samples 10 19 38 76 152
    # 'data': 'data/Mouse_n0/train/train_data.npz', # Mouse data, 5 samples 5 10 19 38 76
    # 'data': 'data/Flywing_n0/train/train_data.npz', # Flywing data, 5 samples 5 10 19 38 76
    'experiment': 'exp_0_dsb_seed1_samples10',
    'batch_size_labeled': 20,
    'num_workers': 6,
    'training_steps': 100000,
    'in_channels': 1,
    'num_fmaps': 32,
    'fmap_inc_factors': 2,
    'downsample_factors': [[2, 2, ], [2, 2, ], [2, 2, ], [2, 2, ], ],
    'num_fmaps_out': 3,
    'constant_upsample': False,
    'padding': 'same',
    'activation': 'ReLU',
    'learning_rate': 2.0e-2,
    'num_annotated': 10,
    'seed': 1,
    'pretrained_model': True,
    'aug_params': {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1, 0.1), 'scale': (0.5, 1.5), 'shear': 0.2, },
                         "p": 0.25},
        'ElasticTransform': {"kwargs": {'alpha': [120.0, 120.0], 'sigma': 8.0}, "p": 0.25},
    }
}


def train_supervised(args):
    params = args.param

    run = None
    if args.wandb:
        run = wandb_init(args)

    global device, step
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    params['data'] = os.path.join(params['base_dir'], params['data'])
    X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks = prepare_data(params)  # BHW
    loss_weight = torch.Tensor([1.0, 1.0, 4.0]).to(device)
    if 'Flywing' in params['data']:
        print('Set interior to zero')
        Y_labeled[Y_labeled == 1] = 0
        loss_weight = torch.Tensor([0.0, 1.0, 4.0]).to(device)
    X_labeled, Y_labeled, X_unlabeled, X_val, Y_val = [
        d[:, np.newaxis, ...] for d in [X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks]
    ]
    Y_labeled = convert_to_oneHot(Y_labeled)
    Y_labeled = Y_labeled.argmax(-1).squeeze(1)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2 * 16) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)  # todo: import error
        torch.manual_seed(worker_seed)

    # TODO put data into TensorDataset and put whole dataset on device
    # implement normalization in torch instead of numpy
    labeled_dataset = SliceDataset(
        raw=X_labeled.repeat(100, axis=0),
        labels=Y_labeled.repeat(100, axis=0)
    )
    labeled_dataloader = DataLoader(labeled_dataset,
                                    batch_size=params['batch_size_labeled'],
                                    prefetch_factor=32 if params['num_workers'] > 1 else None,
                                    sampler=torch.utils.data.RandomSampler(
                                        [1.0] * len(labeled_dataset),
                                        num_samples=params['num_annotated'],
                                        replacement=True
                                    ),
                                    worker_init_fn=worker_init_fn,
                                    num_workers=params['num_workers'] // 2)
    validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=20,
                                       shuffle=False,
                                       prefetch_factor=32 if params['num_workers'] > 1 else None,
                                       num_workers=params['num_workers'] // 2)
    if params['pretrained_model']:
        model = smp.Unet(
            encoder_name="timm-efficientnet-b5",  # "timm-efficientnet-b5", # choose encoder
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=params['in_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['num_fmaps_out'],  # model output channels (number of classes in your dataset)
        ).to(params['device'])
    else:
        model = UNet(
            in_channels=params['in_channels'],
            num_fmaps=params['num_fmaps'],
            fmap_inc_factor=params['fmap_inc_factors'],
            downsample_factors=params['downsample_factors'],
            activation=params['activation'],
            padding=params['padding'],
            num_fmaps_out=params['num_fmaps_out'],
            constant_upsample=params['constant_upsample']
        ).to(params['device'])

    if args.wandb:
        wandb.watch(model, log_freq=100)

    model = model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=1e-4,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['training_steps'], eta_min=1e-5)
    s_augment = prep_spatial_aug_fn(params['aug_params'])
    c_augment = prep_intensity_aug_fn(128, s=0.5)
    loss_weight = torch.Tensor([1.0, 1.0, 4.0]).to(device)
    exp_dir = os.path.join(params['base_dir'], 'experiments', params['experiment'])
    snap_dir = os.path.join(exp_dir, 'train', 'snaps')
    checkpoint_dir = os.path.join(exp_dir, 'train', 'checkpoints')
    params["checkpoint_dir"] = checkpoint_dir
    writer_dir = os.path.join(exp_dir, 'train', 'summary', str(time.time()))
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)
    with open(os.path.join(exp_dir, 'params.toml'), 'w') as f:
        toml.dump(params, f)
    val_ap50 = []
    step = -1
    while step < params['training_steps']:
        tmp_loader = iter(labeled_dataloader)
        for raw_labeled, gt_3c in tmp_loader:
            if step == params['training_steps']:
                break
            step += 1
            for param in model.parameters():
                param.grad = None
            gt_3c = gt_3c.to(device)  # add channel dimension
            raw_labeled = raw_labeled.to(device)
            raw_labeled = DP.Image(raw_labeled)
            gt_3c = DP.Mask(gt_3c)
            raw_labeled_aug = []
            gt_3c_aug = []
            #
            for raw_labeled_tmp, gt_3c_tmp in zip(raw_labeled, gt_3c):
                tmp = torch.cat([raw_labeled_tmp, gt_3c_tmp.unsqueeze(0)], dim=0)
                tmp = s_augment(tmp)
                raw_labeled_tmp = tmp[:raw_labeled_tmp.shape[0]]
                gt_3c_tmp = tmp[raw_labeled_tmp.shape[0]:]
                gt_3c_tmp = gt_3c_tmp.squeeze(0)
                raw_labeled_aug.append(raw_labeled_tmp)
                gt_3c_aug.append(gt_3c_tmp)
            raw_labeled_aug = torch.stack(raw_labeled_aug)
            gt_3c_aug = torch.stack(gt_3c_aug)
            raw_labeled_aug = c_augment(raw_labeled_aug)
            out_labeled = model(raw_labeled_aug)
            ce_loss_img = F.cross_entropy(
                input=out_labeled,
                target=gt_3c_aug.squeeze(0).long(),
                weight=loss_weight,
                reduction='none'
            )
            ce_loss_img *= (gt_3c_aug != 0).to(float)
            ce_loss = ce_loss_img.mean()
            writer.add_scalar('ce_loss', ce_loss.item(), step)
            print('step: ', step, 'ce_loss: ', ce_loss.cpu().item())
            ce_loss.backward()
            optimizer.step()
            scheduler.step()
            # save snaps
            if step % 1000 == 0:
                # save training snap
                out_dict = {}
                out_dict['raw_labeled'] = raw_labeled_aug[0, ...].cpu().detach().numpy()
                out_dict['pred_labeled'] = out_labeled[0, ...].softmax(0).squeeze().cpu().detach().numpy()
                out_dict['label'] = gt_3c_aug[0, ...].cpu().unsqueeze(0).detach().numpy()
                out_dict['ce_loss'] = ce_loss_img[0, ...].cpu().unsqueeze(0).detach().numpy()
                with h5py.File(os.path.join(snap_dir, 'snap_step_' + str(step) + '.hdf'), 'w') as f:
                    for key in list(out_dict.keys()):
                        f.create_dataset(key, data=out_dict[key].astype(np.float32))
            if step % 10000 == 0:
                # save model
                save_model(step, model, optimizer, checkpoint_dir,
                           os.path.join(checkpoint_dir, 'model_step_' + str(step) + '.pth'))

            if step % 100 == 0:
                out_list = []
                gt_list = []
                padding = 16
                for raw, gt in validation_dataloader:
                    raw = raw.to(device)
                    raw = F.pad(raw, [padding, padding, padding, padding], mode="reflect")
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
                               os.path.join(checkpoint_dir, "best_model.pth"))


if __name__ == '__main__':
    args = SimpleNamespace(param=Parameter(params), wandb=False)
    train_supervised(args)
