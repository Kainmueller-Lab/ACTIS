import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
from torch.utils.data import DataLoader
import random
import h5py
from S3.utils.data import *
import numpy as np
import copy
from S3.utils.data import save_model
from torch.utils.tensorboard import SummaryWriter
import time
import segmentation_models_pytorch as smp
from torchvision import datapoints as DP
from S3.evaluate import calculate_scores
from torchvision.transforms import InterpolationMode
from S3.nn.loss import QuantileLoss
import argparse
import toml

torch.backends.cudnn.benchmark = True


def semi_supervised_training(params):
    params['aug_params'] = {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1, 0.1), 'scale': (0.5, 1.5), 'shear': 0.2, },
                         "p": 0.25},
        'ElasticTransform': {"kwargs": {'alpha': [120.0, 120.0], 'sigma': 8.0}, "p": 0.25},
    }
    params['aug_params_semi'] = {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
    }
    params['aug_params_label'] = {
        'RandomHorizontalFlip': {'p': 0.25},
        'RandomVerticalFlip': {'p': 0.25},
        'RandomAffine': {"kwargs": {'degrees': 180, 'translate': (0.1, 0.1), 'scale': (0.5, 1.5), 'shear': 0.2, },
                         "p": 0.25,
                         "interpolation": InterpolationMode.NEAREST},
        'ElasticTransform': {"kwargs": {'alpha': [120.0, 120.0], 'sigma': 8.0}, "p": 0.25,
                             "interpolation": InterpolationMode.NEAREST},
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    params['data'] = os.path.join(params['base_dir'], params['data'])

    loss_weight = torch.Tensor([1.0, 1.0, 4.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight, reduction='none', label_smoothing=0.05)
    params['loss_fn_QuantLoss'] = loss_fn
    loss_fn_semi = QuantileLoss(params, nclasses=3)

    X_labeled, Y_labeled, X_unlabeled, X_val, Y_val_masks = prepare_data(params)  # BHW

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
                                        [1.0] * len(labeled_dataset),
                                        num_samples=len(labeled_dataset),
                                        replacement=True
                                    ),
                                    worker_init_fn=worker_init_fn,
                                    num_workers=params['num_workers'] // 2,
                                    drop_last=True)

    unlabeled_dataset = SliceDataset(
        raw=X_unlabeled,
        labels=None,
    )
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=params['batch_size_unlabeled'],
                                      prefetch_factor=32 if params['num_workers'] > 1 else None,
                                      sampler=torch.utils.data.RandomSampler(
                                          [1.0] * len(unlabeled_dataset),
                                          num_samples=params['training_steps'],
                                          replacement=True
                                      ),
                                      worker_init_fn=worker_init_fn,
                                      num_workers=params['num_workers'] // 2)

    validation_dataset = SliceDataset(
        raw=X_val,
        labels=Y_val,
    )
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=20,
                                       shuffle=False,
                                       prefetch_factor=32 if params['num_workers'] > 1 else None,
                                       num_workers=params['num_workers'] // 2)

    model = smp.Unet(
        encoder_name="timm-efficientnet-b5",  # "timm-efficientnet-b5", # choose encoder
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=params['in_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=params['num_fmaps_out'],  # model output channels (number of classes in your dataset)
    ).to(params['device'])

    if params['load_student_weights']:
        model.load_state_dict(torch.load(params['checkpoint_path'])['model_state_dict'])
    model = model.train()
    slow_model = copy.deepcopy(model)
    slow_model.load_state_dict(torch.load(params['checkpoint_path'])['model_state_dict'])
    slow_model = slow_model.train()

    if "no_moco" in params.keys() and params["no_moco"]:
        del slow_model
        slow_model = model

    optimizer = torch.optim.SGD(
        model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=1e-3, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['training_steps'], eta_min=1e-5
    )

    s_augment = prep_spatial_aug_fn(params['aug_params'])
    s_augment_semi = prep_spatial_aug_fn(params['aug_params_semi'])
    s_augment_label = prep_spatial_aug_fn(params['aug_params_label'])
    c_augment = mono_color_augmentations(params['size'], params['s'])

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
    val_ap50_slow = []
    step = 0

    while step < params['training_steps']:
        for (raw_labeled, gt_3c), raw_unlabeled in zip(labeled_dataloader, unlabeled_dataloader):
            if step > params['training_steps']:
                break
            step += 1
            for param in model.parameters():
                param.grad = None
            gt_3c = gt_3c.to(device)
            raw_labeled = raw_labeled.to(device)
            raw_unlabeled = raw_unlabeled.to(device)
            pseudo_labels = test_time_aug(raw_unlabeled, slow_model, flip=True, rotate=False)
            raw_labeled = DP.Image(raw_labeled)
            raw_unlabeled = DP.Image(raw_unlabeled)
            pseudo_labels = DP.Image(pseudo_labels)
            gt_3c = DP.Mask(gt_3c.to(torch.int32))

            # apply individual transforms on each sample
            raw_labeled_aug = []
            raw_unlabeled_aug = []
            gt_3c_aug = []
            pseudo_labels_aug = []
            for raw_labeled_tmp, gt_3c_tmp in zip(raw_labeled, gt_3c):
                state = torch.get_rng_state()
                raw_labeled_tmp = s_augment(raw_labeled_tmp)
                torch.set_rng_state(state)
                gt_3c_tmp = s_augment_label(gt_3c_tmp.unsqueeze(0))
                gt_3c_tmp = gt_3c_tmp.squeeze(0)
                raw_labeled_aug.append(raw_labeled_tmp)
                gt_3c_aug.append(gt_3c_tmp)
            #
            for raw_unlabeled_tmp, pseudo_labels_tmp in zip(raw_unlabeled, pseudo_labels):
                state = torch.get_rng_state()
                raw_unlabeled_tmp = s_augment_semi(raw_unlabeled_tmp)
                torch.set_rng_state(state)
                pseudo_labels_tmp = s_augment_semi(pseudo_labels_tmp)
                raw_unlabeled_aug.append(raw_unlabeled_tmp)
                pseudo_labels_aug.append(pseudo_labels_tmp)
            raw_labeled_aug = torch.stack(raw_labeled_aug)
            raw_unlabeled_aug = torch.stack(raw_unlabeled_aug)
            gt_3c_aug = torch.stack(gt_3c_aug)
            pseudo_labels_aug = torch.stack(pseudo_labels_aug)
            #
            pseudo_labels_aug_mask = pseudo_labels_aug != 0
            raw_labeled_aug = c_augment(raw_labeled_aug)
            raw_unlabeled_aug = c_augment(raw_unlabeled_aug)

            # concat labeled and unlabeled data + forward pass
            raw_aug = torch.cat([raw_labeled_aug, raw_unlabeled_aug], dim=0)
            model_out = model(raw_aug)

            # split into labeled and unlabeled data
            out_labeled, out_unlabeled = torch.split(
                model_out, [params['batch_size_labeled'], params['batch_size_unlabeled']], dim=0
            )

            # loss_semi 
            loss_img_semi = loss_fn_semi(
                input=out_unlabeled,
                target=pseudo_labels_aug.detach(),
                step=step
            )
            loss_img_semi = loss_img_semi.mean(1)
            # mask out areas that were padded during augmentation
            loss_img_semi *= pseudo_labels_aug_mask.any(1).to(float)

            con_loss = loss_img_semi.mean() * params['semi_loss_weight']
            writer.add_scalar('aug_con_loss', con_loss.item(), step)

            ce_loss_img = loss_fn(
                input=out_labeled,
                target=gt_3c_aug.squeeze(0).long(),
            )
            ce_loss = ce_loss_img.mean()
            writer.add_scalar('ce_loss', ce_loss.item(), step)
            loss = con_loss + ce_loss
            print('step: ', step, 'aug_con_loss: ', con_loss.item(), 'ce_loss: ', ce_loss.cpu().item(), "total loss: ",
                  loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # save snaps
            if step % 1000 == 0:
                # save training snap
                out_dict = {}
                out_dict['raw_labeled'] = raw_labeled_aug[0, ...].cpu().detach().numpy()
                out_dict['pred_labeled'] = out_labeled[0, ...].softmax(0).squeeze().cpu().detach().numpy()
                out_dict['label'] = gt_3c_aug[0, ...].cpu().unsqueeze(0).detach().numpy()
                out_dict['raw_unlabeled'] = raw_unlabeled[0, ...].cpu().detach().numpy()
                out_dict['raw_unlabeled_aug'] = raw_unlabeled_aug[0, ...].cpu().detach().numpy()
                out_dict['pseudo_label'] = pseudo_labels_aug[0, ...].softmax(0).cpu().detach().numpy()
                out_dict['pred_unlabeled'] = out_unlabeled[0, ...].softmax(0).squeeze().cpu().detach().numpy()
                out_dict['ce_loss'] = ce_loss_img[0, ...].cpu().unsqueeze(0).detach().numpy()
                out_dict['con_loss'] = loss_img_semi[0, ...].cpu().unsqueeze(0).detach().numpy()
                out_dict["loss_mask"] = loss_fn_semi.selected_masks[0, ...].cpu().unsqueeze(0).detach().numpy()
                with h5py.File(os.path.join(snap_dir, 'snap_step_' + str(step) + '.hdf'), 'w') as f:
                    for key in list(out_dict.keys()):
                        f.create_dataset(key, data=out_dict[key].astype(np.float32))
            if step % 100 == 0 and step > params['warmup_steps']:
                print('Update Momentum Network')
                for param_slow, param_fast in zip(slow_model.parameters(), model.parameters()):
                    param_slow = 0.99 * param_slow + 0.01 * param_fast
                    writer.add_scalar("semi_quantile", loss_fn_semi.quantile, step)
            if step % 10000 == 0:
                # save model
                save_model(step, model, optimizer, checkpoint_dir,
                           os.path.join(checkpoint_dir, 'model_step_' + str(step) + '.pth'))
            if step % 200 == 0:
                out_list = []
                out_list_slow = []
                gt_list = []
                padding = 16
                for raw, gt in validation_dataloader:
                    raw = raw.to(device)
                    raw = F.pad(raw, [padding, padding, padding, padding], mode="reflect")
                    with torch.no_grad():
                        out = model(raw)
                        out_slow = slow_model(raw)
                    out = out[..., padding:-padding, padding:-padding]
                    out_slow = out_slow[..., padding:-padding, padding:-padding]
                    out_list += out.cpu().split(1)
                    out_list_slow += out_slow.cpu().split(1)
                    gt_list += gt.squeeze(1).split(1)
                metric_dict = calculate_scores(out_list, gt_list, "None")
                metric_dict_slow = calculate_scores(out_list_slow, gt_list, "None")

                val_ap50.append(metric_dict["ap_50"])
                val_ap50_slow.append(metric_dict_slow["ap_50"])
                print("Validation:")
                print(metric_dict)
                for key in metric_dict.keys():
                    writer.add_scalar(key, metric_dict[key], step)
                for key in metric_dict_slow.keys():
                    writer.add_scalar(key + "_slow", metric_dict_slow[key], step)
                if metric_dict["ap_50"] >= np.max(val_ap50):
                    print('Save best model')
                    save_model(step, model, optimizer, loss, os.path.join(checkpoint_dir, "best_model.pth"))
                    if metric_dict["ap_50"] > metric_dict_slow["ap_50"] and params['fast_update_slow_model']:
                        slow_model = copy.deepcopy(model)
                if metric_dict_slow["ap_50"] >= np.max(val_ap50_slow):
                    print('Save best slow model')
                    save_model(step, slow_model, optimizer, checkpoint_dir,
                               os.path.join(checkpoint_dir, "best_slow_model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument('--params', type=str, default="configs/params_semi.toml", help='Path to params.toml file')
    args = parser.parse_args()
    params = toml.load(args.params)
    semi_supervised_training(params)
    sys.subprocess.call(
        ["python", "evaluate_experiment.py", "--experiment", params["experiment"]]
    )
