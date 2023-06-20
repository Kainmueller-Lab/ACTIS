import os
import sys
import toml
import torch
from data_utils import *
from torch.utils.data import DataLoader
from evaluate import calculate_scores
import segmentation_models_pytorch as smp
from torch.nn import functional as F
import pandas as pd
import argparse


def evaluate_experiment(experiment, checkpoint=None):
    params_path = os.path.join("experiments", experiment, "params.toml")
    params = toml.load(params_path)

    if "Flywing" in params["data"]:
        params["test_data"] = '/fast/AG_Kainmueller/jrumber/PhD/semi_supervised_IS/data/Flywing_n0/test/test_data.npz'
    if "DSB2018" in params["data"]:
        params["test_data"] = '/fast/AG_Kainmueller/jrumber/PhD/semi_supervised_IS/data/DSB2018_n0/test/test_data.npz'
    if "Mouse" in params["data"]:
        params["test_data"] = '/fast/AG_Kainmueller/jrumber/PhD/semi_supervised_IS/data/Mouse_n0/test/test_data.npz'

    X_test, Y_test = prepare_test_data(params)
    _, _, _, X_val, Y_val_masks = prepare_data(params) # BHW

    if 'Flywing' in params['test_data']:
        print('Set interior to zero')
        Y_val_masks[Y_val_masks==1] = 0

    X_val, Y_val_masks,  = [
        d[:,np.newaxis,...] for d in [X_val, Y_val_masks]
    ]
    if isinstance(X_test, np.ndarray):
        X_test, Y_test = [
            d[:,np.newaxis,...] for d in [X_test, Y_test]
        ]
    elif isinstance(X_test, list):
        X_test = [
            d[np.newaxis,...] for d in X_test
        ]
        Y_test = [
            d[np.newaxis,...] for d in Y_test
        ]

    validation_dataset = SliceDataset(raw=X_val, labels=Y_val_masks)
    validation_dataloader = DataLoader(validation_dataset,
                        batch_size=20,
                        shuffle=False)

    test_dataset = SliceDataset(raw=X_test, labels=Y_test)
    test_dataloader = DataLoader(test_dataset,
                        batch_size=1 if "DSB2018" in params["test_data"] else 20,
                        shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    print(device)

    model = smp.Unet(
        encoder_name= "timm-efficientnet-b5", # "timm-efficientnet-b5", # choose encoder
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=params['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=params['num_fmaps_out'],          # model output channels (number of classes in your dataset)
        ).to(params['device'])

    exp_dir = os.path.join(params['base_dir'], 'experiments', params['experiment'])
    checkpoint_dir = os.path.join(exp_dir, 'train','checkpoints')

    if checkpoint is None:
        checkpoint = "best_model.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    # inference on validation set
    padding = 16
    out_list = []
    gt_list = []
    for raw, gt in validation_dataloader:
        raw = raw.to(device)
        raw = F.pad(raw, [padding,padding,padding,padding], mode="reflect")
        h, w = raw.shape[-2:]
        with torch.no_grad():
            out = model(raw)
            out = out[..., padding:-padding, padding:-padding]
        out_list += out.cpu().split(1)
        gt_list += gt.squeeze(1).split(1)

    # optimize fg_thresh
    out_dir = os.path.join(exp_dir, 'test', checkpoint.split(".")[0])
    os.makedirs(out_dir, exist_ok=True)
    fg_threshs = np.linspace(0.1,0.9,81)
    seed_thresh = 0.8
    all_scores = {
        'fg_thresh': [],
        'seed_thresh': [],
        'ap_50': [],
        'f1_score': [],
        'avap19': [],
    }
    for fg_thresh in fg_threshs:
        scores = calculate_scores(
            out_list, gt_list, os.path.join(out_dir, "fname"), fg_thresh, seed_thresh
        )
        for key in scores.keys():
            all_scores[key].append(scores[key])
        all_scores['seed_thresh'].append(seed_thresh)
        all_scores['fg_thresh'].append(fg_thresh)
    df = pd.DataFrame(all_scores)
    df.to_csv(os.path.join(out_dir, 'val_scores_fg.csv'))
    best_fg_thresh = df['fg_thresh'][df['ap_50'].argmax()]

    # optimize seed_thresh
    seed_threshs = np.linspace(0.1,0.9,81)
    fg_thresh = best_fg_thresh
    all_scores = {key: [] for key in all_scores.keys()}
    for seed_thresh in seed_threshs:
        scores = calculate_scores(
            out_list, gt_list, os.path.join(out_dir, "fname"), fg_thresh, seed_thresh
        )
        for key in scores.keys():
            all_scores[key].append(scores[key])
        all_scores['seed_thresh'].append(seed_thresh)
        all_scores['fg_thresh'].append(fg_thresh)
    df = pd.DataFrame(all_scores)
    df.to_csv(os.path.join(out_dir, 'val_scores_fg_seed.csv'))
    best_seed_thresh = df['seed_thresh'][df['ap_50'].argmax()]
    print("best fg_thresh: ", best_fg_thresh)
    print("best seed_thresh: ", best_seed_thresh)

    # inference on test set
    # padding = 16
    input_shape = [256, 256]
    out_list = []
    gt_list = []
    for raw, gt in test_dataloader:
        raw = raw.to(device)
        # raw = F.pad(raw, [padding,padding,padding,padding], mode="reflect")
        with torch.no_grad():
            out = tile_and_stitch_ov(
                model, raw, input_shape, device, overlap=(32,32), crop=(16,16), flip=True,
                rotate=False
            )
            out = model(raw)
            # out = out[..., padding:-padding, padding:-padding]
        out_list += out.cpu().split(1)
        gt_list += gt.squeeze(1).split(1)

    # calculate scores
    scores = calculate_scores(
        out_list, gt_list, os.path.join(out_dir, "fname"), best_fg_thresh, best_seed_thresh
    )
    scores['fg_thresh'] = best_fg_thresh
    scores['seed_thresh'] = best_seed_thresh
    print(scores)
    df = pd.DataFrame(scores, index=[0])
    df.to_csv(os.path.join(out_dir, 'test_scores.csv'))

if __name__ == '__main__':
    # read in experiment and checkpoint with argparse
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, default="exp_0_mouse_seed1_samples10_DINO_L1_loss_highLR")
    args.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = args.parse_args()
    evaluate_experiment(args.experiment, args.checkpoint)
 
