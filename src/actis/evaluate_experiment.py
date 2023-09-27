import argparse

import pandas as pd
import segmentation_models_pytorch as smp
import toml
from torch.utils.data import DataLoader

from actis.actis_logging import get_logger
from actis.evaluate import calculate_scores
from actis.utils.data import *


def evaluate_experiment(
        experiment, base_dir="",
        checkpoint=None,
        tile_and_stitch=False,
        best_fg_thresh=None,
        best_seed_thresh=None,
        fg_thresh_linspace_num=90,
        seed_thresh_linspace_num=90,
):  # todo: tile_and_stitch argument not used
    """Evaluate an experiment. Writes evaluation metrics to disk.

    Args:
        experiment:
            The name of the experiment to evaluate.
        base_dir:
            The base directory of the experiment. If None, the current working directory is used.
        checkpoint:
            The name of the checkpoint to evaluate. If None, the best checkpoint is evaluated.
        tile_and_stitch:
            todo: update
        best_fg_thresh:
            The best foreground threshold to use for evaluation. If None, the best foreground threshold is determined.
        best_seed_thresh:
            The best seed threshold to use for evaluation. If None, the best seed threshold is determined.
        seed_thresh_linspace_num:
            Number of seed thresholds to evaluate if best best_seed_thresh not given. Default is 90.
        fg_thresh_linspace_num:
            Number of foreground thresholds to evaluate if best_fg_thresh not given. Default is 90.

    """
    params_path = os.path.join(base_dir, "experiments", experiment, "params.toml")
    params = toml.load(params_path)
    get_logger().info(params["experiment"])

    X_test, Y_test = prepare_test_data(params)
    _, _, _, X_val, Y_val_masks = prepare_data(params)  # BHW

    X_val, Y_val_masks, = [
        d[:, np.newaxis, ...] for d in [X_val, Y_val_masks]
    ]
    if isinstance(X_test, np.ndarray):
        X_test, Y_test = [
            d[:, np.newaxis, ...] for d in [X_test, Y_test]
        ]
    elif isinstance(X_test, list):
        X_test = [
            d[np.newaxis, ...] for d in X_test
        ]
        Y_test = [
            d[np.newaxis, ...] for d in Y_test
        ]

    validation_dataset = SliceDataset(
        raw=X_val,
        labels=Y_val_masks,
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=20, shuffle=False, )

    test_dataset = SliceDataset(raw=X_test, labels=Y_test, )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1 if "DSB2018" in params["test_data"] else 20,
        shuffle=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    get_logger().info(device)

    model = smp.Unet(
        encoder_name="timm-efficientnet-b5",  # "timm-efficientnet-b5", # choose encoder
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=params['in_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=params['num_fmaps_out'],  # model output channels (number of classes in your dataset)
    ).to(params['device'])

    exp_dir = os.path.join(params['base_dir'], 'experiments', params['experiment'])
    checkpoint_dir = os.path.join(exp_dir, 'train', 'checkpoints')

    if checkpoint is None:
        checkpoint = "best_model.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model = model.eval()

    # inference on validation set
    padding = 16
    input_shape = [128, 128]
    out_list = []
    gt_list = []
    for raw, gt in validation_dataloader:
        raw = raw.to(device)
        _, _, h, w = raw.shape
        if h % 32 != 0 or w % 32 != 0:
            h_pad = 32 - (h % 32)
            w_pad = 32 - (w % 32)
            raw = F.pad(raw, [0, w_pad, 0, h_pad], "reflect")
            padded = True
        else:
            padded = False
        raw_tmp = F.pad(raw, [padding, padding, padding, padding], mode="reflect")
        with torch.no_grad():
            out = test_time_aug(raw_tmp, model, flip=True, rotate=False).cpu()
        out = out[..., padding:-padding, padding:-padding]
        if padded:
            out = out[..., :-h_pad, :-w_pad]

        out_list += out.cpu().split(1)
        gt_list += gt.squeeze(1).split(1)
    out_dir = os.path.join(exp_dir, 'test', checkpoint.split(".")[0])
    os.makedirs(out_dir, exist_ok=True)

    # optimize fg_thresh
    if not best_fg_thresh:
        fg_threshs = np.linspace(0.1, 0.99, fg_thresh_linspace_num)
        seed_thresh = 0.6
        all_scores = {
            'fg_thresh': [],
            'seed_thresh': [],
            'ap_50': [],
            'ap_50_denoiseg': [],
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
    if not best_seed_thresh:
        seed_threshs = np.linspace(0.1, 0.99, seed_thresh_linspace_num)
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
    get_logger().info("best fg_thresh: %s" % best_fg_thresh)
    get_logger().info("best seed_thresh: %s" % best_seed_thresh)

    # inference on test set
    out_list = []
    gt_list = []
    for raw, gt in test_dataloader:
        raw = raw.to(device)
        _, _, h, w = raw.shape
        if h % 32 != 0 or w % 32 != 0:
            h_pad = 32 - (h % 32)
            w_pad = 32 - (w % 32)
            raw = F.pad(raw, [0, w_pad, 0, h_pad], "reflect")
            padded = True
        else:
            padded = False
        raw_tmp = F.pad(raw, [padding, padding, padding, padding], mode="reflect")
        with torch.no_grad():
            out = test_time_aug(raw_tmp, model, flip=True, rotate=False).cpu()
        out = out[..., padding:-padding, padding:-padding]
        if padded:
            out = out[..., :-h_pad, :-w_pad]

        out_list += out.cpu().split(1)
        gt_list += gt.squeeze(1).split(1)

    # calculate scores
    scores = calculate_scores(
        out_list, gt_list, os.path.join(out_dir, "fname"), best_fg_thresh, best_seed_thresh
    )
    scores['fg_thresh'] = best_fg_thresh
    scores['seed_thresh'] = best_seed_thresh
    get_logger().info(scores)
    df = pd.DataFrame(scores, index=[0])
    df.to_csv(os.path.join(out_dir, 'test_scores.csv'))


def evaluate(args):
    evaluate_experiment(
        args.experiment,
        args.base_dir,
        args.checkpoint,
        args.tile_and_stitch,
        args.best_fg_thresh,
        args.best_seed_thresh,
        args.fg_thresh_linspace_num,
        args.seed_thresh_linspace_num,
    )


if __name__ == '__main__':
    """Evaluate a model on the test set. This parser is used to run the evaluation via script."""
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, default="mouse_seed1_samples76_promix_fast_updates_strong_aug_semi")  # todo: update default
    args.add_argument("--base_dir", type=str, default="")
    args.add_argument("--checkpoint", type=str, default="best_model.pth")
    args.add_argument("--tile_and_stitch", type=bool, default=False)
    args.add_argument("--best_fg_thresh", type=float, default=None)
    args.add_argument("--best_seed_thresh", type=float, default=None)
    args.add_argument("--fg_thresh_linspace_num", type=int, default=90)
    args.add_argument("--seed_thresh_linspace_num", type=int, default=90)

    args = args.parse_args()

    # evaluate the model
    evaluate_experiment(
        args.experiment,
        args.checkpoint,
        args.tile_and_stitch,
        args.best_fg_thresh,
        args.best_seed_thresh,
    )
