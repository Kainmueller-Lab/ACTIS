from tqdm import tqdm
import random
from evaluate import evaluate_linear_sum_assignment
import os
import numpy as np
import re
import tifffile as tif
import pandas as pd
import copy
import argparse
import scipy.ndimage
from label import label

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def remove_holes(pred_inst):
    out = np.zeros_like(pred_inst)
    for i in np.unique(pred_inst):
        if i == 0:
            continue
        out += scipy.ndimage.binary_fill_holes(pred_inst==i)*i
    return out

def hparam_search(directory, mode):
    print(directory)
    print(mode)

    params = {}
    if 'mouse' in directory:
        params['data'] = 'data/Mouse_n0'
        params['experiment'] = directory
    elif 'dsb' in directory:
        params['data'] = 'data/DSB2018_n0'
        params['experiment'] = directory
    elif 'flywing' in directory:
        params['data'] = 'data/Flywing_n0'
        params['experiment'] = directory
    
    if mode == 'test':
        params['data'] = os.path.join(params['data'], 'test', 'test_data.npz')
        test_data =  np.load(params['data'], allow_pickle=True)
        test_masks = test_data['Y_test']
    elif mode == 'validation':
        params['data'] = os.path.join(params['data'], 'train', 'train_data.npz')
        test_data =  np.load(params['data'], allow_pickle=True)
        test_masks = test_data['Y_val']
        

    pred_path = os.path.join(params['experiment'], mode)
    fnames = os.listdir(pred_path)
    fnames = natural_sort(fnames)
    fnames = [os.path.join(pred_path, fname) for fname in fnames]

    met_template = {
            'avAP19':[],
            'avAP59':[],
            'AP50':[],
            'precision50':[],
            'recall50':[],
            'fscore50':[],
            'AP60':[],
            'precision60':[],
            'recall60':[],
            'fscore60':[],
            'AP70':[],
            'precision70':[],
            'recall70':[],
            'fscore70':[],
            'AP80':[],
            'precision80':[],
            'recall80':[],
            'fscore80':[],
            'AP90':[],
            'precision90':[],
            'recall90':[],
            'fscore90':[],}
    met_aggregated = copy.deepcopy(met_template)
    met_aggregated['fg_thresh'] = []
    met_aggregated['seed_thresh'] = []
    seed_thresh = 0.6
    for fg_thresh in np.arange(0.2,0.98,0.01):
        met = copy.deepcopy(met_template)
        for fname, gt_inst in tqdm(zip(fnames, test_masks)):
            pred = tif.imread(fname)
            pred = np.transpose(pred, [2,0,1])
            labeled = label(pred, fg_thresh, seed_thresh)
            labeled = remove_holes(labeled)
            metrics = evaluate_linear_sum_assignment(gt_inst, labeled, 'metrics')
            #
            met['avAP19'].append(metrics['confusion_matrix']['avAP19'])
            met['avAP59'].append(metrics['confusion_matrix']['avAP59'])
            met['AP50'].append(metrics['confusion_matrix']['th_0_5']['AP'])
            met['precision50'].append(metrics['confusion_matrix']['th_0_5']['precision'])
            met['recall50'].append(metrics['confusion_matrix']['th_0_5']['recall'])
            met['fscore50'].append(metrics['confusion_matrix']['th_0_5']['fscore'])
            met['AP60'].append(metrics['confusion_matrix']['th_0_6']['AP'])
            met['precision60'].append(metrics['confusion_matrix']['th_0_6']['precision'])
            met['recall60'].append(metrics['confusion_matrix']['th_0_6']['recall'])
            met['fscore60'].append(metrics['confusion_matrix']['th_0_6']['fscore'])
            met['AP70'].append(metrics['confusion_matrix']['th_0_7']['AP'])
            met['precision70'].append(metrics['confusion_matrix']['th_0_7']['precision'])
            met['recall70'].append(metrics['confusion_matrix']['th_0_7']['recall'])
            met['fscore70'].append(metrics['confusion_matrix']['th_0_7']['fscore'])
            met['AP80'].append(metrics['confusion_matrix']['th_0_8']['AP'])
            met['precision80'].append(metrics['confusion_matrix']['th_0_8']['precision'])
            met['recall80'].append(metrics['confusion_matrix']['th_0_8']['recall'])
            met['fscore80'].append(metrics['confusion_matrix']['th_0_8']['fscore'])
            met['AP90'].append(metrics['confusion_matrix']['th_0_9']['AP'])
            met['precision90'].append(metrics['confusion_matrix']['th_0_9']['precision'])
            met['recall90'].append(metrics['confusion_matrix']['th_0_9']['recall'])
            met['fscore90'].append(metrics['confusion_matrix']['th_0_9']['fscore'])
        #
        for key in met.keys():
            met[key] = np.mean(met[key])
            met_aggregated[key].append(met[key])
        met_aggregated['fg_thresh'].append(fg_thresh)
        met_aggregated['seed_thresh'].append(seed_thresh)
        # 
    # find best fg_thresh
    best_fg_thresh = met_aggregated['fg_thresh'][np.argmax(met_aggregated['avAP19'])]

    for seed_thresh in np.arange(0.2, 0.98, 0.01):
        met = copy.deepcopy(met_template)
        for fname, gt_inst in tqdm(zip(fnames, test_masks)):
            pred = tif.imread(fname)
            pred = np.transpose(pred, [2,0,1])
            labeled = label(pred, best_fg_thresh, seed_thresh)
            labeled = remove_holes(labeled)
            metrics = evaluate_linear_sum_assignment(gt_inst, labeled, 'metrics')
            #
            #
            met['avAP19'].append(metrics['confusion_matrix']['avAP19'])
            met['avAP59'].append(metrics['confusion_matrix']['avAP59'])
            met['AP50'].append(metrics['confusion_matrix']['th_0_5']['AP'])
            met['precision50'].append(metrics['confusion_matrix']['th_0_5']['precision'])
            met['recall50'].append(metrics['confusion_matrix']['th_0_5']['recall'])
            met['fscore50'].append(metrics['confusion_matrix']['th_0_5']['fscore'])
            met['AP60'].append(metrics['confusion_matrix']['th_0_6']['AP'])
            met['precision60'].append(metrics['confusion_matrix']['th_0_6']['precision'])
            met['recall60'].append(metrics['confusion_matrix']['th_0_6']['recall'])
            met['fscore60'].append(metrics['confusion_matrix']['th_0_6']['fscore'])
            met['AP70'].append(metrics['confusion_matrix']['th_0_7']['AP'])
            met['precision70'].append(metrics['confusion_matrix']['th_0_7']['precision'])
            met['recall70'].append(metrics['confusion_matrix']['th_0_7']['recall'])
            met['fscore70'].append(metrics['confusion_matrix']['th_0_7']['fscore'])
            met['AP80'].append(metrics['confusion_matrix']['th_0_8']['AP'])
            met['precision80'].append(metrics['confusion_matrix']['th_0_8']['precision'])
            met['recall80'].append(metrics['confusion_matrix']['th_0_8']['recall'])
            met['fscore80'].append(metrics['confusion_matrix']['th_0_8']['fscore'])
            met['AP90'].append(metrics['confusion_matrix']['th_0_9']['AP'])
            met['precision90'].append(metrics['confusion_matrix']['th_0_9']['precision'])
            met['recall90'].append(metrics['confusion_matrix']['th_0_9']['recall'])
            met['fscore90'].append(metrics['confusion_matrix']['th_0_9']['fscore'])
        #
        for key in met.keys():
            met[key] = np.mean(met[key])
            met_aggregated[key].append(met[key])
        met_aggregated['fg_thresh'].append(best_fg_thresh)
        met_aggregated['seed_thresh'].append(seed_thresh)
        #
        pd.DataFrame(met_aggregated).to_csv(os.path.join(directory, mode+'_metrics.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hparam search')
    parser.add_argument('--mode', type=str, default="validation", help='train or validation')
    parser.add_argument('--checkpoint_path', type=str, default="exp_0_dsb", help='experiment name')
    args = parser.parse_args()
    mode = args.mode
    directory = args.checkpoint_path
    hparam_search(directory, mode)