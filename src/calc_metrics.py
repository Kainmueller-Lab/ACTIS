
import os
from tqdm import tqdm
import random
from evaluate import evaluate_linear_sum_assignment
import numpy as np
import re
import tifffile as tif
import mahotas
import pandas as pd
import copy
import argparse
import scipy.ndimage

def watershed(surface, markers, fg):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)
    # overlay fg and write
    wsFG = ws * fg
    wsFGUI = wsFG.astype(np.uint16)
    return wsFGUI

def label(prediction, fg_thresh=0.9, seed_thresh=0.9):
    fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
    ws_surface = 1.0 - prediction[1, ...]
    seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
    markers, cnt = scipy.ndimage.label(seeds)
    # compute watershed
    labelling = watershed(ws_surface, markers, fg)
    return labelling

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

#directories = os.listdir()
directories = [
'exp_0_dsb_seed1_samples19_newproj_MSE_softmaxnone_trainableTrue_lw400',
]

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
met_aggregated['experiment'] = []
met_aggregated['data'] = []
met_aggregated['fg_thresh'] = []
met_aggregated['seed_thresh'] = []

for directory in directories:
    print(directory)
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
    else:
        continue
    try:
        params['data'] = os.path.join(params['data'], 'test', 'test_data.npz')
        test_data =  np.load(params['data'], allow_pickle=True)
        test_masks = test_data['Y_test']

        pred_path = os.path.join(params['experiment'], 'test')
        fnames = os.listdir(pred_path)
        fnames = natural_sort(fnames)
        fnames = [os.path.join(pred_path, fname) for fname in fnames]
        #
        validation_metrics = pd.read_csv(os.path.join(params['experiment'], 'validation_metrics.csv'))

        best_seed_thresh = validation_metrics['seed_thresh'][np.argmax(validation_metrics['avAP19'])]
        best_fg_thresh = validation_metrics['fg_thresh'][np.argmax(validation_metrics['avAP19'])]
        #
        met = copy.deepcopy(met_template)
        for fname, gt_inst in tqdm(zip(fnames, test_masks)):
            pred = tif.imread(fname)
            pred = np.transpose(pred, [2,0,1])
            labeled = label(pred, best_fg_thresh, best_seed_thresh)
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
        met_aggregated['experiment'].append(params['experiment'])
        met_aggregated['data'].append(params['data'])
        for key in met.keys():
            met[key] = np.mean(met[key])
            met_aggregated[key].append(met[key])
        met_aggregated['fg_thresh'].append(best_fg_thresh)
        met_aggregated['seed_thresh'].append(best_seed_thresh)
    except:
       print('Failed in ' + directory)

pd.DataFrame(met_aggregated).to_csv('test_metrics_aggregated_2.csv')