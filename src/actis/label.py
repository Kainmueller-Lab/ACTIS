import mahotas
import numpy as np
from scipy import ndimage

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
    markers, cnt = ndimage.label(seeds)
    # compute watershed
    labelling = watershed(ws_surface, markers, fg)
    return labelling
