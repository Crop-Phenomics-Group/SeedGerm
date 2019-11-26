from itertools import chain
from operator import itemgetter

import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import dilation, erosion, remove_small_objects
from skimage.morphology import disk


def extend_green(img):
    img_g = img[:, :, 1]
    thresh = threshold_otsu(img_g)
    img_g = img_g > thresh
    # extend centre green ruler at top
    x_axis = np.nonzero(img_g[:200, :].sum(axis=0)[900:1700] > 150.)[0] + 900
    img[:50, x_axis, 1] = 1.
    # extend centre green ruler at bottom
    x_axis = np.nonzero(img_g[-200:, :].sum(axis=0)[900:1700] > 150.)[0] + 900
    img[-50:, x_axis, 1] = 1.
    return img

def fill_border(img, N, fillval=1.):
    img_copy = img.copy()
    img_copy[:N, :] = fillval
    img_copy[-N:, :] = fillval
    img_copy[:, :N] = fillval
    img_copy[:, -N:] = fillval
    return img_copy

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def segment_panel(img):
    img_g = img[:, :, 1]
    thresh = threshold_otsu(img_g)
    img_g = img_g > thresh
    
    #img_g = dilation(img_g, selem=disk(15))
    #img_g = 1. - binary_fill_holes(1. - img_g)
    #img_g = binary_closing(img_g, selem=disk(11))
    #img_g = fill_border(img_g, 10)
    
    img_g = erosion(img_g, disk(3))
    img_g = remove_small_objects(img_g)
    img_g = dilation(img_g, disk(15))
    img_g = fill_border(img_g, 10, fillval=True)
    img_g = binary_fill_holes(np.logical_not(img_g))
    
    l, n = label(img_g)

    # get regions that are panels based on size of area
    panels = []
    for rp in regionprops(l):  # , coordinates="xy"):
        if rp.area > 250000:
            panels.append((rp, rp.centroid[0], rp.centroid[1]))

    #sort panels based on y first, then x
    panels = sorted(panels, key=itemgetter(1))
    panels = chunks(panels, 2)
    panels = [sorted(p, key=itemgetter(2))   for p in panels]    
    panels = list(chain(*panels))

    # set mask, where 1 is top left, 2 is top right, 3 is middle left, etc...
    new_mask = np.zeros(img_g.shape)
    regions = []
    for idx in range(len(panels)):
        rp, _, _ = panels[idx]
        new_mask[l == rp.label] = idx + 1
        rp.label = idx + 1
        regions.append(rp)
    
    return new_mask.astype(np.int8), regions
