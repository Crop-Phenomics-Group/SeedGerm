# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:10:05 2015

@author: dty09rcu
"""

import datetime
import os
import re
import time

import math
import numpy as np
from matplotlib.path import Path
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from skimage.feature import peak_local_max
from skimage.measure import regionprops
# Vision/imaging imports.
from skimage.morphology import watershed

pj = os.path.join


def check_files_have_date(f_name):
    print(f_name)

    #JIC filename check
    re_result = re.findall("Date-([\d\-\_]+).", f_name)
    if re_result:
        print("Found JIC file name")
        try:
            time.strptime(re_result[0], "%d-%m-%Y_%H-%M")
            return True
        except:
            print("Issue passing JIC date")

    # elif re.findall("_([\d\_]+)_[\d]+.", f_name):
    #     print "Found Syngenta file name"
    #     try:
    #         time.strptime(re_result[0], "%d-%m-%Y_%H-%M")
    #     except:

    print()
    return False


def get_images_from_dir(path):

    file_list = os.listdir(path)
    file_list = [el for el in file_list if not el.startswith(".")]

    jpg_files, png_files = [], []
    re_1, re_2 = False, False

    for _file in file_list:
        if _file.endswith(".jpg"):
            jpg_files.append(_file)
        elif _file.endswith(".png"):
            png_files.append(_file)
        else:
            pass

        if re.findall('ID-(\d+)', _file):
            re_1 = True
        elif re.findall('_(\d+)\.', _file):
            re_2 = True
        else:
            pass

    if jpg_files and png_files:
        print("Found both jpeg and png files in the directory....")
        return []

    if not jpg_files and not png_files:
        print("Could not find any images in directory....")
        return []

    file_list = jpg_files if jpg_files else png_files

    try:
        if re_1:
            file_list = sorted(file_list, key=lambda s: int(re.findall('ID-(\d+)', s)[0]))
        elif re_2:
            file_list = sorted(file_list, key=lambda s: int(re.findall('_(\d+)\.', s)[0]))
    except:
        print("Could not perform regexp on files.")
        return []

    return file_list

def find_min_idx(x):
    for i, j in enumerate(x):
        if j == 1:
            return i
            break

def get_xy_range(l):
    x_sum = np.sum(l>0, axis=0)
    y_sum = np.sum(l>0, axis=1)
    x = ((x_sum > np.max(x_sum)/3.))
    y = ((y_sum > np.max(y_sum)/3.))
    
    x_min = find_min_idx(x)
    x_max = len(x) - find_min_idx(x[::-1])
    y_min = find_min_idx(y)
    y_max = len(y) - find_min_idx(y[::-1])
    xy_range = [x_min, x_max, y_min, y_max]
    return xy_range

def find_pts_in_range(pts, xy_range):
    ymed, xmed = np.median(pts, axis=0)
    ystd, xstd = pts.std(axis=0)
    alpha = 0.25
    
    x_min, x_max, y_min, y_max = xy_range

    
    bbPath = Path(np.array([
                [x_min - (xstd*alpha), y_min - (ystd*alpha)],
                [x_min - (xstd*alpha), y_max + (ystd*alpha)],
                [x_max + (xstd*alpha), y_max + (ystd*alpha)],
                [x_max + (xstd*alpha), y_min - (ystd*alpha)],
                [x_min - (xstd*alpha), y_min - (ystd*alpha)]
            ]))
    
    in_mask = []
    for y, x in pts:
        in_mask = np.concatenate((in_mask,[bbPath.contains_point((x, y))]),axis=0)   
    
    print("Removed %d boundary seed(s) ..." % (sum(in_mask == False)))
    
    return in_mask


#sort by y asc. and x desc.
def calculate_barycenter(point_list):
    centre_x, centre_y = 0,0
    for x,y in point_list:
        centre_x += x
        centre_y += y

    return (centre_x/len(point_list), centre_y/len(point_list))


def order_pts_lr_tb(pts, desired_pts, xy_range, cols, rows):
    dim_y = 0
    dim_x = 1
    y_values = pts[:, dim_y]
    hist_y, edges = np.histogram(y_values, bins=rows)

    y_rows = []
    #start from 1, so we dont use the left hand edge.
    #construct the bins of y points.
    for val1, val2 in zip(edges[:-1], edges[1:]):
        row = y_values[y_values >= val1]
        row = row[row <= val2]
        y_rows.append(row)

    #for each row of points, find the corresponding

    pts_rows = []
    new_order = np.zeros((1, 1))
    for row in y_rows:
        row_data = []
        for val in row:
            row_data.append(pts[np.where(pts[:, dim_y] == val)])
        if len(row_data) == 0:
            break
        row_data = np.concatenate(np.array(row_data), axis=0)
        pts_rows.append(row_data[np.argsort(row_data[:, dim_x])])  # sort on opposite dimension to the rows.

    # IF NEED TO FIX ORDER, LOOK HERE
    # distances = np.zeros((y_values.shape[0], y_values.shape[0]))
    # nearest = np.zeros((distances.shape[0], 1))
    # for i in range(y_values.shape[0]):
    #     for j in range(y_values.shape[0]):
    #         if i != j:
    #             distances[i, j] = y_values[i] - y_values[j]
    # for k in range(distances.shape[0]):
    #     nearest[k] = np.argmin(np.abs(distances[k, np.nonzero(distances[k, :])]))
    #     # nearest[k] = distances.argsort()[:3]
    #
    # y_values1 = np.sort(y_values)
    # differences = np.zeros(y_values1.shape[0]-1)
    # for k in range(y_values1.shape[0]-1):
    #     differences[k] = np.abs(y_values1[k+1] - y_values1[k])

    if len(pts_rows) > 0:
        new_order = np.concatenate(pts_rows, axis=0)

    #inefficient but dont care
    output = []
    if new_order.size == 1:
        None
    else:
        for new in new_order:
            output.append(np.argwhere(pts[:, np.newaxis] == new[:, np.newaxis])[0][0])

    return output

    ymed, xmed = np.median(pts, axis=0)
    ystd, xstd = pts.std(axis=0)
    alpha = 1.25

    x_min, x_max, y_min, y_max = xy_range

    #find the pts on the left most column.
    #what happens if one point in this row is missing?
    ppts = pts[pts[:, 0] < y_min + (ystd*0.3), :]

    #find the point that is closest to the top left hand corner.
    origin_pt = np.sqrt(np.power(ppts[:, 0], 2) + np.power(ppts[:, 1], 2)).argmin()

    seen = [origin_pt]
    
    _iter = -1
    seen = [origin_pt]
    ordered_pts = [pts[origin_pt]]
    row_start_pt = [y_min, x_min]
    while len(ordered_pts) < len(pts):
        
        _iter += 1
        if _iter > (desired_pts + 10):
            print("Not converged ordering seeds, breaking....")
            break
        
        curr_pt = ordered_pts[-1]
        
        bbPath = Path(np.array([
                [curr_pt[1], curr_pt[0]],
                [curr_pt[1] + (xstd*alpha), curr_pt[0] + (ystd*alpha)],
                [curr_pt[1] + (xstd*alpha), curr_pt[0] - (ystd*alpha)],
                [curr_pt[1], curr_pt[0]],
            ]))  
        
        in_mask = []
        for idx, p in enumerate(pts):
            if idx in seen:
                continue
            if bbPath.contains_point((p[1], p[0])):
                in_mask.append(idx)
        
        if not len(in_mask):
            # Reached end of line....
            bbPath = Path(np.array([
                [row_start_pt[1], row_start_pt[0]],
                [row_start_pt[1] - (xstd*alpha), row_start_pt[0] + (ystd*alpha)],
                [row_start_pt[1] + (xstd*alpha), row_start_pt[0] + (ystd*alpha)],
                [row_start_pt[1], row_start_pt[0]],
            ]))  
            
            in_mask = []
            for idx, p in enumerate(pts):
                if idx in seen:
                    continue
                if bbPath.contains_point((p[1], p[0])):
                    in_mask.append(idx)
            
            if not len(in_mask):
                # Reached final row...
                break
            
            closest_pt = np.sqrt(np.power(pts[in_mask, 0] - row_start_pt[0], 2) + np.power(pts[in_mask, 1] - row_start_pt[1], 2)).argsort()[0]
            new_pt = in_mask[closest_pt]
            row_start_pt = pts[new_pt]
            seen.append(new_pt)
            ordered_pts.append(pts[new_pt])
            continue
        
        closest_pt = np.sqrt(np.power(pts[in_mask, 0] - curr_pt[0], 2) + np.power(pts[in_mask, 1] - curr_pt[1], 2)).argsort()[0]
        new_pt = in_mask[closest_pt]
        seen.append(new_pt)
        ordered_pts.append(pts[new_pt])

    print(len(seen))

    return seen
    

def find_closest_n_points(pts, desired_pts):

    print(pts.shape, desired_pts)

    ymed, xmed = np.median(pts, axis=0)
    ystd, xstd = pts.std(axis=0)
    alpha = 0.25

    curr_x_low, curr_x_high = xmed - (alpha * xstd), xmed + (alpha * xstd)
    curr_y_low, curr_y_high = ymed - (alpha * ystd), ymed + (alpha * ystd)

    _iter = 0
    while True:

        bbPath = Path(np.array([
                    [curr_x_low, curr_y_low],
                    [curr_x_low, curr_y_high],
                    [curr_x_high, curr_y_high],
                    [curr_x_high, curr_y_low],
                    [curr_x_low, curr_y_low]
                ]))

        in_mask = []
        for y, x in pts:
            in_mask.append(bbPath.contains_point((x, y)))
        in_mask = np.array(in_mask)
        
        if in_mask.sum() == desired_pts:
            break
        
        curr_x_low, curr_x_high = curr_x_low - (alpha * xstd), curr_x_high + (alpha * xstd)
        curr_y_low, curr_y_high = curr_y_low - (alpha * ystd), curr_y_high + (alpha * ystd)
        
        _iter += 1
        
        if _iter >= 20:
            print("Converged without finding exact number of seeds...")
            break
            
    return in_mask

def delta(arr, wn=3):
    wn_2 = int(math.floor(wn / 2))
    arr_pad = np.pad(arr, [(wn_2, wn_2), (0, 0)], 'edge')
    arr_delta = arr_pad[(wn - 1):, :] - arr_pad[:-(wn - 1), :]
    return (arr_delta / float(2 * wn_2))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def cummean(arr,axis=0):
    return np.true_divide(arr.cumsum(axis), np.arange(1,arr.shape[axis]+1).reshape((arr.shape[axis],1)))
    
def s_to_datetime(s):
    ds = re.findall("Date-([\d\-\_]+).", s)[0]
    td = time.strptime(ds, "%d-%m-%Y_%H-%M")
    td = datetime.datetime(*td[:6])
    return td

def hours_between(start, end, round_minutes=False):
    delta_dt = end - start
    total_mins = divmod(delta_dt.total_seconds(), 60)[0]
    hr, mins = divmod(total_mins, 60)
    if round_minutes:
        min_extra = 1 if mins >= 30 else 0
        return "%d" % (hr + min_extra)
    else:
        return "%d:%02d" % (hr, mins)

def in_range(img, low, high):
    img = img.astype('f')
    
    y = img[:, :, 0]
    u = img[:, :, 1]
    v = img[:, :, 2]
    
    mask = np.logical_and(low[0] <= y, y <= high[0])
    mask = np.logical_and(mask, low[1] <= u)
    mask = np.logical_and(mask, u <= high[1])
    mask = np.logical_and(mask, low[2] <= v)
    mask = np.logical_and(mask, v <= high[2])
    return mask.astype(np.uint8) * 255

def rgb2ycrcb(img):
    img = img.astype(np.float)
    
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    
    Y = (r * 0.299) + (g * 0.587) + (b * 0.114)
    
    cr = ((r - Y) * 0.713) + 128.
    cb = ((b - Y) * 0.564) + 128.
    
    return np.round(np.dstack([Y, cr, cb])).astype(np.uint8)

def slugify(value):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '-', value)

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def flatten_img(img):
    """Convert an image with size (M, N, 3) to (M * N, 3). 
    Flatten the 2D pixels into a 1D where each row is a pixel and the columns are RGB values.
    """
    # Ought to check that the image contains 3 channels...
    return img.reshape((np.multiply(*img.shape[:2]), 3))

def label_next_frame(prev_l, prev_rprops, panel):
    curr_l, n = measurements.label(panel)

    curr_rprops = regionprops(curr_l)  # , coordinates="xy")
    
    new_curr_l = np.zeros(curr_l.shape)
    
    assigned = [[]] * (len(curr_rprops) + 1)

    for rp1 in prev_rprops:
        
        bins = np.bincount(curr_l[prev_l == rp1.label])
        
        if(len(bins) > 1):
            idx = np.argmax(bins[1:])
        else:
            yx1 = rp1.centroid
            yx2 = np.vstack([rp2.centroid    for rp2 in curr_rprops])

            dist = np.sqrt(np.power(yx2 - yx1, 2).sum(axis=1))
            idx = np.argmin(dist)
        
        #print idx, idx2
        
        new_curr_l[curr_l == curr_rprops[idx].label] = rp1.label
        new_curr_l = new_curr_l.astype(np.int)
        
        assigned[curr_rprops[idx].label] = assigned[curr_rprops[idx].label] + [rp1.label]
     
    for i in range(1,len(assigned)):
        if len(assigned[i]) > 1:
            #print assigned[i], i
            new_curr_l = separate_seeds(new_curr_l, prev_l, assigned[i], (curr_l == i))

    new_rprops = regionprops(new_curr_l)  #, coordinates="xy")
        
    return new_curr_l, new_rprops


#uses local_maxi from previous separated seed mask
#use this one
def separate_seeds(curr_l, prev_l, indexes, curr_mask):
    #curr_mask = curr_l == max(indexes)
    
    #foot = np.array([[1.,0.,1.], [0.,1.,0.], [1.,0.,1.]])
    foot = np.ones((3, 3))
    
    markers = np.zeros(curr_l.shape)
    s_lm = []
    count = 0
    
    for i in range(len(indexes)):
        seed_mask = prev_l == indexes[i]
        distance = ndi.distance_transform_edt(seed_mask)
        local_maxi = peak_local_max(distance, indices=False, footprint=foot, labels=seed_mask)
        sm_m = ndi.label(local_maxi)[0]
    
        markers[sm_m > 0] = sm_m[sm_m > 0] + count #+ 1
        count = count + sm_m.ravel().max()
        s_lm = s_lm + [sm_m.ravel().max()]
          
    #markers[1,1] = 1 
    
    distance = ndi.distance_transform_edt(curr_mask)
    labels = watershed(-distance, markers, mask=curr_mask)
    #labels = random_walker(curr_mask, markers)
    #labels = labels - 1
    
    count = 1
    
    #print'\t', s_lm, '\t', markers.ravel().max(), '\t', labels.ravel().max()
    
    for i in range(len(indexes)):
        #print 'i = ', i
        for j in range(count, s_lm[i]+count):
            #print 'j = ' , j
            curr_l[labels == j] = indexes[i]
            
        count = count + s_lm[i]
    
    return curr_l
    
    
def simple_label_next_frame(prev_l, prev_rprops, panel):
    """ Given current frame labels and region properties, label the next frame so that the label
    values are consistent across the frames.
    """
    curr_l, n = measurements.label(panel)
    curr_rprops = regionprops(curr_l)  #, coordinates="xy")

    new_curr_l = np.zeros(curr_l.shape)

    for rp1 in prev_rprops:
        yx1 = rp1.centroid
        yx2 = np.vstack([rp2.centroid for rp2 in curr_rprops])

        dist = np.sqrt(np.power(yx2 - yx1, 2).sum(axis=1))
        idx = np.argmin(dist)

        new_curr_l[curr_l == curr_rprops[idx].label] = rp1.label

    new_curr_l = new_curr_l.astype(np.int)
    new_rprops = regionprops(new_curr_l)  #, coordinates="xy")

    return new_curr_l, new_rprops
