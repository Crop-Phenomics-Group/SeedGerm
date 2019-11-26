# -*- coding: utf-8 -*-

""" processor.py - Handles the main processing of the germination experiments.

Sets up experiment directory variables for reading and writing data. Includes
various functions for processing the data for each of the main parts of the
processing. Data is produced at all stages for checking that processes have
functioned correctly.

Processing of an experiment runs as follows:
a.  Requires user to have set the YUV ranges.
1.  Save initial image.
2.  Extract panels and save data.
3.  Save contour image showing extracted panel boundaries.
4.  Extract bg/fg pixels and train ensemble classifiers.
5.  Remove backgrounds producing fg masks for each image.
6.  Label the seed positions using the first few frames.
7.  Perform the germination classification.
8.  Analyse the results producing the required output data.
9.  Perform quantification of the seed morphological and colour data.
"""

import copy
import glob
import json
# General python imports.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import threading
import time
import traceback
from helper.experiment import Experiment
from helper.functions import *
# Germapp imports.
from helper.horprasert import *
from helper.panel_segmenter import fill_border
from helper.panelclass import Panel
from helper.seedpanelclass import SeedPanel
from itertools import chain
from matplotlib import pyplot as plt
from numpy import random
from operator import itemgetter
# Imaging/vision imports.
# import cv2
import scipy
import scipy.ndimage
from scipy.ndimage.morphology import binary_fill_holes
# Machine learning and statistics imports.
from scipy.stats import mode
from skimage.morphology import *
from skimage.segmentation import clear_border
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import imageio
from sklearn import *
import seaborn as sn
from skimage.morphology import convex_hull_object
import numpy as np

import matplotlib
matplotlib.use('Agg')
import sys
import time
import os
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import stats

if not os.path.exists("./data/"):
    os.makedirs("./data/")

from gui.application import Application
from brain.core import Core
from brain.speciesclassifier import SpeciesClassifier as spp

exp_images_dir = "Corn"
yuv_range = [[0, 0, 153], [255, 255, 255]]
n_rows = 1
n_cols = 1
n_panels = n_rows*n_cols
n_seeds = 35
LGBM = LGBMClassifier(boosting_type='gbdt', n_estimators=100, n_jobs=-1, random_state=0)
SGD = SGDClassifier(max_iter=100, tol=1e-5, n_jobs=-1, random_state=0)
LR = LogisticRegression(max_iter=100, tol=1e-5, solver='lbfgs', n_jobs=-1, random_state=0)
core = Core
app = Application
#
#
#
#
# def compare_pts(pt1, pt2):
#     return pt1 > pt2
#
#
# def _save_init_image(exp_images_dir):
#     out_f = pj(exp_images_dir, "Images/init_img.jpg")
#     all_img = []
#     for image_path in glob.glob(exp_images_dir + "\\*.jpg"):
#         im = imageio.imread(image_path)
#         all_img.append(im)
#     for image_path in glob.glob(exp_images_dir + "\\*.png"):
#         im = imageio.imread(image_path)
#         all_img.append(im)
#     img_01 = all_img[0]
#     fig = plt.figure()
#     plt.imshow(img_01)
#     fig.savefig(out_f)
#     plt.close(fig)
#     return all_img
#
#
#
# def _yuv_clip_image(img, yuv_range):
#     yuv_low = yuv_range[0]
#     yuv_high = yuv_range[1]
#     img_yuv = rgb2ycrcb(img)
#     mask_img = in_range(img_yuv, yuv_low, yuv_high)
#     return mask_img.astype(np.bool)
#
#
#
#
# def _extract_panels(img, chunk_no, chunk_reverse, yuv_range, n_panels):
#     def get_mask_objects(idx, rp):
#         tmp_mask = np.zeros(mask_img.shape)
#         tmp_mask[l == rp.label] = 1
#
#         both_mask = np.logical_and(obj_only_mask, tmp_mask)
#         both_mask = remove_small_objects(both_mask)
#         _, panel_object_count = measurements.label(both_mask)  # , return_num=True)
#         return panel_object_count
#
#
#     mask_img = _yuv_clip_image(img, yuv_range)
#     mask_img = remove_small_objects(fill_border(mask_img, 10, fillval=False), min_size=1024)
#     mask_img_cleaned_copy = mask_img.copy()
#     mask_img = erosion(binary_fill_holes(mask_img), disk(7))
#
#     obj_only_mask = np.logical_and(mask_img, np.logical_not(mask_img_cleaned_copy))
#
#     # get labels
#     l, n = measurements.label(mask_img)
#
#     rprops = regionprops(l, coordinates='xy')
#     rprops = [(rp, get_mask_objects(idx, rp)) for idx, rp in enumerate(rprops)]
#     rprops = sorted(rprops, key=itemgetter(1), reverse=True)
#     panels = [(rp, rp.centroid[0], rp.centroid[1]) for rp, _ in rprops[:n_panels]]
#
#     panels = sorted(panels, key=itemgetter(1))
#     panels = chunks(panels, chunk_no)
#     panels = [sorted(p, key=itemgetter(2), reverse=chunk_reverse) for p in panels]
#     print(panels)
#     panels = list(chain(*panels))
#
#     panel_list = []  # List[Panel]
#     for idx in range(len(panels)):
#         rp, _, _ = panels[idx]
#         new_mask = np.zeros(mask_img.shape)
#         new_mask[l == rp.label] = 1
#         panel_list.append(Panel(idx + 1, new_mask.astype(np.bool), rp.centroid, rp.bbox))
#     panel_list = panel_list  # type: List[Panel]
#     return panel_list, rprops
#
#
#
#
#
# def _save_contour_image(img, exp_images_dir):
#     fig = plt.figure()
#
#     img_full_mask = np.zeros(img.shape[:-1])
#     for p in panel_list:
#         plt.annotate(str(p.label), xy=p.centroid[::-1], color='r', fontsize=20)
#         min_row, min_col, max_row, max_col = p.bbox
#         img_full_mask[min_row:max_row, min_col:max_col] += p.mask_crop
#
#     panels_mask = img_full_mask.astype(np.bool)
#
#     out_f = pj(exp_images_dir, "Images/img_panels.jpg")
#
#     plt.contour(np.flip(img_full_mask, axis=0), [0.5], colors='r')
#     fig.savefig(out_f)
#     plt.close(fig)
#
#     fig = plt.figure()
#     plt.imshow(panels_mask)
#     fig.savefig(pj(exp_images_dir, "Images/panels_mask.jpg"))
#     plt.close(fig)
#     return panels_mask
#
#
# def _ensemble_predict(clfs, X):
#     preds = []
#     for clf in clfs:
#         preds.append(clf.predict(X))
#     y_pred = mode(np.vstack(preds), axis=0)[0][0]
#     return y_pred
#
# def _train_clfs(clf_in, all_im, exp_images_dir, panels_mask, yuv_range):
#     fig, axarr = plt.subplots(2, 3)
#     axarr = list(chain(*axarr))
#
#     train_masks = []
#     train_images = []
#
#     train_img_ids = all_im[0:3]
#     train_img_ids += [all_im[int(len(all_im) / 2) - 1], all_im[len(all_im) - 2]]
#
#     for idx, img_i in enumerate(train_img_ids):
#         # curr_img = cv2.imread(os.path.join(self.exp.img_path,
#         # self.imgs[img_i]))
#         train_images.append(img_i)
#
#         curr_mask = _yuv_clip_image(np.array(img_i).astype('uint16'), yuv_range)
#         curr_mask = dilation(curr_mask, disk(3))
#         train_masks.append(curr_mask.astype(np.bool))
#
#         axarr[idx].imshow(curr_mask)
#         axarr[idx].axis('off')
#
#     fig.savefig(pj(exp_images_dir, "Images/train_imgs.jpg"))
#     plt.close(fig)
#
#     all_bg_pixels = []
#     all_fg_pixels = []
#
#     for idx, (mask, curr_img) in enumerate(zip(train_masks, train_images)):
#         # produce the background and foreground masks for this mask
#         bg_mask3 = np.dstack([np.logical_and(mask, panels_mask)] * 3)
#         fg_mask3 = np.dstack([np.logical_and(np.logical_not(mask), panels_mask)] * 3)
#
#         bg_rgb_pixels = _create_transformed_data(curr_img * bg_mask3)
#         fg_rgb_pixels = _create_transformed_data(curr_img * fg_mask3)
#
#         print(bg_rgb_pixels.shape)
#         all_bg_pixels.append(bg_rgb_pixels)
#         all_fg_pixels.append(fg_rgb_pixels)
# #
# #     bg_rgb_pixels = np.vstack(all_bg_pixels)
# #     fg_rgb_pixels = np.vstack(all_fg_pixels)
# #     print(bg_rgb_pixels.shape, fg_rgb_pixels.shape)
#         bg_rgb_pixels = np.vstack(all_bg_pixels)
#         blue_E, blue_s = bg_rgb_pixels.mean(axis=0), bg_rgb_pixels.std(axis=0)
#         alpha = flatBD(bg_rgb_pixels, blue_E, blue_s)
#         a = np.sqrt(np.power(alpha - 1, 2) / bg_rgb_pixels.shape[0])
#         b = np.sqrt(np.power(flatCD(bg_rgb_pixels, blue_E, blue_s), 2) / bg_rgb_pixels.shape[0])
#         TCD = self._gmm_get_TCD(bg_rgb_pixels, blue_E, blue_s, a, b)
#         fg_rgb_pixels = np.vstack(all_fg_pixels)
#         sgd = SGDClassifier()
# #
# #     # bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
# #     # bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain)
# #
# #     # fg_retain = int(fg_rgb_pixels.shape[0] * 0.1)
# #     # fg_retain = random.choice(fg_rgb_pixels.shape[0], fg_retain)
# #
# #     # bg_rgb_pixels = bg_rgb_pixels[bg_retain, :]
# #     # fg_rgb_pixels = fg_rgb_pixels[fg_retain, :]
# #
# #     # print bg_rgb_pixels.shape, fg_rgb_pixels.shape
# #
# #     # make training data
# #     X = np.vstack([bg_rgb_pixels, fg_rgb_pixels])
# #     y = np.concatenate([
# #         np.zeros(bg_rgb_pixels.shape[0]),
# #         np.ones(fg_rgb_pixels.shape[0])
# #     ])
# #
# #     classifiers = clf_in
# #     classifiers = _train_clf(classifiers, ensemble_clf_f=None, X=X, y=y)
# #     return classifiers
# #
# # def _train_clf(clf_in, ensemble_clf_f, X, y):
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# #     print(X.shape, y.shape)
# #     print(X_train.shape, y_train.shape)
# #     print(X_test.shape, y_test.shape)
# #
# #     for clf in clf_in:
# #         clf.fit(X_train, y_train)
# #         print(clf.score(X_test, y_test))
# #
# #     y_pred = _ensemble_predict(clf_in, X_test)
# #     print(y_pred.shape)
# #     print(accuracy_score(y_test, y_pred))
# #
# #     return clf_in
# #
# #
# # def _create_transformed_data(rgb_pixels):
# #     # removed the random selection. Use as much data as possible.
# #     # get all of the bg pixels
# #     rgb_pixels = flatten_img(rgb_pixels)
# #     rgb_pixels = rgb_pixels[rgb_pixels.all(axis=1), :]
# #     # bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
# #     # bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain, replace=False)
# #     # bg_rgb_pixels = bg_rgb_pixels[bg_retain, :]
# #
# #     return rgb_pixels
# #
# # def _remove_background(all_im, panel_list, classifiers):
# #     all_masks = []
# #     print("Number of images:", len(all_im))
# #     for idx in range(0, len(all_im)):
# #         print(idx,)
# #
# #         img_masks = []
# #
# #         img = all_im[idx]
# #
# #         X = []
# #         y = []
# #
# #         for p in panel_list:
# #             panel_img = p.get_cropped_image(img)
# #             pp_predicted = _ensemble_predict(classifiers, flatten_img(panel_img))
# #             pp_predicted.shape = p.mask_crop.shape
# #             pp_predicted = pp_predicted.astype(np.bool)
# #
# #             img_masks.append(pp_predicted)
# #
# #         all_masks.append(img_masks)
# #         print("Current number of masks:", len(all_masks))
# #     print("Number of masks:", len(all_masks))
# #     return all_masks
# #
# # def _label_seeds(n_panels, all_im, all_masks, panel_list, n_seeds, n_rows, n_cols):
# #     fig, axarr = plt.subplots(n_panels, 1, figsize=(16, 16 * n_panels))
# #     try:
# #         axarr.shape
# #     except:
# #         axarr = [axarr]
# #
# #     panel_l_rprops = []
# #
# #     # retain = int(len(all_im) / 10.)
# #
# #     init_masks = all_masks
# #     # for i in range(0, retain):
# #     #     data = all_masks[i]
# #     #     init_masks.append(data)
# #     all_rprops = []
# #     print(len(all_masks))
# #     for idx in range(0, len(all_masks)):
# #         panel_l_rprops = []
# #         fig, axarr = plt.subplots(n_panels, 1, figsize=(16, 16 * n_panels))
# #         for ipx, panel in enumerate(panel_list):
# #             # :10 for tomato, :20 for corn/brassica
# #             print('IDX = ', idx, 'Panel = ', panel)
# #             mask_med = init_masks[idx][ipx]
# #             mask_med = clear_border(mask_med).astype(np.bool)
# #             mask_med = remove_small_objects(mask_med)
# #
# #             # label features in an array using the default structuring element which is a cross.
# #             labelled_array, num_features = measurements.label(mask_med)
# #             print('Number of features: ', num_features)
# #             rprops = regionprops(labelled_array,  coordinates='xy')
# #             print('Length of rprops: ', len(rprops))
# #
# #             all_seed_rprops = []  # type: List[SeedPanel]
# #             for rp in rprops:
# #                 all_seed_rprops.append(
# #                     SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
# #                               rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent,
# #                               rp.equivalent_diameter, rp.convex_area))
# #
# #             # Get maximum number of seeds
# #             pts = np.vstack([el.centroid for el in all_seed_rprops])
# #             in_mask = find_closest_n_points(pts, n_seeds)
# #
# #             # if we've got more seeds than we should do, should we throw them away?
# #             if len(in_mask) > n_seeds:
# #                 all_seed_rprops_new = []
# #                 for rp, im in zip(all_seed_rprops, in_mask):
# #                     if im:
# #                         all_seed_rprops_new.append(rp)
# #                     else:
# #                         # Remove false seed rprops from mask
# #                         labelled_array[labelled_array == rp.label] = 0
# #                 all_seed_rprops = all_seed_rprops_new
# #             # end if-----------------------------------#
# #
# #             # Remove extra 'seeds' (QR labels) from boundary
# #             pts = np.vstack([el.centroid for el in all_seed_rprops])
# #             xy_range = get_xy_range(labelled_array)
# #             in_mask = find_pts_in_range(pts, xy_range)
# #
# #             # if we've got less seeds than we should do, should we throw them away?
# #             if len(in_mask) > n_seeds:
# #                 all_seed_rprops_new = []
# #                 for rp, im in zip(all_seed_rprops, in_mask):
# #                     if im:
# #                         all_seed_rprops_new.append(rp)
# #                     else:
# #                         # Remove false seed rprops from mask
# #                         labelled_array[labelled_array == rp.label] = 0
# #                 all_seed_rprops = all_seed_rprops_new
# #             # end if-----------------------------------#
# #
# #             # need to update pts if we have pruned.
# #             pts = np.vstack([el.centroid for el in all_seed_rprops])
# #
# #             pts_order = order_pts_lr_tb(pts, n_seeds, xy_range, n_cols, n_rows)
# #
# #             new_order = []
# #             new_mask = np.zeros(labelled_array.shape)
# #             for s_idx, s in enumerate(pts_order):
# #                 sr = all_seed_rprops[s]
# #                 # reorder mask
# #                 new_mask[labelled_array == sr.label] = s_idx + 1
# #                 sr.label = s_idx + 1
# #                 new_order.append(sr)
# #
# #             all_seed_rprops = new_order
# #             labelled_array = new_mask
# #
# #             # we add an array of labels and the region proprties for each panel.
# #             print("all seed rprops length: " + str(len(all_seed_rprops)))
# #             panel_l_rprops.append((labelled_array, all_seed_rprops))
# #
# #             # axarr[ipx].imshow(mask_med)
# #             # axarr[ipx].annotate(str(panel.label), xy=(10, 10), color='r', fontsize=20)
# #
# #             # for rp in all_seed_rprops:
# #             #     axarr[ipx].annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r', fontsize=16)
# #
# #             # axarr[ipx].axis('off')
# #         c = str(idx)
# #         filename = "Images/seeds_labelled" + c + ".png"
# #         print(c, filename)
# #         fig.savefig(pj(exp_images_dir, filename))
# #         plt.close(fig)
# #         all_rprops.append(panel_l_rprops)
# #     return all_rprops
# #
# # def _build_classifiers(panel_masks, use_colour, all_imgs, panel_labels, panel_regionprops, panel):
# #     to_analyse = int(len(panel_masks) * 0.1)
# #     hu_feas = []
# #     areas = []
# #     lengths = []
# #
# #     if use_colour:
# #         colors_r = []
# #         colors_g = []
# #         colors_b = []
# #
# #     # loop through the first 10% of the image/mask set.
# #     for index, (mask, img_f) in enumerate(list(zip(panel_masks, all_imgs))[:to_analyse]):
# #         img = all_imgs[index]
# #         img = panel.get_bbox_image(img)
# #         c_label, c_rprops = simple_label_next_frame(panel_labels, panel_regionprops, mask)
# #
# #         # loop through each seed found.
# #         for idx, rp in enumerate(c_rprops):
# #             if use_colour:
# #                 r, g, b = generate_color_histogram(img, rp)
# #                 colors_r.append(r)
# #                 colors_g.append(g)
# #                 colors_b.append(b)
# #             hu_feas.append(rp.moments_hu)
# #             areas.append(rp.area)
# #             lengths.append([rp.minor_axis_length, rp.major_axis_length, float(rp.minor_axis_length+1.0)/float(rp.major_axis_length+1.0)])
# #
# #
# #     areas = np.vstack(areas)
# #     hu_feas = np.vstack(hu_feas)
# #     hu_feas = np.hstack([hu_feas, delta(hu_feas), areas, delta(areas), lengths, delta(lengths)]) #added in area and delta area.
# #     if use_colour:
# #         color_feas = np.hstack([np.vstack(colors_r), np.vstack(colors_g), np.vstack(colors_b)])
# #
# #     # normalise the hu features and the delta mean.
# #     hu_feas_mu = hu_feas.mean(axis=0)
# #     hu_feas_stds = hu_feas.std(axis=0)
# #     hu_feas = (hu_feas - hu_feas_mu) / (hu_feas_stds+0.0000001)
# #     # train on hu_features
# #
# #     # self.clf_hu = GaussianMixture(n_components=3)
# #     clf_hu = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# #     clf_hu.fit(hu_feas)
# #     # hu_feas_score = self.clf_hu.score_samples(hu_feas) #GMM
# #
# #     # normalise the histograms.
# #     if use_colour:
# #         color_feas_mu = color_feas.mean(axis=0)
# #         color_feas_stds = color_feas.std(axis=0)
# #         color_feas = (color_feas - color_feas_mu) / color_feas_stds
# #         # train on color features
# #         clf_color = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# #         # self.clf_color = GaussianMixture(n_components=3) #GMM
# #         clf_color.fit(color_feas)
# #
# #
# #         # color_feas_score = self.clf_color.score_samples(color_feas) #GMM
# #         # average score to generate probability. from the two models.
# #         # self.combined_prob_low = (np.percentile(hu_feas_score, 5.) + np.percentile(color_feas_score, 5.)) / 2 #GMM
# #         # else:
# #         # or just use the regular model.
# #         # self.combined_prob_low = np.percentile(hu_feas_score, 5.) #GMM
# #
# # def generate_color_histogram(img, region_prop):
# #     (min_row, min_col, max_row, max_col) = region_prop.bbox
# #     pixels = np.array(img[min_row:max_row, min_col:max_col])
# #     r, g, b = np.dsplit(pixels, pixels.shape[-1])
# #     hist_r, _ = np.histogram(r)
# #     hist_g, _ = np.histogram(g)
# #     hist_b, _ = np.histogram(b)
# #     return hist_r, hist_g, hist_b
# #
# # def _get_seed_mask_set(seed_rp, panel_masks, panel_labels, spp_mask_dilate):
# #     min_row, min_col, max_row, max_col = seed_rp.bbox + np.array([-50, -50, -50, -50])
# #
# #     row_max, col_max = panel_masks[0].shape
# #
# #     if min_row < 0:
# #         min_row = 0
# #     if min_col < 0:
# #         min_col = 0
# #     if max_row > row_max:
# #         max_row = row_max
# #     if max_col > col_max:
# #         max_col = col_max
# #
# #     seed_masks = [el[min_row:max_row, min_col:max_col] for el in panel_masks]
# #
# #     init_mask = (panel_labels == seed_rp.label)[min_row:max_row, min_col:max_col]
# #     sm_extracted = [np.logical_and(seed_masks[0], init_mask)]
# #
# #     # for prev_mask, curr_mask in zip(sm_extracted[-1], seed_masks[1:])
# #     for i in range(1, len(seed_masks)):
# #         prev_mask = sm_extracted[i - 1]
# #         curr_mask = seed_masks[i]
# #
# #         prev_mask_dilated = dilation(prev_mask, disk(spp_mask_dilate))
# #         new_mask = np.logical_and(prev_mask_dilated, curr_mask)
# #         sm_extracted.append(new_mask)
# #
# #     return sm_extracted, (min_row, min_col, max_row, max_col)
# #
# #
# #
# #
# #
# # def _classify(p_idx, panel, all_imgs, p_labels, p_rprops, p_masks, all_masks):
# #     seed_classification = {}
# #     all_areas = []
# #     cols = []
# #     use_colour = False
# #     spp_mask_dilate = 3
# #     _build_classifiers(all_masks, use_colour, all_imgs, p_labels, p_rprops, panel)
# #
# #     print(len(p_rprops))
# #     # for all the found seeds of the input panel.
# #     for index, rp in enumerate(p_rprops):
# #         cols.append(rp.label)
# #         seed_masks, _ = _get_seed_mask_set(rp, p_masks, p_labels, spp_mask_dilate)
# #
# #         list_error = False
# #         areas = []
# #         hu_feas = []
# #         lengths = []
# #         if use_colour:
# #             colors_r = []
# #             colors_g = []
# #             colors_b = []
# #         # for each seed in all the images.
# #         for idx, m in enumerate(seed_masks):
# #             # extract image region.
# #             m = m.astype('i')
# #
# #             m_rp = regionprops(m, coordinates="xy")
# #             if not m_rp:
# #                 list_error = True
# #                 print("Empty regionprops list in classifier")
# #                 break
# #
# #             m_rp = m_rp[0]
# #
# #             if use_colour:
# #                 # generate the color histogram.
# #                 img = imageio.imread(all_imgs[idx])
# #                 img = panel.get_bbox_image(img)
# #                 r, g, b = generate_color_histogram(img, m_rp)
# #                 colors_r.append(r)
# #                 colors_g.append(g)
# #                 colors_b.append(b)
# #             hu_feas.append(m_rp.moments_hu)
# #             areas.append(m_rp.area)
# #             #do laplacian correction on the axis ratio to ensure its not zero.
# #             lengths.append([m_rp.minor_axis_length, m_rp.major_axis_length, float(m_rp.minor_axis_length+1.0)/float(m_rp.major_axis_length+1.0)])
# #
# #         if list_error:
# #             seed_classification[rp.label] = [0] * len(seed_masks)
# #             print("Error with current seed,", rp.label)
# #             continue
# #
# #         # trying to use the cummean of the areas. ie its rate of growth over time.
# #         hu_feas = np.vstack(hu_feas)
# #         hu_feas_mu = hu_feas.mean(axis=0)
# #         hu_feas_stds = hu_feas.std(axis=0)
# #         areas1 = np.vstack(areas)
# #         lengths = np.vstack(lengths)
# #
# #         hu_feas = np.hstack([hu_feas, delta(hu_feas), areas1, delta(areas1), lengths, delta(lengths)])
# #         hu_feas = (hu_feas - hu_feas_mu) / (hu_feas_stds+0.000001)
# #
# #         # hu_preds = self.clf_hu.score_samples(hu_feas) #GMM
# #         clf_hu = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# #         hu_preds = clf_hu.predict(hu_feas)
# #
# #         if use_colour:
# #             color_feas = np.hstack([np.vstack(colors_r), np.vstack(colors_g), np.vstack(colors_b)])
# #             color_feas_mu = color_feas.mean(axis=0)
# #             color_feas_stds = color_feas.std(axis=0)
# #             color_feas = (color_feas - color_feas_mu) / color_feas_stds
# #             # color_preds = self.clf_color.score_samples(color_feas) #GMM
# #             # average_pred = ((color_preds + hu_preds) / 2) < self.combined_prob_low #GMM
# #             clf_color = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# #             color_preds = clf_color.predict(color_feas)
# #             average_pred = ((color_preds + hu_preds) / 2) <= 0
# #
# #         else:
# #             # average_pred = hu_preds < self.combined_prob_low #GMM
# #             average_pred = hu_preds <= 0  # convert -1 or 1 to boolean
# #
# #         areas = np.array(areas)
# #         all_areas.append(areas)
# #
# #         to_analyse = int(len(p_masks) * 0.1)
# #         area_mu = areas[:to_analyse].mean()
# #         area_growth_min = 1.25
# #
# #         if not (areas > (area_mu * area_growth_min)).sum():
# #             seed_classification[rp.label] = [0] * len(seed_masks)
# #             continue
# #
# #         img_ten_percent = int(len(seed_masks) * 0.05)
# #         img_start = np.argmax(areas > (area_mu * area_growth_min)) - img_ten_percent
# #         if img_start < 0:
# #             img_start = 0
# #
# #         seed_area_mask = np.array([0] * len(seed_masks))
# #         seed_area_mask[img_start:] = 1
# #
# #         all_preds = np.logical_and(average_pred, seed_area_mask)
# #
# #         seed_classification[rp.label] = all_preds.astype('i').tolist()
# #     return seed_classification
# #
# #
# #
# #
# #
# # def _perform_classification(panel_list, rprops, all_imgs, all_masks):
# #     panel_germ = []
# #     for panel_idx, panel_object in enumerate(panel_list):
# #         try:
# #             print("panel %d" % panel_idx)
# #
# #             # this is the tuple of the labelled arrays genereated from mesurements.
# #             # label and the regionprops generated by the measurements.regionprops
# #             panel_labels, panel_regionprops = rprops[0][panel_idx]
# #
# #             # extract all the masks for the specific panel, for every image.
# #             p_masks = [el[panel_idx] for el in all_masks]
# #             # self.spp_processor.use_colour = self.exp.use_colour ######## RETURN TO THIS LINE AND FIX ########
# #
# #             panel_germ = _classify(p_idx=panel_idx,
# #                                        panel=panel_object,
# #                                        all_imgs=all_imgs[0:],
# #                                        p_labels=panel_labels,
# #                                        p_rprops=panel_regionprops,
# #                                        p_masks=p_masks,
# #                                    all_masks=all_masks
# #                                        )
# #
# #             print("panel germ length " + str(len(panel_germ)))
# #
# #             out_f = pj(
# #                 "Results/",
# #                 "germ_panel_%d.json" % (panel_idx)
# #             )
# #
# #             with open(out_f, "w") as fh:
# #                 json.dump(panel_germ, fh)
# #
# #
# #         except Exception as e:
# #             print("Could not run panel %d" % (panel_idx))
# #             print(e)
# #             traceback.print_exc()
# #     return panel_germ
# #
# # def _get_cumulative_germ(self, germ, win=5):
# #     for m in range(germ.shape[0]):
# #         curr_seed = germ[m, :]
# #         idx = 0
# #         while idx < (curr_seed.shape[0] - win):
# #             if curr_seed[idx:idx + win].all():
# #                 curr_seed[:idx - 1] = 0
# #                 curr_seed[idx - 1:] = 1
# #                 break
# #             idx += 1
# #         if idx >= (curr_seed.shape[0] - win):
# #             curr_seed[:] = 0
# #     return germ.sum(axis=0), germ
# #
# # def _analyse_results(self, proprtions):
# #     all_germ = []
# #     for i in range(self.exp.panel_n):
# #         # if the with fails to open, we should not perform operations on germ_d.
# #         with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (i))) as fh:
# #             germ_d = json.load(fh)
# #
# #             # ensure the germ_d isn't empty.
# #             if len(germ_d) == 0:
# #                 continue
# #
# #             germinated = []
# #             for j in range(1, len(germ_d.keys()) + 1):
# #                 germinated.append(germ_d[str(j)])
# #
# #             germinated = np.vstack(germinated)
# #             all_germ.append(germinated)
# #
# #     p_totals = []
# #     for i in range(self.exp.panel_n):
# #         l, rprop = self.panel_l_rprops[i]
# #         p_totals.append(len(rprop))
# #
# #     print(len(all_germ))
# #     if len(all_germ) == 0:
# #         raise Exception("Germinated seeds found is 0. Try changing YUV values.")
# #
# #     print(p_totals)
# #
# #     cum_germ_data = []
# #     for germ in all_germ:
# #         cum_germ_data.append(self._get_cumulative_germ(germ, win=7)[0])
# #
# #     initial_germ_time_data = []
# #     for germ in all_germ:
# #         cols, rows = germ.shape
# #         init_germ_time = []
# #         for m in range(cols):
# #             for n in range(rows):
# #                 if germ[m, n]:
# #                     # For adding offset
# #                     init_germ_time.append(n + self.exp.start_img)
# #                     break
# #         initial_germ_time_data.append(init_germ_time)
# #
# #     cum_germ_data = np.vstack(cum_germ_data).T.astype('f')
# #
# #     print(len(cum_germ_data))
# #
# #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12., 10.))
# #     fig.suptitle(self.exp.name)
# #
# #     p_t_text = ""
# #     for i in range(self.exp.panel_n):
# #         p_t_text += "panel %d: %d" % (i + 1, p_totals[i])
# #         if ((i + 1) % 2) == 0:
# #             p_t_text += "\n"
# #         else:
# #             p_t_text += "    "
# #
# #     plt.figtext(0.05, 0.93, p_t_text)
# #
# #     # Only use date information if it is contained in the filename
# #     has_date = check_files_have_date(self.imgs[0])
# #     start_dt = None
# #     if has_date:
# #         start_dt = s_to_datetime(self.imgs[0])
# #
# #     n_frames = cum_germ_data.shape[0]
# #
# #     for idx in range(cum_germ_data.shape[1]):
# #         ax1.plot(
# #             range(self.exp.start_img, self.exp.start_img + n_frames),
# #             cum_germ_data[:, idx] / float(p_totals[idx]), label=str(idx + 1)
# #         )
# #     ax1.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])
# #
# #     if has_date:
# #         # Sort out xtick labels in hours
# #         xtick_labels = []
# #         for val in ax1.get_xticks():
# #             if int(val) >= len(self.imgs):
# #                 break
# #             curr_dt = s_to_datetime(self.imgs[int(val)])
# #             xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))
# #
# #         ax1.set_xlabel("Time (hours)")
# #         ax1.set_xticklabels(xtick_labels, )
# #     else:
# #         ax1.set_xlabel("Image number")
# #
# #     ax1.legend(loc="upper left")
# #     ax1.set_ylabel("Cumulative germinated percent")
# #     ax1.set_title("Cumulative germination as percent")
# #     ax1.grid()
# #
# #     data = []
# #     for idx in range(cum_germ_data.shape[1]):
# #         cum_germ = cum_germ_data[:, idx].copy().ravel()
# #         cum_germ /= p_totals[idx]
# #
# #         germ_pro_total = cum_germ[-1]
# #         prop_idxs = []
# #         for pro in proprtions:
# #             if (cum_germ > pro).any():
# #                 pos_idx = np.argmax(cum_germ >= pro)
# #
# #                 if has_date:
# #                     curr_dt = s_to_datetime(
# #                         self.imgs[self.exp.start_img + pos_idx])
# #                     pos_idx = hours_between(start_dt, curr_dt)
# #             else:
# #                 pos_idx = 'n/a'
# #             prop_idxs.append(str(pos_idx))
# #         data.append(prop_idxs)
# #
# #     columns = tuple('%d%%' % (100 * prop) for prop in proprtions)
# #     rows = ['  %d  ' % (x) for x in range(1, self.exp.panel_n + 1)]
# #     the_table = ax2.table(cellText=data,
# #                           rowLabels=rows,
# #                           colLabels=columns,
# #                           loc='center')
# #
# #     tbl_props = the_table.properties()
# #     tbl_cells = tbl_props['child_artists']
# #     for cell in tbl_cells:
# #         cell.set_height(0.1)
# #     ax2.set_title("Percentage T values")
# #     ax2.axis('off')
# #
# #     ax3.boxplot(initial_germ_time_data, vert=False)
# #     ax3.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])
# #     ax3.set_ylabel("Panel number")
# #     ax3.set_title('Germination time box plot')
# #     ax3.grid()
# #
# #     if has_date:
# #         ax3.set_xlabel("Time (hours)")
# #         xtick_labels = []
# #         for val in ax3.get_xticks():
# #             if int(val) >= len(self.imgs):
# #                 break
# #             curr_dt = s_to_datetime(self.imgs[int(val)])
# #             xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))
# #         ax3.set_xticklabels(xtick_labels)
# #     else:
# #         ax3.set_xlabel("Image number")
# #
# #     print(cum_germ_data[-1, :] / np.array(p_totals))
# #
# #     ax4.barh(np.arange(self.exp.panel_n) + 0.75, cum_germ_data[-1, :] / np.array(p_totals), height=0.5)
# #     ax4.set_yticks(range(1, 1 + self.exp.panel_n))
# #     ax4.set_ylim([0.5, self.exp.panel_n + .5])
# #     ax4.set_xlim([0., 1.])
# #     ax4.set_ylabel("Panel number")
# #     ax4.set_xlabel("Germinated proportion")
# #     ax4.set_title("Proportion germinated")
# #
# #     fig.savefig(pj(self.exp_results_dir, "results.jpg"))
# #     plt.close(fig)
# #
# #     img_index = np.arange(n_frames) + self.exp.start_img
# #
# #     if has_date:
# #         times_index = []
# #         for _i in img_index:
# #             curr_dt = s_to_datetime(self.imgs[_i])
# #             times_index.append(hours_between(start_dt, curr_dt))
# #
# #         times_index = np.array(times_index).reshape(-1, 1)
# #         cum_germ_data = np.hstack([times_index, cum_germ_data])
# #
# #     print(cum_germ_data.shape)
# #
# #     df = pd.DataFrame(data=cum_germ_data)
# #     df.index = img_index
# #
# #     if has_date:
# #         df.columns = ["Time"] + [str(i) for i in range(1, self.exp.panel_n + 1)]
# #         df.loc['Total seeds', 1:] = p_totals
# #     else:
# #         df.columns = [str(i) for i in range(1, self.exp.panel_n + 1)]
# #         df.loc['Total seeds', :] = p_totals
# #
# #     print(df.columns)
# #
# #     df.to_csv(pj(
# #         self.exp_results_dir,
# #         "panel_germinated_cumulative.csv"
# #     ))
# #
# # def _quantify_first_frame(self, proprtions):
# #     """ Quantify the seed data from the first frame.
# #     To quantify:
# #         - total seed number
# #         - seeds analysed
# #         - initial seed size
# #         - initial seed roundness
# #         - width/height ratio
# #         - RGB mean
# #         - germ rate at various percents
# #         - seed x, y
# #     """
# #
# #     # Only use date information if it is contained in the filename
# #     has_date = check_files_have_date(self.imgs[0])
# #     start_dt = None
# #     if has_date:
# #         start_dt = s_to_datetime(self.imgs[0])
# #
# #     img_f = imageio.imread(self.all_imgs[self.exp.start_img])
# #     f_masks = np.load(self.exp_masks_dir_frame % (self.exp.start_img), allow_pickle=True)
# #
# #     img_l = imageio.imread(self.all_imgs[-1])
# #     l_masks = np.load(self.exp_masks_dir_frame % (len(self.all_imgs) - 1), allow_pickle=True)
# #
# #     all_panel_data = []
# #
# #     # Panel analysis
# #     for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):
# #
# #         with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
# #             germ_d = json.load(fh)
# #
# #         germinated = []
# #         for j in range(1, len(germ_d.keys()) + 1):
# #             germinated.append(germ_d[str(j)])
# #         germinated = np.vstack(germinated)
# #
# #         cum_germ = self._get_cumulative_germ(germinated, win=7)[0].astype('f')
# #         cum_germ /= len(p_rprops)
# #
# #         germ_pro_total = cum_germ[-1]
# #
# #         prop_idxs = []
# #         for pro in proprtions:
# #             if (cum_germ > pro).any():
# #                 pos_idx = np.argmax(cum_germ >= pro) + self.exp.start_img
# #                 if has_date:
# #                     curr_dt = s_to_datetime(self.imgs[pos_idx])
# #                     pos_idx = hours_between(start_dt, curr_dt)
# #             else:
# #                 pos_idx = 'n/a'
# #             prop_idxs.append(pos_idx)
# #
# #         # print prop_idxs
# #
# #         p_f_img = self.panel_list[p_idx].get_bbox_image(img_f)
# #         p_f_mask = f_masks[p_idx]
# #
# #         p_l_img = self.panel_list[p_idx].get_bbox_image(img_l)
# #         p_l_mask = l_masks[p_idx]
# #
# #
# #         p_f_mask = np.hstack((p_f_mask, np.zeros((p_f_mask.shape[0], 1)))).astype('uint16')
# #         p_l_mask = np.hstack((p_l_mask, np.zeros((p_l_mask.shape[0], 1)))).astype('uint16')
# #         f_rgb_mu = p_f_img[p_f_mask].mean(axis=0)
# #         l_rgb_mu = p_l_img[p_l_mask].mean(axis=0)
# #         f_rgb_mu = tuple(np.round(f_rgb_mu).astype('i'))
# #         l_rgb_mu = tuple(np.round(l_rgb_mu).astype('i'))
# #
# #         # print "init_rgb", f_rgb_mu
# #         # print "germ_rgb", l_rgb_mu
# #
# #         avg_feas = []
# #         for rp in p_rprops:
# #             min_row, min_col, max_row, max_col = rp.bbox
# #             w = float(max_col - min_col)
# #             h = float(max_row - min_row)
# #             whr = w / h
# #             avg_feas.append([w, h, whr, rp.area, rp.eccentricity])
# #
# #         avg_feas = np.vstack(avg_feas)
# #         avg_feas_mu = avg_feas.mean(axis=0)
# #
# #         panel_data = [p_idx + 1, len(p_rprops)]
# #         panel_data += np.round(avg_feas_mu, 2).tolist()
# #         panel_data += [f_rgb_mu, l_rgb_mu]
# #         panel_data += prop_idxs
# #         panel_data += [round(germ_pro_total, 2)]
# #
# #         all_panel_data.append(panel_data)
# #
# #     columns = [
# #         'panel_ID',
# #         'total_seeds',
# #         'avg_width',
# #         'avg_height',
# #         'avg_wh_ratio',
# #         'avg_area',
# #         'avg_eccentricity',
# #         'avg_initial_rgb',
# #         'avg_final_rgb',
# #     ]
# #     columns.extend(['germ_%d%%' % (100 * prop) for prop in proprtions])
# #     columns.append('total_germ_%')
# #
# #     df = pd.DataFrame(all_panel_data, columns=columns)
# #     df.to_csv(pj(self.exp_results_dir, "overall_results.csv"), index=False)
# #
# #     # Seed analysis
# #     all_seed_results = []
# #
# #     panel_seed_idxs = {}
# #
# #     for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):
# #
# #         panel_seed_idxs[int(p_idx)] = []
# #
# #         with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
# #             germ_d = json.load(fh)
# #
# #         germinated = []
# #         for j in range(1, len(germ_d.keys()) + 1):
# #             germinated.append(germ_d[str(j)])
# #         germinated = np.vstack(germinated)
# #
# #         cum_germ, germ_proc = self._get_cumulative_germ(germinated, win=7)
# #
# #         for seed_rp in p_rprops:
# #
# #             germ_row = germ_proc[seed_rp.label - 1]
# #
# #             germ_idx = 'n/a'
# #             if germ_row.any():
# #                 germ_idx = np.argmax(germ_row) + self.exp.start_img
# #
# #             min_row, min_col, max_row, max_col = seed_rp.bbox
# #             w = float(max_col - min_col)
# #             h = float(max_row - min_row)
# #             whr = w / h
# #
# #             if germ_idx == 'n/a':
# #                 germ_time = 'n/a'
# #             else:
# #                 if has_date:
# #                     curr_dt = s_to_datetime(self.imgs[germ_idx])
# #                     germ_time = hours_between(start_dt, curr_dt)
# #                 else:
# #                     germ_time = germ_idx
# #
# #             seed_result = [
# #                 p_idx + 1,
# #                 seed_rp.label,
# #                 int(w),
# #                 int(h),
# #                 round(whr, 2),
# #                 int(seed_rp.area),
# #                 round(seed_rp.eccentricity, 2),
# #                 # (0,0,0),
# #                 # (0,0,0),
# #                 germ_idx,
# #                 germ_time,
# #                 ]
# #
# #             if germ_idx == 'n/a':
# #                 germ_idx = -1
# #
# #             panel_seed_idxs[int(p_idx)].append((
# #                 int(germ_idx),
# #                 int(seed_rp.centroid[0]),
# #                 int(seed_rp.centroid[1])
# #             ))
# #             all_seed_results.append(seed_result)
# #
# #     columns = [
# #         'panel_ID',
# #         'seed_ID',
# #         'width',
# #         'height',
# #         'wh_ratio',
# #         'area',
# #         'eccentricity',
# #         # 'initial_rgb',
# #         # 'final_rgb',
# #         'germ_point',
# #         'germ_time' if has_date else 'germ_image_number',
# #     ]
# #
# #     df = pd.DataFrame(all_seed_results, columns=columns)
# #     df.to_csv(pj(self.exp_results_dir, "panel_results.csv"), index=False)
# #
# #     with open(pj(self.exp_results_dir, "panel_seed_idxs.json"), "w") as fh:
# #         json.dump(panel_seed_idxs, fh)
# #
# #
# #
# #
# #     def run(self):
# #         print("Processor started")
# #
# #         if self.running:
# #
# #             start = time.time()
# #
# #             try:
# #                 self._save_init_image(self.imgs[self.exp.start_img])
# #
# #                 self._extract_panels(self.imgs[self.exp.start_img], self.core.chunk_no, self.core.chunk_reverse)
# #
# #                 self.app.status_string.set("Saving contour image")
# #
# #                 self._save_contour_image(self.imgs[self.exp.start_img])
# #
# #                 self.app.status_string.set("Training background removal clfs")
# #
# #                 print(self.exp.bg_remover)
# #                 if self.exp.bg_remover == "SGD":
# #                     self._train_clfs([("sgd", SGDClassifier(max_iter=10000, tol=1e-3, n_jobs=-1))])
# #                     # ("sgd_avg", SGDClassifier(average=True)),
# #                     # ("sag", LogisticRegression(solver='sag', \
# #                     # tol=1e-1, C=1.e4 / X_train.shape[0]))
# #                     self.app.status_string.set("Removing background")
# #                     self._remove_background()
# #                 elif self.exp.bg_remover == "RandF":
# #                     self._train_clfs([("randf", RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1))])
# #                     self.app.status_string.set("Removing background")
# #                     self._remove_background()
# #                 else:
# #                     print(".... unknown BG classifier")
# #
# #                 self.app.status_string.set("Labelling seeds")
# #                 self._label_seeds()
# #
# #                 self.app.status_string.set("Performing classification")
# #                 self._perform_classification()
# #
# #                 self.app.status_string.set("Analysing results")
# #
# #                 self._analyse_results(self.core.proportions)
# #
# #                 self.app.status_string.set("Quantifying initial seed data")
# #                 self._quantify_frames(self.core.proportions)
# #
# #                 self.app.status_string.set("Finished processing")
# #                 self.exp.status = "Finished"
# #
# #                 print()
# #                 print(time.time() - start)
# #                 print()
# #
# #                 print("Finished")
# #
# #             except Exception as e:
# #                 raise e
# #                 #self.exp.status = "Error"
# #                 #self.app.status_string.set("Error whilst processing")
# #                 #print("Exception args: " + str(e.args))
# #
# #             self.running = False
# #
# #             self.core.stop_processor(self.exp.eid)
# #
# #     def die(self):
# #         self.running = False
# #
# #
# #
# #     def _perform_classification(self):
# #         """ Also need to quantify whether the seed merges, and whether it has
# #         moved.
# #         """
# #
# #         if len(glob.glob(pj(self.exp_results_dir, "germ_panel_*.json"))) >= self.exp.panel_n:
# #             print("Already analysed data")
# #             # print "Continuing for testing..."
# #             return
# #
# #         if self.all_masks is None:
# #             self.all_masks = []
# #             for i in range(self.exp.start_img, len(self.imgs)):
# #                 data = np.load(self.exp_masks_dir_frame % (i), allow_pickle=True)
# #                 self.all_masks.append(data)
# #
# #         for panel_idx, panel_object in enumerate(self.panel_list):
# #
# #             # if panel_idx != 4:
# #             # print "Continuing"
# #             # continue
# #
# #             try:
# #                 print("panel %d" % (panel_idx))
# #
# #                 # this is the tuple of the labelled arrays genereated from mesurements.
# #                 # label and the regionprops generated by the measurements.regionprops
# #                 panel_labels, panel_regionprops = self.panel_l_rprops[panel_idx]
# #
# #                 # extract all the masks for the specific panel, for every image.
# #                 p_masks = [el[panel_idx] for el in self.all_masks]
# #                 self.spp_processor.use_colour = self.exp.use_colour
# #
# #                 panel_germ = self.spp_processor._classify(
# #                     panel_idx,
# #                     panel_object,
# #                     self.all_imgs[self.exp.start_img:],
# #                     panel_labels,
# #                     panel_regionprops,
# #                     p_masks
# #                 )
# #
# #                 print("panel germ length " + str(len(panel_germ)))
# #
# #                 out_f = pj(
# #                     self.exp_results_dir,
# #                     "germ_panel_%d.json" % (panel_idx)
# #                 )
# #
# #                 with open(out_f, "w") as fh:
# #                     json.dump(panel_germ, fh)
# #
# #
# #             except Exception as e:
# #                 print("Could not run panel %d" % (panel_idx))
# #                 print(e)
# #                 traceback.print_exc()
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # all_im = _save_init_image(exp_images_dir)
# # panel_list, rprops = _extract_panels(all_im[0], 2, True, yuv_range, n_panels)
# # panels_mask = _save_contour_image(all_im[0], exp_images_dir)
# # clf_in = _train_clfs(list([LGBM, SGD, LR]), all_im, exp_images_dir, panels_mask, yuv_range)
# # all_masks = _remove_background(all_im, panel_list, clf_in)
# # total_rprops = _label_seeds(n_panels, all_im, all_masks, panel_list, n_seeds, n_rows, n_cols)
# # panel_germ = _perform_classification(panel_list, total_rprops, all_im, all_masks)
#
# # X_stats_total = []
# #
# # for j in range(len(total_rprops)):
# #     print(j+1)
# #     x = total_rprops[j]
# #     X_stats = np.zeros((len(x[0][1])+len(x[1][1])+len(x[2][1])+len(x[3][1])+len(x[4][1])+len(x[5][1]), 11))
# #     counter = 0
# #     for i in range(len(x)):
# #         x0 = x[i][1]
# #     #     if len(x0) > 64:
# #     #         n_drop = len(x0) - 64
# #     #         areas = np.zeros((len(x0), 1))
# #     #         for p in range(len(x0)):
# #     #             areas[p, 0] = x0[p].area
# #     #         trim_mean = stats.trim_mean(areas, 0.05)
# #     #         distances = np.abs(areas - trim_mean)
# #     #         ind = np.argpartition(distances.T, -n_drop)[0, -n_drop:]
# #     #         ind = np.sort(ind)[::-1]
# #     #         for id in ind:
# #     #             del x0[id]
# #     #             print("DELETE")
# #         for k in range(len(x0)):
# #             X_stats[counter, :] = [i+1, k+1, x0[k].area, x0[k].eccentricity, x0[k].equivalent_diameter, x0[k].extent,
# #                                    x0[k].major_axis_length, x0[k].minor_axis_length, x0[k].perimeter, x0[k].solidity, x0[k].convex_area]
# #             counter = counter + 1
# #     X_stats_total.append(X_stats)
# #     X_stats = pd.DataFrame(X_stats, columns=['Panel Index', 'Seed Index', 'Seed Area', 'Seed Eccentricity', 'Seed Equivalent Diameter', 'Seed Extent', 'Seed Major Axis Length', 'Seed Minor Axis Length', 'Seed Perimeter', 'Seed Solidity', 'Seed Convex Area'])
# #     X_stats['Panel Index'] = X_stats['Panel Index'].astype('uint8')
# #     X_stats['Seed Index'] = X_stats['Seed Index'].astype('uint8')
# #     filename = exp_images_dir + "/Results/" + "panel_stats_" + str(j) + ".csv"
# #     X_stats.to_csv(filename, index=False)
# #
# #
# # seed_stats = np.zeros((1, 10))
# # for i in range(len(X_stats_total)):
# #     c = X_stats_total[i]
# #     x = np.concatenate((c[:, 2:], np.full(shape=(c.shape[0], 1), fill_value=i)), axis=1)
# #     seed_stats = np.concatenate((seed_stats, x))
# #
# # seed_stats = np.delete(seed_stats, 0, axis=0)
# # stats_over_time = pd.DataFrame(seed_stats, columns=['Seed Area', 'Seed Eccentricity', 'Seed Equivalent Diameter', 'Seed Extent', 'Seed Major Axis Length', 'Seed Minor Axis Length', 'Seed Perimeter', 'Seed Solidity', 'Seed Convex Area', 'Image Index'])
# # stats_over_time['Image Index'] = stats_over_time['Image Index'].astype('uint8')
# #
#
# # ax = sn.lineplot(x="Image Index", y="Seed Area", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Area", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Eccentricity", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Equivalent Diameter", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Extent", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Major Axis Length", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Minor Axis Length", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Perimeter", data=stats_over_time)
# # plt.show()
# # ax = sn.lineplot(x="Image Index", y="Seed Solidity", data=stats_over_time)
# # plt.show()
# #
# # stats_over_time.to_csv('CornStats.csv', index=False)
#
#
#
#
#
#
#
#
#
#
#
barley_stats = pd.read_csv('BarleyStats.csv')
brassica_stats = pd.read_csv('BrassicaStats.csv')
corn_stats = pd.read_csv('CornStats.csv')
pepper_stats = pd.read_csv('PepperStats.csv')
tomato_stats = pd.read_csv('TomatoStats.csv')
wheat_stats = pd.read_csv('WheatStats.csv')


brassica_stats = brassica_stats.iloc[1126:12212, :]
corn_stats = corn_stats.iloc[105:1148, :]
pepper_stats = pepper_stats.iloc[743:13583, :]
tomato_stats = tomato_stats.iloc[1146:11058, :]


def remove_small_items(df):
    print(df.shape[0])
    minidx = df['Image Index'].min()
    x_0 = df.loc[df['Image Index'] < minidx+3]
    minarea = x_0['Seed Area'].quantile(0.05)
    df = df.loc[df['Seed Area'] > 0.85*minarea]
    print(df.shape[0])
    return df


barley_stats = remove_small_items(barley_stats)
brassica_stats = remove_small_items(brassica_stats)
corn_stats = remove_small_items(corn_stats)
pepper_stats = remove_small_items(pepper_stats)
tomato_stats = remove_small_items(tomato_stats)
wheat_stats = remove_small_items(wheat_stats)

wheat_stats = wheat_stats.loc[wheat_stats['Seed Area'] < 10000]


def remove_outliers(df):
    x = df.iloc[:, :8].values
    print(x.shape)
    low = np.percentile(x, 2, axis=0)
    high = np.percentile(x, 98, axis=0)
    x1 = np.logical_and(x > low, x < high)
    x1 = np.where(x1.all(axis=1) == True)[0]
    print(x1.shape)
    x = x[x1, :]
    print(x.shape)
    x2 = df.iloc[:x.shape[0], 8].values.reshape((x.shape[0], 1))
    print(x2.shape)
    x = np.hstack((x, x2))
    print(x.shape)
    df1 = pd.DataFrame(data=x, columns=df.columns)
    return df1


def remove_outliers_area(df):
    x = df.iloc[:, :10].values
    df_new = pd.DataFrame(df.iloc[0, :]).T
    for idx in df['Image Index'].unique():
        X_idx = df.loc[df['Image Index'] == idx]
        J = stats.zscore(X_idx.iloc[:, 0])
        X_idx_new = X_idx[np.abs(J) < 3]
        df_new = pd.concat((df_new, X_idx_new), axis=0)
    df_new = df_new.reset_index(drop=True)
    df_new.drop(0, inplace=True)
    return df_new


def remove_outliers_minor_axis(df):
    x = df.iloc[:, :10].values
    df_new = pd.DataFrame(df.iloc[5, :]).T
    for idx in df['Image Index'].unique():
        X_idx = df.loc[df['Image Index'] == idx]
        J = stats.zscore(X_idx.iloc[:, 5])
        X_idx_new = X_idx[np.abs(J) < 3]
        df_new = pd.concat((df_new, X_idx_new), axis=0)
    df_new = df_new.reset_index(drop=True)
    df_new.drop(0, inplace=True)
    return df_new


def remove_outliers_convex_area(df):
    x = df.iloc[:, :10].values
    df_new = pd.DataFrame(df.iloc[8, :]).T
    for idx in df['Image Index'].unique():
        X_idx = df.loc[df['Image Index'] == idx]
        J = stats.zscore(X_idx.iloc[:, 5])
        X_idx_new = X_idx[np.abs(J) < 3]
        df_new = pd.concat((df_new, X_idx_new), axis=0)
    df_new = df_new.reset_index(drop=True)
    df_new.drop(0, inplace=True)
    return df_new


def get_ratio(df):
    x = df.values
    x_major = df.iloc[:, 4].values
    x_minor = df.iloc[:, 5].values
    x_ratio = (x_minor/x_major)
    x_ratio = np.reshape(x_ratio, (x_minor.shape[0], 1))
    df1 = pd.DataFrame(data=x, columns=df.columns)
    df1['Seed Minor/Major Ratio'] = x_ratio
    return df1


print("Barley with outliers: ", barley_stats.shape[0])
barley_stats = remove_outliers_area(barley_stats)
barley_stats = remove_outliers_minor_axis(barley_stats)
barley_stats = remove_outliers_convex_area(barley_stats)
barley_stats = barley_stats.loc[barley_stats['Seed Area'] > 1000]
print("Barley without outliers: ", barley_stats.shape[0])
print("Brassica with outliers: ", brassica_stats.shape[0])
brassica_stats = remove_outliers_area(brassica_stats)
brassica_stats = remove_outliers_minor_axis(brassica_stats)
brassica_stats = remove_outliers_convex_area(brassica_stats)
print("Brassica without outliers: ", brassica_stats.shape[0])
print("Corn with outliers: ", corn_stats.shape[0])
corn_stats = remove_outliers_area(corn_stats)
corn_stats = remove_outliers_minor_axis(corn_stats)
corn_stats = remove_outliers_convex_area(corn_stats)
print("Corn without outliers: ", corn_stats.shape[0])
print("Pepper with outliers: ", pepper_stats.shape[0])
pepper_stats = remove_outliers_area(pepper_stats)
pepper_stats = remove_outliers_minor_axis(pepper_stats)
pepper_stats = remove_outliers_convex_area(pepper_stats)
print("Pepper without outliers: ", pepper_stats.shape[0])
print("Tomato with outliers: ", tomato_stats.shape[0])
tomato_stats = remove_outliers_area(tomato_stats)
tomato_stats = remove_outliers_minor_axis(tomato_stats)
tomato_stats = remove_outliers_convex_area(tomato_stats)
print("Tomato without outliers: ", tomato_stats.shape[0])
print("Wheat with outliers: ", wheat_stats.shape[0])
wheat_stats = remove_outliers_area(wheat_stats)
wheat_stats = remove_outliers_minor_axis(wheat_stats)
wheat_stats = remove_outliers_convex_area(wheat_stats)
print("Wheat without outliers: ", wheat_stats.shape[0])

barley_stats = get_ratio(barley_stats)
brassica_stats = get_ratio(brassica_stats)
corn_stats = get_ratio(corn_stats)
pepper_stats = get_ratio(pepper_stats)
tomato_stats = get_ratio(tomato_stats)
wheat_stats = get_ratio(wheat_stats)

pepper_stats = pepper_stats.iloc[743:12970, :]
tomato_stats = tomato_stats.iloc[:9683, :]
corn_stats = corn_stats.iloc[:, :]
brassica_stats = brassica_stats.iloc[1126:12970, :]
barley_stats = barley_stats.iloc[:3478, :]

barley_stats['Seed Type'] = 'Barley (n=40)'
brassica_stats['Seed Type'] = 'Brassica (n=384)'
corn_stats['Seed Type'] = 'Corn (n=70)'
pepper_stats['Seed Type'] = 'Pepper (n=384)'
tomato_stats['Seed Type'] = 'Tomato (n=384)'
wheat_stats['Seed Type'] = 'Wheat (n=40)'

scaler = MinMaxScaler()

# barley_stats.iloc[:, :9] = scaler.fit_transform(barley_stats.iloc[:, :9])
# brassica_stats.iloc[:, :9] = scaler.fit_transform(brassica_stats.iloc[:, :9])
# corn_stats.iloc[:, :9] = scaler.fit_transform(corn_stats.iloc[:, :9])
# pepper_stats.iloc[:, :9] = scaler.fit_transform(pepper_stats.iloc[:, :9])
# tomato_stats.iloc[:, :9] = scaler.fit_transform(tomato_stats.iloc[:, :9])
# wheat_stats.iloc[:, :9] = scaler.fit_transform(wheat_stats.iloc[:, :9])

scaler = MinMaxScaler(feature_range=(0, 100))
wheat_stats = wheat_stats.iloc[:2670, :]
pepper_stats = pepper_stats.iloc[:-7, :]
brassica_stats = brassica_stats.iloc[:-119, :]

# barley_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(barley_stats.iloc[:, 10].values, (-1, 1)))
# brassica_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(brassica_stats.iloc[:, 10].values, (-1, 1)))
# corn_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(corn_stats.iloc[:, 10].values, (-1, 1)))
# pepper_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(pepper_stats.iloc[:, 10].values, (-1, 1)))
# tomato_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(tomato_stats.iloc[:, 10].values, (-1, 1)))
# wheat_stats.iloc[:, 10] = scaler.fit_transform(np.reshape(wheat_stats.iloc[:, 10].values, (-1, 1)))

barley_stats.iloc[:, 9] = 100*(barley_stats.iloc[:, 9] - barley_stats.iloc[:, 9].min())/(barley_stats.iloc[:, 9].max() - barley_stats.iloc[:, 9].min())
brassica_stats.iloc[:, 9] = 100*(brassica_stats.iloc[:, 9] - brassica_stats.iloc[:, 9].min())/(brassica_stats.iloc[:, 9].max() - brassica_stats.iloc[:, 9].min())
corn_stats.iloc[:, 9] = 100*(corn_stats.iloc[:, 9] - corn_stats.iloc[:, 9].min())/(corn_stats.iloc[:, 9].max() - corn_stats.iloc[:, 9].min())
pepper_stats.iloc[:, 9] = 100*(pepper_stats.iloc[:, 9] - pepper_stats.iloc[:, 9].min())/(pepper_stats.iloc[:, 9].max() - pepper_stats.iloc[:, 9].min())
tomato_stats.iloc[:, 9] = 100*(tomato_stats.iloc[:, 9] - tomato_stats.iloc[:, 9].min())/(tomato_stats.iloc[:, 9].max() - tomato_stats.iloc[:, 9].min())
wheat_stats.iloc[:, 9] = 100*(wheat_stats.iloc[:, 9] - wheat_stats.iloc[:, 9].min())/(wheat_stats.iloc[:, 9].max() - wheat_stats.iloc[:, 9].min())

# all_stats = pd.concat((barley_stats, brassica_stats, corn_stats, pepper_stats, tomato_stats, wheat_stats), axis=0)
fruit_stats = pd.concat((brassica_stats, pepper_stats, tomato_stats), axis=0)
cereal_stats = pd.concat((barley_stats, corn_stats, wheat_stats), axis=0)

# scaler = MinMaxScaler(feature_range=(0, 100))

# all_stats['Image Index'] = scaler.fit_transform(all_stats['Image Index'])

labelsize = 14
ticksize = 11
legendsize = 10
xlabel = "Duration through experiment (%)"

ax = sn.lineplot(x='Image Index', y='Seed Area', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed area', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitArea.png")
ax = sn.lineplot(x='Image Index', y='Seed Eccentricity', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed eccentricity', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitEccentricity.png")
ax = sn.lineplot(x='Image Index', y='Seed Extent', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed extent', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitExtent.png")
# ax = sn.lineplot(x='Image Index', y='Seed Minor Axis Length', data=fruit_stats, hue='Seed Type')
# plt.xlabel(xlabel=xlabel, fontsize=labelsize)
# plt.ylabel(ylabel='Seed minor axis length', fontsize=labelsize)
# plt.yticks(size=ticksize)
# plt.xticks(size=ticksize)
# plt.legend(loc=0, fontsize=legendsize)
# plt.show()
# ax.figure.savefig("FruitMinorAxisLength.png")
ax = sn.lineplot(x='Image Index', y='Seed Major Axis Length', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed major axis length', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitMajorAxisLength.png")
ax = sn.lineplot(x='Image Index', y='Seed Perimeter', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed perimeter', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitPerimeter.png")
ax = sn.lineplot(x='Image Index', y='Seed Solidity', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed solidity', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitSolidity.png")
ax = sn.lineplot(x='Image Index', y='Seed Minor/Major Ratio', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed minor/major ratio', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitMajorMinorRatio.png")
ax = sn.lineplot(x='Image Index', y='Seed Convex Area', data=fruit_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed convex area', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("FruitConvexArea.png")


ax = sn.lineplot(x='Image Index', y='Seed Area', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed area', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealArea.png")
ax = sn.lineplot(x='Image Index', y='Seed Eccentricity', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed eccentricity', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealEccentricity.png")
ax = sn.lineplot(x='Image Index', y='Seed Extent', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed extent', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealExtent.png")
# ax = sn.lineplot(x='Image Index', y='Seed Minor Axis Length', data=cereal_stats, hue='Seed Type')
# plt.xlabel(xlabel=xlabel, fontsize=labelsize)
# plt.ylabel(ylabel='Seed minor axis length', fontsize=labelsize)
# plt.yticks(size=ticksize)
# plt.xticks(size=ticksize)
# plt.legend(loc=0, fontsize=legendsize)
# plt.show()
# ax.figure.savefig("CerealMinorAxisLength.png")
ax = sn.lineplot(x='Image Index', y='Seed Major Axis Length', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed major axis length', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealMajorAxisLength.png")
ax = sn.lineplot(x='Image Index', y='Seed Perimeter', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed perimeter', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealPerimeter.png")
ax = sn.lineplot(x='Image Index', y='Seed Solidity', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed solidity', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealSolidity.png")
ax = sn.lineplot(x='Image Index', y='Seed Minor/Major Ratio', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed minor/major ratio', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealMajorMinorRatio.png")
ax = sn.lineplot(x='Image Index', y='Seed Convex Area', data=cereal_stats, hue='Seed Type')
plt.xlabel(xlabel=xlabel, fontsize=labelsize)
plt.ylabel(ylabel='Seed convex area', fontsize=labelsize)
plt.yticks(size=ticksize)
plt.xticks(size=ticksize)
plt.legend(loc=0, fontsize=legendsize)
plt.show()
ax.figure.savefig("CerealConvexArea.png")
