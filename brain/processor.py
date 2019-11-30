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
from tqdm import tqdm
import pandas as pd
import pickle
import threading
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
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# Machine learning and statistics imports.
from skimage.morphology import *
from skimage.segmentation import clear_border
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from imageio import imread
import warnings
warnings.filterwarnings("ignore")

# Enables ability to reproduce results
np.random.seed(0)
random.seed(0)

def get_crop_shape(target, refer):
    # Width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # Height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def create_unet(img_shape, num_class):
    # Define and return the U-Net model, e.g. number of layers, number of filters, activation function, etc.

    concat_axis = 3
    inputs = layers.Input(shape=img_shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
    up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
    up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
    up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
    up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = layers.Conv2D(num_class, (1, 1), activation='softmax')(conv9)

    model = models.Model(inputs=inputs, outputs=conv10)

    return model


def compare_pts(pt1, pt2):
    return pt1 > pt2


class ImageProcessor(threading.Thread):
    def __init__(self, core=None, app=None, experiment=None):
        """ Set and generate required experiment data variables. """
        super(ImageProcessor, self).__init__()

        self.core = core
        self.app = app
        self.exp = experiment  # type: Experiment
        self.running = True

        self.all_masks = []
        self.total_stats = []

        # Read image file names for the experiments, sort based on image number
        self.imgs = get_images_from_dir(self.exp.img_path)
        self.all_imgs = [pj(self.exp.img_path, el) for el in self.imgs]

        self.exp_masks_dir = pj(self.exp.exp_path, "masks")
        self.exp_masks_dir_frame = pj(self.exp_masks_dir, "frame_%d.npy")
        self.exp_images_dir = pj(self.exp.exp_path, "images")
        self.exp_results_dir = pj(self.exp.exp_path, "results")
        self.exp_gzdata_dir = pj(self.exp.exp_path, "gzdata")

        self.rprops = []
        self.all_rprops = []
        self.all_imgs_list = []
        for image_path in self.all_imgs:
            im = imread(image_path)
            self.all_imgs_list.append(im)

        self.yuv_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges.json"
        )

        with open(self.yuv_json_file) as fh:
            data = json.load(fh)

        self.yuv_low = np.array(data['low'])
        self.yuv_high = np.array(data['high'])

        try:
            self.spp_processor = copy.deepcopy(self.core.species_classes[
                                                   self.exp.species])
        except KeyError:
            print("No species module found for %s" % (self.exp.species))
            print("ought to use default, shouldn't occur as populate species list from these modules")
            print("consider adding parameters to the config if you're confident")

    def _run_check(self):
        if not self.running:
            pass

    def _save_init_image(self, img):
        # Saves the initial RGB image of the experiment
        out_f = pj(self.exp_images_dir, "init_img.jpg")
        if os.path.exists(out_f):
            return
        img_01 = self.all_imgs_list[0] / 255.
        fig = plt.figure(dpi=600)
        plt.imshow(img_01)
        fig.savefig(out_f)
        plt.close(fig)

    def _yuv_clip_image(self, img_f):
        # Returns a binary mask of an image using the manually set thresholds
        img = self.all_imgs_list[img_f]
        img_yuv = rgb2ycrcb(img)
        mask_img = in_range(img_yuv, self.yuv_low, self.yuv_high)
        return mask_img.astype(np.bool)

    def _yuv_clip_panel_image(self, img_f, p):
        # Returns a binary mask of an image of a panel using the individual panel thresholds
        img = self.all_imgs_list[img_f]
        img_yuv = rgb2ycrcb(img)
        self.yuv_panel_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges_{}.json".format(p)
        )

        if os.path.exists(self.yuv_panel_json_file):
            with open(self.yuv_panel_json_file) as fh:
                data = json.load(fh)
        else:
            with open(self.yuv_json_file) as fh:
                data = json.load(fh)

        self.yuv_panel_low = np.array(data['low'])
        self.yuv_panel_high = np.array(data['high'])
        mask_img = in_range(img_yuv, self.yuv_panel_low, self.yuv_panel_high)
        return mask_img.astype(np.bool)

    def _extract_panels(self, img, chunk_no, chunk_reverse, img_idx):
        # If panel data already exists, load it
        panel_data_f = os.path.join(self.exp_gzdata_dir, "panel_data.pkl")
        if os.path.exists(panel_data_f):
            with open(panel_data_f, 'rb') as fh:
                try:
                    self.panel_list = pickle.load(fh)
                    return
                except EOFError:
                    print("pickle is broken")

        # Obtain binary mask using the set YUV thresholds
        mask_img = self._yuv_clip_image(img_idx)
        mask_img = remove_small_objects(
                   fill_border(mask_img, 10, fillval=False),
                   min_size=1024
                   )
        mask_img_cleaned_copy = mask_img.copy()
        mask_img = erosion(binary_fill_holes(mask_img), disk(7))
        obj_only_mask = np.logical_and(mask_img, np.logical_not(mask_img_cleaned_copy))

        # Order the panels in the mask, count the number of panels found in the mask
        ordered_mask_img, n_panels = measurements.label(mask_img)
        # Returns region properties of each panel in a list
        rprops = regionprops(ordered_mask_img, coordinates='xy')
        # Remove items which are unlikely to be real panels
        rprops = [x for x in rprops if x.area > self.all_imgs_list[0].shape[0]*self.all_imgs_list[0].shape[1]/self.exp.panel_n/6]

        def get_mask_objects(idx, rp):
            tmp_mask = np.zeros(mask_img.shape)
            tmp_mask[ordered_mask_img == rp.label] = 1

            both_mask = np.logical_and(obj_only_mask, tmp_mask)
            both_mask = remove_small_objects(both_mask)
            _, panel_object_count = measurements.label(both_mask)  # , return_num=True)
            return panel_object_count, tmp_mask, both_mask

        rprops = [(rp, get_mask_objects(idx, rp)) for idx, rp in enumerate(rprops)]
        rprops = [[item[0], item[1][0], item[1][1], item[1][2]] for idx, item in enumerate(rprops)]
        rprops = sorted(rprops, key=itemgetter(1), reverse=True)

        # Check the panel has seeds in it
        panels = [(rp, rp.centroid[0], rp.centroid[1], tmp, both) for rp, _, tmp, both in rprops[:self.exp.panel_n]]

        # Sort panels based on y first, then x
        panels = sorted(panels, key=itemgetter(1))
        panels = chunks(panels, chunk_no)
        panels = [sorted(p, key=itemgetter(2), reverse=chunk_reverse) for p in panels]
        panels = list(chain(*panels))

        # Set mask, where 1 is top left, 2 is top right, 3 is middle left, etc
        panel_list = []
        tmp_list = []
        both_list = []
        for idx in range(len(panels)):
            rp, _, _, tmp, both = panels[idx]
            new_mask = np.zeros(mask_img.shape)
            new_mask[ordered_mask_img == rp.label] = 1
            panel_list.append(Panel(idx + 1, new_mask.astype(np.bool), rp.centroid, rp.bbox))
            tmp_list.append(tmp)
            both_list.append(both)

        self.panel_list = panel_list
        self.rprops = rprops
        for i in range(len(tmp_list)):
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Mask for panel {}'.format(i + 1))
            ax[0].imshow(tmp_list[i])
            ax[1].imshow(both_list[i])
            fig.savefig(pj(self.exp_images_dir, "mask_img_{}.jpg".format(i + 1)))
            plt.close(fig)

        with open(panel_data_f, "wb") as fh:
            pickle.dump(self.panel_list, fh)

    def _save_contour_image(self):
        fig = plt.figure(dpi=600)
        img_01 = self.all_imgs_list[self.exp.start_img]
        # plt.imshow(img_01)

        img_full_mask = np.zeros(img_01.shape[:-1])
        for p in self.panel_list:
            plt.annotate(str(p.label), xy=p.centroid[::-1], color='r', fontsize=20)
            min_row, min_col, max_row, max_col = p.bbox
            img_full_mask[min_row:max_row, min_col:max_col] += p.mask_crop

        self.panels_mask = img_full_mask.astype(np.bool)

        out_f = pj(self.exp_images_dir, "img_panels.jpg")
        if os.path.exists(out_f):
            return

        plt.gca().invert_yaxis()
        plt.contour(img_full_mask, [0.5], colors='r')
        fig.savefig(out_f)
        plt.close(fig)

        fig = plt.figure(dpi=600)
        plt.imshow(self.panels_mask)
        fig.savefig(pj(self.exp_images_dir, "panels_mask.jpg"))
        plt.close(fig)

    def _gmm_update_gmm(self, gmm, y):
        y_pred = gmm.predict(y)

        # Previously used 1 / # examples, but parameters are updated too quickly
        alpha = 0.25 / float(y.shape[0])

        for x, m in zip(y, y_pred):
            obs = np.zeros(gmm.n_components)
            obs[m] = 1.
            delta_m = x - gmm.means_[m]
            new_weights = gmm.weights_ + alpha * (obs - gmm.weights_)
            new_mean = gmm.means_[m] + (alpha / gmm.means_[m]) * delta_m
            new_std = gmm.covariances_[m] + (alpha / gmm.means_[m]) * (
                delta_m.T * delta_m - np.power(gmm.covariances_[m], 2))
            gmm.weights_ = new_weights
            gmm.means_[m] = new_mean
            gmm.covariances_[m] = new_std

    def _gmm_get_TCD(self, X, E, s, b):
        A = np.sum((X * E) / np.power(s, 2), axis=1)
        B = np.sum(np.power(E / s, 2))
        alpha = A / B

        alpha_tiled = np.repeat(alpha.reshape(alpha.shape[0], 1), 3, axis=1)
        inner = np.power((X - (E * alpha_tiled)) / s, 2)
        all_NCD = np.sqrt(np.sum(inner, axis=1)) / b
        TCD = np.percentile(all_NCD, 99.75)
        return TCD

    def _train_gmm_clfs(self):
        gmm_clf_f = pj(self.exp_gzdata_dir, "gmm_clf.pkl")
        if os.path.exists(gmm_clf_f):
            with open(gmm_clf_f, 'rb') as fh:
                self.classifiers = pickle.load(fh)
            return

        curr_img = self.all_imgs_list[self.exp.start_img] / 255.

        curr_mask = self._yuv_clip_image(self.exp.start_img)

        curr_mask = dilation(curr_mask, disk(2))

        bg_mask3 = np.dstack([np.logical_and(curr_mask, self.panels_mask)] * 3)
        bg_rgb_pixels = curr_img * bg_mask3

        # Get all of the bg pixels
        bg_rgb_pixels = flatten_img(bg_rgb_pixels)
        bg_rgb_pixels = bg_rgb_pixels[bg_rgb_pixels.all(axis=1), :]
        bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
        bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain, replace=True)
        X = bg_rgb_pixels[bg_retain, :]

        blue_E, blue_s = X.mean(axis=0), X.std(axis=0)

        alpha = flatBD(X, blue_E, blue_s)
        a = np.sqrt(np.power(alpha - 1, 2) / X.shape[0])
        b = np.sqrt(np.power(flatCD(X, blue_E, blue_s), 2) / X.shape[0])

        TCD = self._gmm_get_TCD(X, blue_E, blue_s, b)
        print("Training GMM background remover...")
        bg_gmm = GaussianMixture(n_components=3, random_state=0)
        bg_gmm.fit(X)

        thresh = np.percentile(bg_gmm.score(X), 1.)

        new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)

        self.classifiers = [bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E]

        with open(gmm_clf_f, "wb") as fh:
            pickle.dump(self.classifiers, fh)

    def _gmm_remove_background(self):
        if len(os.listdir(self.exp_masks_dir)) >= ((self.exp.end_img - self.exp.start_img) - self.exp.start_img):
            return

        bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E = self.classifiers

        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img - self.exp.start_img):
            return

        # 2d array of all the masks that are indexed first by image, then by panel.
        self.all_masks = []
        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img = self.all_imgs_list[idx] / 255.

            img_masks = []
            # Generate the predicted mask for each panel and add them to the mask list.
            for p in self.panel_list:
                panel_img = p.get_cropped_image(img)
                pp_predicted = NCD(panel_img, new_E, blue_s, b) > TCD
                pp_predicted = pp_predicted.astype(np.bool)
                img_masks.append(pp_predicted)

            y = flatten_img(img)
            predicted = bg_gmm.score_samples(y)
            if new_E[0] < 1e-4:
                with open(self.exp_masks_dir_frame % (idx), "wb") as fh:
                    np.save(fh, img_masks)
                self.all_masks.append(img_masks)
                continue
            bg_retain = predicted > thresh
            y_bg = y[bg_retain, :]
            retain = random.choice(y_bg.shape[0], min(y_bg.shape[0], 100000), replace=False)
            y_bg = y_bg[retain, :]
            self._gmm_update_gmm(bg_gmm, y_bg)

            new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)

            print(new_E)

            with open(self.exp_masks_dir_frame % (idx), "wb") as fh:
                np.save(fh, img_masks)

            self.all_masks.append(img_masks)

    def _ensemble_predict(self, clfs, X, p):
        y_pred = clfs[p.label - 1].predict(X)
        return y_pred

    def _train_clfs(self, clf_in):
        print("Classifier: ", self.exp.bg_remover)
        self.classifiers = []
        # For each panel in the list of panels, create training data then train the defined classifier
        for p in self.panel_list:
            print("Training classifier for panel {}".format(p.label))

            # Attempt to load classifiers if they already exist
            ensemble_clf_f = pj(self.exp_gzdata_dir, "ensemble_clf_{}.pkl".format(p.label))
            if os.path.exists(ensemble_clf_f):
                with open(ensemble_clf_f, 'rb') as fh:
                    self.classifiers = pickle.load(fh)

            if len(self.classifiers) == len(self.panel_list):
                print('Loaded previous classifiers successfully')
                return

            # 4 x 4 figure defined, each subplot will show one training mask
            fig, axarr = plt.subplots(4, 4)
            fig.suptitle('Training images for panel {}'.format(p.label))
            axarr = list(chain(*axarr))

            train_masks = []
            train_images = []

            # 16 training images selected as first 10 IDs, middle ID and last 5 IDs
            train_img_ids = list(range(self.exp.start_img, self.exp.start_img + 10))
            train_img_ids += [int((self.exp.end_img + self.exp.start_img) / 2) - 1, self.exp.end_img - 2,
                              self.exp.end_img - 1, self.exp.end_img - 3, self.exp.end_img - 4, self.exp.end_img - 5]

            # For each training image, training mask labels created using the yuv thresholds chosen at experiment setup,
            # training mask labels plotted in 4 x 4 figure
            for idx, img_i in enumerate(train_img_ids):
                curr_img = self.all_imgs_list[img_i][p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] / 255.
                train_images.append(curr_img)

                curr_mask = self._yuv_clip_panel_image(img_i, p.label)[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
                curr_mask = dilation(curr_mask, disk(2))
                train_masks.append(curr_mask.astype(np.bool))

                axarr[idx].imshow(curr_mask)
                axarr[idx].axis('off')

            # Figure showing 16 training images for this panel saved in images directory
            fig.savefig(pj(self.exp_images_dir, "train_imgs_panel_{}.jpg".format(p.label)))
            plt.close(fig)

            all_bg_pixels = []
            all_fg_pixels = []

            # For each training mask, get corresponding RGB values for background and foreground pixels,
            # and append to a list of all RGB values at foreground locations as well as a list of all RGB values
            # at background locations
            for idx, (mask, curr_img) in enumerate(zip(train_masks, train_images)):
                bg_mask3 = np.dstack([np.logical_and(mask, self.panels_mask[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]])] * 3)
                fg_mask3 = np.dstack([np.logical_and(np.logical_not(mask), self.panels_mask[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]])] * 3)

                bg_rgb_pixels = self._create_transformed_data(
                    curr_img * bg_mask3)
                fg_rgb_pixels = self._create_transformed_data(
                    curr_img * fg_mask3)

                all_bg_pixels.append(bg_rgb_pixels)
                all_fg_pixels.append(fg_rgb_pixels)

            bg_rgb_pixels = np.vstack(all_bg_pixels)
            fg_rgb_pixels = np.vstack(all_fg_pixels)

            # Concatenate training data from all images, X contains RGB values corresponding to Y label values (0 being
            # background, 1 being foreground)
            X = np.vstack([bg_rgb_pixels, fg_rgb_pixels])
            y = np.concatenate([
                np.zeros(bg_rgb_pixels.shape[0]),
                np.ones(fg_rgb_pixels.shape[0])
            ])

            # Train the classifier on this panel's training data
            self._train_clf(clf_in, ensemble_clf_f, X, y)

    def _train_unet(self):
        print("Classifier: ", self.exp.bg_remover)

        callbacks = [
            # EarlyStopping(patience=20, verbose=1),
            # ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
            # ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]

        images = self.all_imgs_list

        print("Selecting training data")

        train_img_ids = list(range(self.exp.start_img, self.exp.start_img + 10))
        train_img_ids += [int((self.exp.end_img + self.exp.start_img) / 2) - 1,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 2,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 3,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 4, self.exp.end_img - 2,
                          self.exp.end_img - 1, self.exp.end_img - 3, self.exp.end_img - 4]

        train_images = []
        train_masks = []

        for idx, img_i in enumerate(train_img_ids):
            curr_img = self.all_imgs_list[img_i] / 255.
            curr_img = resize(curr_img, (int(curr_img.shape[0] / 2), int(curr_img.shape[1] / 2)),
                              anti_aliasing=True)
            curr_img = np.expand_dims(curr_img, axis=0)
            train_images.append(curr_img)

            curr_mask = self._yuv_clip_image(img_i)
            curr_mask = dilation(curr_mask, disk(2))
            curr_mask = resize(curr_mask, (int(curr_mask.shape[0] / 2), int(curr_mask.shape[1] / 2)),
                               anti_aliasing=True)
            curr_mask = to_categorical(curr_mask, num_classes=2, dtype='uint8')
            curr_mask = np.expand_dims(curr_mask, axis=0)
            train_masks.append(curr_mask.astype(np.bool))

        panel_images = {}
        panel_masks = {}

        for p in self.panel_list:
            panels_img = []
            panel_mask = []
            for idx, img_i in enumerate(train_img_ids):
                img = self.all_imgs_list[idx]
                panel_img = p.get_cropped_image(img)
                panel_img = np.expand_dims(panel_img, axis=0)
                panels_img.append(panel_img)
                curr_mask = self._yuv_clip_image(img_i)
                curr_mask = dilation(curr_mask, disk(2))
                curr_mask = to_categorical(curr_mask, num_classes=2, dtype='uint8')
                curr_mask = p.get_cropped_image(curr_mask)
                curr_mask = np.expand_dims(curr_mask, axis=0)
                panel_mask.append(curr_mask)
            panel_images[p.label] = panels_img
            panel_masks[p.label] = panel_mask

        X = np.vstack(train_images)
        Y = np.vstack(train_masks)

        print("Training data selected:", X.shape, Y.shape)

        models = {}

        for j in range(1, len(panel_images) + 1):
            panel_images_j = panel_images[j]
            panel_images_j = np.vstack(panel_images_j)
            panel_masks_j = panel_masks[j]
            panel_masks_j = np.vstack(panel_masks_j)
            pshape = panel_images_j.shape[1:]
            model = create_unet(self, pshape, 2)
            callbacks = [
                EarlyStopping(patience=20, verbose=1),
                ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.0000001, verbose=1),
                ModelCheckpoint('model_panel%s.h5' % j, verbose=1, save_best_only=True, save_weights_only=False)
            ]
            model.compile(optimizer=Adam(lr=0.00001), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(panel_images_j, panel_masks_j, batch_size=2, epochs=200, callbacks=callbacks, verbose=2,
                      validation_split=0.2)
            models[j] = load_model('model_panel%s.h5' % j)

        score = models[1].predict(panel_images[1][0])
        preds = (score > 0.5).astype('uint8')
        preds1 = preds[0, :, :, :]
        preds1 = np.argmax(preds1, axis=2)

        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img_masks = []
            img = self.all_imgs_list[idx]
            for p in self.panel_list:
                panel_img = p.get_cropped_image(img)
                panel_img = np.expand_dims(panel_img, axis=0)
                pp_predicted = models[p.label].predict(panel_img)
                pp_predicted = np.argmax((pp_predicted > 0.4).astype('uint8')[0, :, :, :], axis=2)
                pp_predicted.shape = p.mask_crop.shape
                pp_predicted = pp_predicted.astype(np.bool)
                img_masks.append(pp_predicted)
            fig = plt.figure(dpi=300)
            plt.imshow(pp_predicted)
            fig.savefig(pj(self.exp_images_dir, "panels_mask_%s.jpg" % idx))  # MASKS
            plt.close(fig)
            self.all_masks.append(img_masks)

            with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                np.save(fh, img_masks)

        self.app.status_string.set("Removing background %d %%" % int(
            float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))



    def _train_clf(self, clf_in, ensemble_clf_f, X, y):
        # Split X and Y into train/test to get an estimate of classification accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Shape of entire dataset: ", X.shape, y.shape)
        print("Shape of training dataset: ", X_train.shape, y_train.shape)
        print("Shape of testing dataset: ", X_test.shape, y_test.shape)

        # Fit classifier on training data, print train and test accuracy scores
        for clf_n, clf in clf_in:
            clf.fit(X_train, y_train)
            print(clf_n, " train score: ", clf.score(X_train, y_train))
            print(clf_n, " test score: ", clf.score(X_test, y_test))
        # Append trained classifier to list of classifiers
        self.classifiers.append(clf)

        # Save trained classifier
        with open(ensemble_clf_f, "wb") as fh:
            pickle.dump(clf_in, fh)

    # def _sgd_hyperparameters(self, clf_in, ensemble_clf_f, X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     params = {'alpha': [0.0003], 'max_iter': [50000], 'tol': [1e-4], 'random_state': [0],
    #               'learning_rate': ['optimal'], 'early_stopping': [True], 'validation_fraction': [0.2],
    #               'n_iter_no_change': [4], 'average': [True], 'n_jobs': [-1]}
    #     clf = SGDClassifier()
    #     grid = GridSearchCV(clf, param_grid=params, cv=5, refit=True, verbose=2)
    #     print(X.shape)
    #     print(y.shape)
    #     grid.fit(X, y)
    #     print("Best validation score: ", grid.best_score_)
    #     print("Best validation params: ", grid.best_params_)
    #     y_pred = grid.predict(X_test)
    #     print("Score of the optimised classifiers: ", accuracy_score(y_test, y_pred))
    #     grid = grid.best_estimator_
    #     with open(ensemble_clf_f, "wb") as fh:
    #         pickle.dump(grid, fh)

    def _create_transformed_data(self, rgb_pixels):
        # Removed the random selection so to use as much data as possible.
        # Get all of the bg pixels
        rgb_pixels = flatten_img(rgb_pixels)
        rgb_pixels = rgb_pixels[rgb_pixels.all(axis=1), :]
        return rgb_pixels

    def _remove_background(self):
        print("Removing background...")

        # If number of masks in masks directory is greater than number of images to be analysed, skip background removal
        # step as masks already exist
        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img - self.exp.start_img):
            return

        self.all_masks = []

        # Predict the background and foreground pixels in each image
        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img_masks = []
            # Try to load predicted background/foreground mask for this image ID if it exists
            try:
                img_masks = np.load(self.exp_masks_dir_frame % idx, allow_pickle=True)
            except Exception as e:
                img = self.all_imgs_list[idx]
                # For each panel in this image, crop the image to just panel of interest, predict the BG/FG pixels
                # in this panel, apply dilation and erosion to the predicted mask, remove small objects that are
                # unlikely to be seeds. Append this predicted mask to a list of masks for this image.
                for p in self.panel_list:
                    panel_img = p.get_cropped_image(img)
                    pp_predicted = self._ensemble_predict(self.classifiers, flatten_img(panel_img), p)
                    pp_predicted.shape = p.mask_crop.shape
                    pp_predicted = pp_predicted.astype(np.bool)
                    # pp_predicted = dilation(pp_predicted, disk(4))
                    # pp_predicted = erosion(pp_predicted, disk(2))
                    pp_predicted = remove_small_objects(pp_predicted)
                    img_masks.append(pp_predicted)

                # Save list of masks for this image
                with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                    np.save(fh, img_masks)

            # Append list of masks for this image to a list that will contain masks for all images
            self.app.status_string.set("Removing background %d %%" % int(
                float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))
            self.all_masks.append(img_masks)

    def _label_seeds(self):
        l_rprops_f = pj(self.exp_gzdata_dir, "l_rprops_data.pkl")
        if os.path.exists(l_rprops_f):
            with open(l_rprops_f, 'rb') as fh:
                try:
                    self.panel_l_rprops = pickle.load(fh)
                    return
                except EOFError:
                    print("pickle error")

        fig, axarr = plt.subplots(self.exp.panel_n, 1, figsize=(16, 16 * self.exp.panel_n))
        try:
            axarr.shape
        except:
            axarr = [axarr]

        retain = int((self.exp.end_img - self.exp.start_img) / 10.)

        init_masks = self.all_masks

        if len(init_masks) == 0:
            for i in range(self.exp.start_img, self.exp.start_img + retain):
                data = np.load(self.exp_masks_dir_frame % i, allow_pickle=True)
                init_masks.append(data)
        self.panel_l_rprops = []

        for idx, panel in enumerate(self.panel_list):

            # :10 for tomato, :20 for corn/brassica
            mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
            mask_med = clear_border(np.median(mask_med, axis=2)).astype(np.bool)
            mask_med = remove_small_objects(mask_med)

            # Label features in an array using the default structuring element which is a cross.
            labelled_array, num_features = measurements.label(mask_med)
            rprops = regionprops(labelled_array, coordinates='xy')

            all_seed_rprops = []  # type: List[SeedPanel]
            for rp in rprops:
                all_seed_rprops.append(
                    SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
                              rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent, rp.convex_area))

            # Get maximum number of seeds
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            in_mask = find_closest_n_points(pts, self.exp.seeds_n)

            # If we've got less seeds than we should do, should we throw them away?
            if len(in_mask) > self.exp.seeds_n:
                all_seed_rprops_new = []
                for rp, im in zip(all_seed_rprops, in_mask):
                    if im:
                        all_seed_rprops_new.append(rp)
                    else:
                        # Remove false seed rprops from mask, probably need to reorder after
                        labelled_array[labelled_array == rp.label] = 0
                all_seed_rprops = all_seed_rprops_new
            # end if-----------------------------------#

            # Remove extra 'seeds' (QR labels) from boundary
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            xy_range = get_xy_range(labelled_array)
            in_mask = find_pts_in_range(pts, xy_range)

            # If we've got more seeds than we should do, should we throw them away?
            if len(in_mask) > self.exp.seeds_n:
                all_seed_rprops_new = []
                for rp, im in zip(all_seed_rprops, in_mask):
                    if im:
                        all_seed_rprops_new.append(rp)
                    else:
                        # Remove false seed rprops from mask
                        labelled_array[labelled_array == rp.label] = 0
                all_seed_rprops = all_seed_rprops_new
            # end if-----------------------------------#

            # Need to update pts if we have pruned.
            pts = np.vstack([el.centroid for el in all_seed_rprops])

            pts_order = order_pts_lr_tb(pts, self.exp.seeds_n, xy_range, self.exp.seeds_col_n, self.exp.seeds_row_n)

            new_order = []
            new_mask = np.zeros(labelled_array.shape)
            for s_idx, s in enumerate(pts_order):
                sr = all_seed_rprops[s]
                # Reorder mask
                new_mask[labelled_array == sr.label] = s_idx + 1
                sr.label = s_idx + 1
                new_order.append(sr)

            all_seed_rprops = new_order
            labelled_array = new_mask

            # We add an array of labels and the region proprties for each panel.
            self.panel_l_rprops.append((labelled_array, all_seed_rprops))

        minimum_areas = []
        self.end_idx = np.full(len(self.panel_list), self.exp.end_img)

        for idx in tqdm(range(0, len(self.all_masks))):
            self.panel_l_rprops_1 = []
            fig, axarr = plt.subplots(self.exp.panel_n, 1, figsize=(16, 16 * self.exp.panel_n))
            for ipx, panel in enumerate(self.panel_list):
                # :10 for tomato, :20 for corn/brassica
                # mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
                mask_med = init_masks[idx][ipx]
                mask_med = remove_small_objects(mask_med)

                # Label features in an array using the default structuring element which is a cross.
                labelled_array, num_features = measurements.label(mask_med)
                rprops = regionprops(labelled_array, coordinates='xy')

                all_seed_rprops = []  # type: List[SeedPanel]
                for rp in rprops:
                    all_seed_rprops.append(
                        SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
                                  rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent, rp.convex_area))

                if idx == 0:
                    minimum_areas.append(np.zeros((len(all_seed_rprops))))
                    for i in range(len(all_seed_rprops)):
                        minimum_areas[ipx][i] = all_seed_rprops[i].area

                # Get maximum number of seeds
                if all_seed_rprops == []:
                    break
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                in_mask = find_closest_n_points(pts, self.exp.seeds_n)

                # If we've got less seeds than we should do, should we throw them away?
                if len(in_mask) > self.exp.seeds_n:
                    all_seed_rprops_new = []
                    for rp, im in zip(all_seed_rprops, in_mask):
                        if rp.area < 0.6 * np.percentile(minimum_areas[ipx], 10):
                            print("Removed object with area =" + str(rp.area))
                            labelled_array[labelled_array == rp.label] = 0
                            labelled_array[labelled_array > rp.label] -= 1
                        elif im:
                            all_seed_rprops_new.append(rp)
                        else:
                            # Remove false seed rprops from mask
                            labelled_array[labelled_array == rp.label] = 0
                            labelled_array[labelled_array > rp.label] -= 1
                    all_seed_rprops = all_seed_rprops_new

                # end if-----------------------------------#

                # Remove extra 'seeds' (QR labels) from boundary
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                xy_range = get_xy_range(labelled_array)
                in_mask = find_pts_in_range(pts, xy_range)

                # If we've got more seeds than we should do, should we throw them away?
                # if len(in_mask) > self.exp.seeds_n:
                #     all_seed_rprops_new = []
                #     for rp, im in zip(all_seed_rprops, in_mask):
                #         if im:
                #             all_seed_rprops_new.append(rp)
                #         else:
                #             # Remove false seed rprops from mask
                #             labelled_array[labelled_array == rp.label] = 0
                #     all_seed_rprops = all_seed_rprops_new
                # end if-----------------------------------#

                # Need to update pts if we have pruned.
                pts = np.vstack([el.centroid for el in all_seed_rprops])

                pts_order = order_pts_lr_tb(pts, self.exp.seeds_n, xy_range, self.exp.seeds_col_n, self.exp.seeds_row_n)

                new_order = []
                new_mask = np.zeros(labelled_array.shape)
                for s_idx, s in enumerate(pts_order):
                    sr = all_seed_rprops[s]
                    # reorder mask
                    new_mask[labelled_array == sr.label] = s_idx + 1
                    sr.label = s_idx + 1
                    new_order.append(sr)

                all_seed_rprops = new_order
                labelled_array = new_mask

                # We add an array of labels and the region properties for each panel.
                print("Number of seeds identified in panel {}: ".format(panel.label) + str(len(all_seed_rprops)))
                self.panel_l_rprops_1.append((labelled_array, all_seed_rprops))
                if self.exp.panel_n > 1:
                    axarr[ipx].imshow(mask_med)
                    for rp in all_seed_rprops:
                        axarr[ipx].annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r',
                                            fontsize=16)
                        axarr[ipx].annotate(str(panel.label), xy=(20, 10), color='r', fontsize=28)
                        axarr[ipx].axis('off')
                else:
                    axarr.imshow(mask_med)
                    for rp in all_seed_rprops:
                        axarr.annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r',
                                       fontsize=16)
                        axarr.annotate(str(panel.label), xy=(20, 10), color='r', fontsize=28)
                        axarr.axis('off')

                if len(all_seed_rprops) < 0.8 * self.exp.seeds_n and self.end_idx[ipx] == self.exp.end_img:
                    self.end_idx[ipx] = idx
            fig.savefig(pj(self.exp_images_dir, 'seeds_labelled_{}.png'.format(str(idx))))
            plt.close('all')
            self.all_rprops.append(self.panel_l_rprops_1)

    def _generate_statistics(self):
        l_rprops_f = pj(self.exp_gzdata_dir, "l_rprops_data.pkl")
        if os.path.exists(l_rprops_f):
            with open(l_rprops_f, 'rb') as fh:
                self.all_rprops = pickle.load(fh)
        for j in range(len(self.all_rprops)):
            x = self.all_rprops[j]
            n_seeds = 0
            for p in range(len(x)):
                n_seeds += len(x[p][1])
            X_stats = np.zeros((n_seeds, 10))
            counter = 0
            for i in range(len(x)):
                x0 = x[i][1]
                for k in range(len(x0)):
                    X_stats[counter, :] = [i + 1, k + 1, x0[k].area, x0[k].eccentricity, x0[k].extent,
                                           x0[k].major_axis_length, x0[k].minor_axis_length, x0[k].perimeter,
                                           x0[k].solidity, x0[k].convex_area]
                    counter = counter + 1
            self.total_stats.append(X_stats)
        seed_stats = np.zeros((1, 11))
        for i in range(len(self.total_stats)):
            c = self.total_stats[i]
            x = np.concatenate((np.full(shape=(c.shape[0], 1), fill_value=i), c), axis=1)
            seed_stats = np.concatenate((seed_stats, x))
        seed_stats = np.delete(seed_stats, 0, axis=0)
        stats_over_time = pd.DataFrame(seed_stats, columns=['Image Index', 'Panel Number', 'Seed Number', 'Seed Area',
                                                            'Seed Eccentricity', 'Seed Extent',
                                                            'Seed Major Axis Length', 'Seed Minor Axis Length',
                                                            'Seed Perimeter', 'Seed Solidity', 'Seed Convex Area'])
        stats_over_time['Image Index'] = stats_over_time['Image Index'].astype('uint8')
        stats_over_time['Panel Number'] = stats_over_time['Panel Number'].astype('uint8')
        stats_over_time['Seed Number'] = stats_over_time['Seed Number'].astype('uint8')
        direc = pj(self.exp_results_dir, "stats_over_time.csv")
        stats_over_time.to_csv(direc, index=False)
        return

    def _perform_classification(self):
        """ Also need to quantify whether the seed merges, and whether it has 
        moved.
        """
        print("Classifying seeds")

        if len(glob.glob(pj(self.exp_results_dir, "germ_panel_*.json"))) >= self.exp.panel_n:
            print("Already analysed data")
            return

        if self.all_masks is None:
            self.all_masks = []
            for i in range(self.exp.start_img, self.exp.end_img):
                data = np.load(self.exp_masks_dir_frame % (i), allow_pickle=True)
                self.all_masks.append(data)

        # Evaluate each panel separately so that variance between genotypes doesn't worsen results
        for panel_idx, panel_object in enumerate(tqdm(self.panel_list)):
            try:
                # This is the tuple of the labelled arrays generated from regionprops
                panel_labels, panel_regionprops = self.panel_l_rprops[panel_idx]
                p_masks = []
                # Extract all the masks for the specific panel, for every image
                for i in range(len(self.all_masks)):
                    p_masks.append(self.all_masks[i][panel_idx])

                self.spp_processor.use_colour = self.exp.use_colour
                self.spp_processor.use_delta = self.exp.use_delta

                panel_germ = self.spp_processor._classify(
                    panel_object,
                    self.all_imgs[self.exp.start_img:self.exp.end_img],
                    panel_labels,
                    panel_regionprops,
                    p_masks,
                    self.all_imgs_list[self.exp.start_img:self.exp.end_img]
                )

                out_f = pj(
                    self.exp_results_dir,
                    "germ_panel_%d.json" % panel_idx
                )

                with open(out_f, "w") as fh:
                    json.dump(panel_germ, fh)


            except Exception as e:
                print("Could not run panel %d" % (panel_idx))
                print(e)
                traceback.print_exc()

    def _get_cumulative_germ(self, germ, win=5):
        for m in range(germ.shape[0]):
            curr_seed = germ[m, :]
            idx = 0
            while idx < (curr_seed.shape[0] - win):
                if curr_seed[idx:idx + win].all():
                    curr_seed[:idx + win - 1] = 0
                    curr_seed[idx + win - 1:] = 1
                    break
                idx += 1
            if idx >= (curr_seed.shape[0] - win):
                curr_seed[:] = 0
        return germ.sum(axis=0), germ

    def _analyse_results(self, proprtions):
        all_germ = []
        for i in range(self.exp.panel_n):
            # If the with fails to open, we should not perform operations on germ_d.
            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (i))) as fh:
                germ_d = json.load(fh)

                # Ensure the germ_d isn't empty.
                if len(germ_d) == 0:
                    continue

                germinated = []
                for j in range(1, len(germ_d.keys()) + 1):
                    germinated.append(germ_d[str(j)])

                germinated = np.vstack(germinated)
                all_germ.append(germinated)

        p_totals = []
        for i in range(self.exp.panel_n):
            # l, rprop = self.panel_l_rprops[i]
            l, rprop = self.all_rprops[0][i]
            p_totals.append(len(rprop))

        if len(all_germ) == 0:
            raise Exception("Germinated seeds found is 0. Try changing YUV values.")

        print(p_totals)

        cum_germ_data = []

        np.save(self.exp_results_dir + '/all_germ.npy', all_germ)

        for germ in all_germ:
            cum_germ_data.append(self._get_cumulative_germ(germ, win=4)[0])

        initial_germ_time_data = []

        for germ in all_germ:
            rows, cols = germ.shape
            init_germ_time = []
            for m in range(rows):
                for n in range(cols):
                    if germ[m, n]:
                        # For adding offset
                        init_germ_time.append(n + self.exp.start_img)
                        break  # Probably this is what causes problem
            initial_germ_time_data.append(init_germ_time)

        for i in range(len(cum_germ_data)):
            cum_germ_data[i] = pd.Series(cum_germ_data[i])

        cum_germ_data = pd.concat(cum_germ_data, axis=1).astype('f')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     figsize=(18., 15.),
                                                     dpi=650)

        fig.suptitle(self.exp.name)

        p_t_text = ""
        for i in range(self.exp.panel_n):
            p_t_text += "panel %d: %d" % (i + 1, p_totals[i])
            if ((i + 1) % 2) == 0:
                p_t_text += "\n"
            else:
                p_t_text += "    "

        plt.figtext(0.05, 0.93, p_t_text)

        # Only use date information if it is contained in the filename
        has_date = check_files_have_date(self.imgs[0])
        start_dt = None
        if has_date:
            start_dt = s_to_datetime(self.imgs[0])

        n_frames = cum_germ_data.shape[0]

        for idx in range(cum_germ_data.shape[1]):
            ax1.plot(
                range(self.exp.start_img, n_frames+self.exp.start_img),
                cum_germ_data.iloc[:, idx] / float(p_totals[idx]), label="Genotype" + str(idx + 1)
            )
        ax1.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])

        if has_date:
            # Sort out xtick labels in hours
            xtick_labels = []
            for val in ax1.get_xticks():
                if int(val) >= (self.exp.end_img - self.exp.start_img):
                    break
                curr_dt = s_to_datetime(self.imgs[int(val)])
                xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))

            ax1.set_xlabel("Time (hours)")
            ax1.set_xticklabels(xtick_labels, )
        else:
            ax1.set_xlabel("Image ID")

        ax1.legend(loc="upper left")
        ax1.set_ylabel("Cumulative germinated percent")
        ax1.set_title("Cumulative germination as percent")
        ax1.grid()

        data = []
        for idx in range(cum_germ_data.shape[1]):
            cum_germ = cum_germ_data.iloc[:, idx].copy().ravel()
            cum_germ /= p_totals[idx]

            germ_pro_total = cum_germ[-1]
            prop_idxs = []
            for pro in proprtions:
                if (cum_germ > pro).any():
                    pos_idx = np.argmax(cum_germ >= pro) + self.exp.start_img

                    if has_date:
                        curr_dt = s_to_datetime(
                            self.imgs[self.exp.start_img + pos_idx])
                        pos_idx = hours_between(start_dt, curr_dt)
                else:
                    pos_idx = 'n/a'
                prop_idxs.append(str(pos_idx))
            data.append(prop_idxs)

        columns = tuple('%d%%' % (100 * prop) for prop in proprtions)
        rows = ['  %d  ' % (x) for x in range(1, self.exp.panel_n + 1)]
        the_table = ax2.table(cellText=data,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='center')

        tbl_props = the_table.properties()
        tbl_cells = tbl_props['child_artists']
        for cell in tbl_cells:
            cell.set_height(0.1)
        ax2.set_title("Percentage T values")
        ax2.axis('off')

        # Old code for setting an end point when roots overlap
        # for i in range(len(initial_germ_time_data)):
        #     for j in range(len(initial_germ_time_data[i]) - 1, -1, -1):
        #         if initial_germ_time_data[i][j] > self.end_idx[i]:
        #             initial_germ_time_data[i].pop(j)
        ax3.boxplot(initial_germ_time_data, vert=False
                    # , whis='range'
                    )
        ax3.set_xlim([self.exp.start_img, self.exp.start_img + n_frames + 5])
        ax3.set_ylabel("Panel number")
        ax3.set_title('Germination time box plot')
        ax3.grid()

        if has_date:
            ax3.set_xlabel("Time (hours)")
            xtick_labels = []
            for val in ax3.get_xticks():
                if int(val) >= len(self.imgs):
                    break
                curr_dt = s_to_datetime(self.imgs[int(val)])
                xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))
            ax3.set_xticklabels(xtick_labels)
        else:
            ax3.set_xlabel("Image ID")

        print(cum_germ_data.iloc[-1, :] / np.array(p_totals))

        # ax4.barh(np.arange(self.exp.panel_n) + 0.75, np.flipud((cum_germ_data.max(axis=0) / np.array(p_totals)).values.reshape(-1, 1)).ravel(), height=0.5)
        ax4.barh(np.arange(self.exp.panel_n) + 0.75, (cum_germ_data.max(axis=0) / np.array(p_totals)), height=0.5)
        ax4.set_yticks(range(1, 1 + self.exp.panel_n))
        ax4.set_ylim([0.5, self.exp.panel_n + .5])
        ax4.set_xlim([0., 1.])
        ax4.set_ylabel("Panel number")
        ax4.set_xlabel("Germinated proportion")
        ax4.set_title("Proportion germinated")

        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax4.set_ylim(ax4.get_ylim()[::-1])
        fig.savefig(pj(self.exp_results_dir, "results.jpg"))
        plt.close(fig)

        img_index = np.arange(n_frames) + self.exp.start_img

        if has_date:
            times_index = []
            for _i in img_index:
                curr_dt = s_to_datetime(self.imgs[_i])
                times_index.append(hours_between(start_dt, curr_dt))

            times_index = np.array(times_index).reshape(-1, 1)
            cum_germ_data = np.hstack([times_index, cum_germ_data.values])

        df = pd.DataFrame(data=cum_germ_data)
        df.index = img_index

        if has_date:
            df.columns = ["Time"] + [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', 1:] = p_totals
        else:
            df.columns = [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', :] = p_totals

        df.to_csv(pj(
            self.exp_results_dir,
            "panel_germinated_cumulative.csv"
        ))

    def _quantify_first_frame(self, proprtions):
        """ Quantify the seed data from the first frame. 
        To quantify:
            - total seed number
            - seeds analysed
            - initial seed size
            - initial seed roundness
            - width/height ratio
            - RGB mean
            - germ rate at various percents
            - seed x, y
        """

        # Only use date information if it is contained in the filename
        has_date = check_files_have_date(self.imgs[0])
        start_dt = None
        if has_date:
            start_dt = s_to_datetime(self.imgs[0])

        img_f = self.all_imgs_list[self.exp.start_img]
        f_masks = np.load(self.exp_masks_dir_frame % (self.exp.start_img), allow_pickle=True)

        img_l = self.all_imgs_list[self.exp.end_img-1]
        l_masks = np.load(self.exp_masks_dir_frame % ((self.exp.end_img - self.exp.start_img) - 1), allow_pickle=True)

        all_panel_data = []

        # Panel analysis
        for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):

            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
                germ_d = json.load(fh)

            germinated = []
            for j in range(1, len(germ_d.keys()) + 1):
                germinated.append(germ_d[str(j)])
            germinated = np.vstack(germinated)

            cum_germ = self._get_cumulative_germ(germinated, win=7)[0].astype('f')
            cum_germ /= len(p_rprops)

            germ_pro_total = cum_germ[-1]

            prop_idxs = []
            for pro in proprtions:
                if (cum_germ > pro).any():
                    pos_idx = np.argmax(cum_germ >= pro) + self.exp.start_img
                    if has_date:
                        curr_dt = s_to_datetime(self.imgs[pos_idx])
                        pos_idx = hours_between(start_dt, curr_dt)
                else:
                    pos_idx = 'n/a'
                prop_idxs.append(pos_idx)

            p_f_img = self.panel_list[p_idx].get_bbox_image(img_f)
            p_f_mask = f_masks[p_idx]

            p_l_img = self.panel_list[p_idx].get_bbox_image(img_l)
            p_l_mask = l_masks[p_idx]

            f_rgb_mu = p_f_img[p_f_mask].mean(axis=0)
            l_rgb_mu = p_l_img[p_l_mask].mean(axis=0)
            f_rgb_mu = tuple(np.round(f_rgb_mu).astype('i'))
            l_rgb_mu = tuple(np.round(l_rgb_mu).astype('i'))

            avg_feas = []
            for rp in p_rprops:
                min_row, min_col, max_row, max_col = rp.bbox
                w = float(max_col - min_col)
                h = float(max_row - min_row)
                whr = w / h
                avg_feas.append([w, h, whr, rp.area, rp.eccentricity])

            avg_feas = np.vstack(avg_feas)
            avg_feas_mu = avg_feas.mean(axis=0)

            panel_data = [p_idx + 1, len(p_rprops)]
            panel_data += np.round(avg_feas_mu, 2).tolist()
            panel_data += [f_rgb_mu, l_rgb_mu]
            panel_data += prop_idxs
            panel_data += [round(germ_pro_total, 2)]

            all_panel_data.append(panel_data)

        columns = [
            'panel_ID',
            'total_seeds',
            'avg_width',
            'avg_height',
            'avg_wh_ratio',
            'avg_area',
            'avg_eccentricity',
            'avg_initial_rgb',
            'avg_final_rgb',
        ]
        columns.extend(['germ_%d%%' % (100 * prop) for prop in proprtions])
        columns.append('total_germ_%')

        df = pd.DataFrame(all_panel_data, columns=columns)
        df.to_csv(pj(self.exp_results_dir, "overall_results.csv"), index=False)

        # Seed analysis
        all_seed_results = []

        panel_seed_idxs = {}

        for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):

            panel_seed_idxs[int(p_idx)] = []

            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
                germ_d = json.load(fh)

            germinated = []
            for j in range(1, len(germ_d.keys()) + 1):
                germinated.append(germ_d[str(j)])
            germinated = np.vstack(germinated)

            cum_germ, germ_proc = self._get_cumulative_germ(germinated, win=7)

            for seed_rp in p_rprops:

                germ_row = germ_proc[seed_rp.label - 1]

                germ_idx = 'n/a'
                if germ_row.any():
                    germ_idx = np.argmax(germ_row) + self.exp.start_img

                min_row, min_col, max_row, max_col = seed_rp.bbox
                w = float(max_col - min_col)
                h = float(max_row - min_row)
                whr = w / h

                if germ_idx == 'n/a':
                    germ_time = 'n/a'
                else:
                    if has_date:
                        curr_dt = s_to_datetime(self.imgs[germ_idx])
                        germ_time = hours_between(start_dt, curr_dt)
                    else:
                        germ_time = germ_idx

                seed_result = [
                    p_idx + 1,
                    seed_rp.label,
                    int(w),
                    int(h),
                    round(whr, 2),
                    int(seed_rp.area),
                    round(seed_rp.eccentricity, 2),
                    # (0,0,0),
                    # (0,0,0),
                    germ_idx,
                    germ_time,
                ]

                if germ_idx == 'n/a':
                    germ_idx = -1

                panel_seed_idxs[int(p_idx)].append((
                    int(germ_idx),
                    int(seed_rp.centroid[0]),
                    int(seed_rp.centroid[1])
                ))
                all_seed_results.append(seed_result)

        columns = [
            'panel_ID',
            'seed_ID',
            'width',
            'height',
            'wh_ratio',
            'area',
            'eccentricity',
            # 'initial_rgb',
            # 'final_rgb',
            'germ_point',
            'germ_time' if has_date else 'germ_image_number',
        ]

        df = pd.DataFrame(all_seed_results, columns=columns)
        df.to_csv(pj(self.exp_results_dir, "panel_results.csv"), index=False)

        with open(pj(self.exp_results_dir, "panel_seed_idxs.json"), "w") as fh:
            json.dump(panel_seed_idxs, fh)

    def run(self):
        print("Processor started")

        if self.running:

            start = time.time()

            try:
                self._save_init_image(self.imgs[self.exp.start_img])

                self._extract_panels(self.imgs[self.exp.start_img], self.core.chunk_no, self.core.chunk_reverse, self.exp.start_img)

                self.app.status_string.set("Saving contour image")

                self._save_contour_image()

                self.app.status_string.set("Training background removal clfs")

                # Seed segmentation performed by the classifier defined in experiment setup
                if self.exp.bg_remover == 'UNet':
                    self.app.status_string.set("Removing background")
                    self._train_unet()
                elif self.exp.bg_remover == "SGD":
                    # Define stochastic gradient descent classifier's hyperparameters
                    self._train_clfs([("SGD", SGDClassifier(max_iter=50, random_state=0, tol=1e-5))])
                    self.app.status_string.set("Removing background")
                    self._remove_background()
                elif self.exp.bg_remover == "GMM":
                    self._train_gmm_clfs()
                    self.app.status_string.set("Removing background")
                    self._gmm_remove_background()
                else:
                    print(".... unknown BG classifier")

                self.app.status_string.set("Labelling seeds")
                self._label_seeds()

                self.app.status_string.set("Generating statistics")
                self._generate_statistics()

                self.app.status_string.set("Performing classification")
                self._perform_classification()

                self.app.status_string.set("Analysing results")

                self._analyse_results(self.core.proportions)

                self.app.status_string.set("Quantifying initial seed data")
                self._quantify_first_frame(self.core.proportions)

                self.app.status_string.set("Finished processing")
                self.exp.status = "Finished"

                print("End values: ", self.end_idx + self.exp.start_img)

                print(time.time() - start)

                print("Finished")

            except Exception as e:
                raise e
                self.exp.status = "Error"
                self.app.status_string.set("Error whilst processing")
                print("Exception args: " + str(e.args))

            self.running = False

            self.core.stop_processor(self.exp.eid)

    def die(self):
        self.running = False
