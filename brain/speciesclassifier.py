# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:11:47 2016

@author: dty09rcu
"""

import os

import numpy as np
from imageio import imread
from skimage.measure import regionprops
from skimage.morphology import dilation
from skimage.morphology import disk
from sklearn import svm
import time

from helper.functions import delta
from helper.functions import simple_label_next_frame

# from pomegranate import *

pj = os.path.join


class SpeciesClassifier:
    def __init__(self, seed="default", spp_mask_dilate=3, seed_pix_n=50, germ_true_width=5, clf_contamination=0.1,
                 area_growth_min=1.25, use_colour=False):
        self.seed = seed
        self.spp_mask_dilate = spp_mask_dilate
        self.seed_pix_n = seed_pix_n
        self.germ_true_width = germ_true_width
        self.clf_contamination = clf_contamination
        self.area_growth_min = area_growth_min
        self.use_colour = use_colour

    @property
    def extra(self):
        return np.array([
            -self.seed_pix_n,
            -self.seed_pix_n,
            self.seed_pix_n,
            self.seed_pix_n
        ])

    def _build_classifiers(self):
        # Train on first 20% of images and create empty lists of features
        to_analyse = int(len(self.all_imgs) * 0.2)
        hu_feas = []
        areas = []
        lengths = []
        initial_areas = []
        initial_lengths = []
        initial_hu_feas = []

        # Create empty lists of colour features if using colour
        if self.use_colour:
            colors_r = []
            colors_g = []
            colors_b = []

        # For all images and masks
        for index, (mask, img_f) in enumerate(list(zip(self.panel_masks[:], self.all_imgs[:]))):
            # Read respective image and create empty lists of features for this image
            img = imread(img_f)
            hu_feas_labelled = []
            areas_labelled = []
            lengths_labelled = []
            # Crop image to just the panel, get region properties of objects in the panel
            img = self.panel.get_bbox_image(img)
            c_label, c_rprops = simple_label_next_frame(self.panel_labels, self.panel_regionprops, mask)

            # For each seed found in the panel
            for idx, rp in enumerate(c_rprops):
                # If colour is being used, append the respective colour channel lists with rgb values f from
                # the colour histogram function
                if self.use_colour:
                    r, g, b = self.generate_color_histogram(img, rp)
                    colors_r.append(r)
                    colors_g.append(g)
                    colors_b.append(b)

                # Append the features of the seed (Hu moments, area, major axis length, minor axis length, minor/major
                # axis length ratio) to a list of seed features
                hu_feas.append(rp.moments_hu)
                hu_feas_labelled.append(np.hstack((rp.moments_hu, rp.label)))
                areas.append(rp.area)
                areas_labelled.append([rp.area, rp.label])
                lengths.append([rp.minor_axis_length, rp.major_axis_length,
                                float(rp.minor_axis_length + 1.0) / float(rp.major_axis_length + 1.0)])
                lengths_labelled.append(np.hstack(([rp.minor_axis_length, rp.major_axis_length,
                                            float(rp.minor_axis_length + 1.0) / float(rp.major_axis_length + 1.0)],
                                           rp.label)))
            # Append the list of seed features for that image to a list of all images' seed features
            initial_areas.append(np.array(areas_labelled))
            initial_lengths.append(np.array(lengths_labelled))
            initial_hu_feas.append(np.array(hu_feas_labelled))

        areas = np.vstack(areas)
        hu_feas = np.vstack(hu_feas)
        lengths = np.vstack(lengths)
        self.delta_area = np.zeros((areas.shape[0], 1))
        self.delta_hu_feas = np.zeros((hu_feas.shape[0], 7))
        self.delta_lengths = np.zeros((lengths.shape[0], 3))
        counter = 0
        # For i in total number of images
        for i in range(len(initial_areas)):
            # For j in largest seed label
            for j in range(np.max(initial_areas[i][:, 1])):
                # If first image
                if i == 0:
                    # If seed label is present in current image array
                    if np.isin(j + 1, initial_areas[i][:, 1]):
                        id = j + 1
                        if np.isin(id, initial_areas[i + 1][:, 1]):
                            curr_arr = initial_areas[i][:, 1]
                            curr = np.argwhere(curr_arr == id)
                            next_arr = initial_areas[i + 1][:, 1]
                            next = np.argwhere(next_arr == id)
                            # As the delta for the first image is undefined, set it to the difference between the first
                            # and second image
                            self.delta_area[counter, 0] = np.abs(initial_areas[i + 1][next, 0] - initial_areas[i][curr, 0])
                            self.delta_lengths[counter, :] = np.abs(
                                initial_lengths[i + 1][next, :3] - initial_lengths[i][curr, :3])
                            self.delta_hu_feas[counter, :] = np.abs(
                                initial_hu_feas[i + 1][next, :7] - initial_hu_feas[i][curr, :7])
                            counter += 1
                else:
                    # If seed label is present in current image array
                    if np.isin(j + 1, initial_areas[i][:, 1]):
                        id = j + 1
                        # Get indices of same seed in previous and current image array
                        curr_arr = initial_areas[i][:, 1]
                        curr = np.argwhere(curr_arr == id)
                        prev_arr = initial_areas[i - 1][:, 1]
                        prev = np.argwhere(prev_arr == id)
                        if curr.size != prev.size:
                            # If seed disappears or new seed, set it's delta to the mean of other seeds
                            self.delta_area[counter, 0] = np.mean(self.delta_area[0:counter, 0])
                            self.delta_lengths[counter, :] = np.mean(self.delta_lengths[0:counter, :])
                            self.delta_hu_feas[counter, :] = np.mean(self.delta_hu_feas[0:counter, :])
                            counter += 1
                        else:
                            # Create delta features i.e. seed feature from this image - seed feature from previous image
                            self.delta_area[counter, 0] = np.abs(
                                initial_areas[i][curr, 0] - initial_areas[i - 1][prev, 0])
                            self.delta_lengths[counter, :] = np.abs(
                                initial_lengths[i][curr, :3] - initial_lengths[i - 1][prev, :3])
                            self.delta_hu_feas[counter, :] = np.abs(
                                initial_hu_feas[i][curr, :7] - initial_hu_feas[i - 1][prev, :7])
                            counter += 1

        # Get the number of seeds to train on
        to_analyse = np.sum(item.shape[0] for item in initial_areas[:to_analyse])
        # Create array containing seed features from all images
        self.all_data = np.hstack([hu_feas, self.delta_hu_feas, areas, self.delta_area, lengths, self.delta_lengths])
        # Create training data for one class SVM
        hu_feas = np.hstack([hu_feas[:to_analyse, :], self.delta_hu_feas[:to_analyse, :], areas[:to_analyse, :], self.delta_area[:to_analyse, :], lengths[:to_analyse, :], self.delta_lengths[:to_analyse, :]]) #added in area and delta area.
        if self.use_colour:
            color_feas = np.hstack([np.vstack(colors_r), np.vstack(colors_g), np.vstack(colors_b)])

        # Normalise the hu features and the delta mean i.e. z = (x-mu)/sigma
        self.hu_feas_mu = hu_feas.mean(axis=0)
        self.hu_feas_stds = hu_feas.std(axis=0)
        hu_feas = (hu_feas - self.hu_feas_mu) / (self.hu_feas_stds + 1e-9)

        # Train a one class SVM on the hu features
        self.clf_hu = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.001, random_state=0)
        self.clf_hu.fit(hu_feas)

        # If using colour, normalise the colour histograms i.e. z = (x-mu)/sigma
        if self.use_colour:
            self.color_feas_mu = color_feas.mean(axis=0)
            self.color_feas_stds = color_feas.std(axis=0)
            color_feas = (color_feas - self.color_feas_mu) / (self.color_feas_stds + 1e-9)
            # Train a one class SVM on the colour features
            self.clf_color = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.001, random_state=0)
            self.clf_color.fit(color_feas)

    def HMM_TEST_build_classifier(self):
        hu_feas = []
        areas = []
        lengths = []
        all_areas = None
        for index, (mask, img_f) in enumerate(zip(self.panel_masks, self.all_imgs)):
            c_label, c_rprops = simple_label_next_frame(self.panel_labels, self.panel_regionprops, mask)
            for idx, rp in enumerate(c_rprops):
                hu_feas.append(rp.moments_hu)
                areas.append(rp.area)
                lengths.append([rp.minor_axis_length, rp.major_axis_length,
                                float(rp.minor_axis_length + 1.0) / float(rp.major_axis_length + 1.0)])

            areas = np.array(areas)
            print(areas.shape)
            if all_areas is None:
                all_areas = np.vstack(areas)
            else:
                all_areas = np.vstack((all_areas, areas))
            print(all_areas)

        print(all_areas)
        # areas = np.vstack(areas)
        # hu_feas = np.vstack(hu_feas)
        # hu_feas = np.hstack([hu_feas, delta(hu_feas), areas, delta(areas), lengths, delta(lengths)]) #added in area and delta area.

        # normalise the hu features and the delta mean.
        # self.hu_feas_mu = hu_feas.mean(axis=0)
        # self.hu_feas_stds = hu_feas.std(axis=0)
        # hu_feas = (hu_feas - self.hu_feas_mu) / self.hu_feas_stds

        print(areas.shape)

        # z-norm series.
        areas_norm = []
        for series in areas:
            series = (series - np.mean(series, axis=0)) / np.std(areas, axis=0)
            areas_norm.append(series)

        areas = np.array(areas_norm)
        print(areas)
        model = self._create_hmm(areas)

        for area in areas:
            print(model.predict(area))

    def generate_color_histogram(self, img, region_prop):
        (min_row, min_col, max_row, max_col) = region_prop.bbox
        # imwrite('pictures/outfile_%i_%i.jpg' % (idx, index), img[min_row:max_row, min_col:max_col])
        pixels = np.array(img[min_row:max_row, min_col:max_col])
        r, g, b = np.dsplit(pixels, pixels.shape[-1])
        hist_r, _ = np.histogram(r)
        hist_g, _ = np.histogram(g)
        hist_b, _ = np.histogram(b)
        return hist_r, hist_g, hist_b

    def _get_seed_mask_set(self, seed_rp):
        min_row, min_col, max_row, max_col = seed_rp.bbox + self.extra

        row_max, col_max = self.panel_masks[0].shape

        if min_row < 0:
            min_row = 0
        if min_col < 0:
            min_col = 0
        if max_row > row_max:
            max_row = row_max
        if max_col > col_max:
            max_col = col_max

        seed_masks = [el[min_row:max_row, min_col:max_col] for el in self.panel_masks[:len(self.all_imgs)]]

        init_mask = (self.panel_labels == seed_rp.label)[min_row:max_row, min_col:max_col]
        sm_extracted = [np.logical_and(seed_masks[0], init_mask)]

        # for prev_mask, curr_mask in zip(sm_extracted[-1], seed_masks[1:])
        for i in range(1, len(seed_masks)):
            prev_mask = sm_extracted[i - 1]
            curr_mask = seed_masks[i]

            prev_mask_dilated = dilation(prev_mask, disk(self.spp_mask_dilate))
            new_mask = np.logical_and(prev_mask_dilated, curr_mask)
            sm_extracted.append(new_mask)

        return sm_extracted, (min_row, min_col, max_row, max_col)

    def _classify(self, panel, all_imgs, p_labels, p_rprops, p_masks, all_imgs_list):
        self.panel = panel
        self.all_imgs = all_imgs
        self.panel_labels = p_labels
        self.panel_regionprops = p_rprops
        self.panel_masks = p_masks
        self._build_classifiers()
        self.all_imgs_list = all_imgs_list

        seed_classification = {}
        cols = []

        # For all the found seeds of the input panel.
        for index, rp in enumerate(self.panel_regionprops):
            cols.append(rp.label)
            seed_masks, _ = self._get_seed_mask_set(rp)

            list_error = False
            areas = []
            hu_feas = []
            lengths = []
            areas_total = []
            hu_feas_total = []
            lengths_total = []
            if self.use_colour:
                colors_r = []
                colors_g = []
                colors_b = []

            for idx, m in enumerate(seed_masks):
                m = m.astype('i')

                m_rp = regionprops(m, coordinates="xy")
                if not m_rp:
                    list_error = True
                    print("Empty regionprops list in classifier")
                    break

                m_rp = m_rp[0]
                areas_total.append(m_rp.area)
                hu_feas_total.append(m_rp.moments_hu)
                lengths_total.append([m_rp.minor_axis_length, m_rp.major_axis_length,
                                            float(m_rp.minor_axis_length + 1.0) / float(m_rp.major_axis_length + 1.0)])

            # For each seed in all the images.
            for idx, m in enumerate(seed_masks):
                # Extract image region.
                m = m.astype('i')

                m_rp = regionprops(m, coordinates="xy")
                if not m_rp:
                    list_error = True
                    print("Empty regionprops list in classifier")
                    break

                m_rp = m_rp[0]

                if self.use_colour:
                    img = all_imgs_list[idx]
                    img = self.panel.get_bbox_image(img)

                    r, g, b = self.generate_color_histogram(img, m_rp)
                    colors_r.append(r)
                    colors_g.append(g)
                    colors_b.append(b)

                hu_feas.append(m_rp.moments_hu)
                areas.append(m_rp.area)
                # Apply Laplacian correction on the axis ratio to ensure it's not zero
                lengths.append([m_rp.minor_axis_length, m_rp.major_axis_length, float(m_rp.minor_axis_length+1.0)/float(m_rp.major_axis_length+1.0)])

            if list_error:
                seed_classification[rp.label] = [0] * len(seed_masks)
                print("Error with current seed,", rp.label)
                continue

            # trying to use the cummean of the areas. ie its rate of growth over time.
            hu_feas = np.vstack(hu_feas)

            # TEST:!!!! see if we can detect the change point.
            '''from helper.change_point_analysis import CPA_PoissonMean, CPA_Mean, CPA_Variance, CPA_BernoulliMean

            cpas = [CPA_Mean(),CPA_Variance(),CPA_BernoulliMean(),CPA_PoissonMean()]
            print(areas)
            for cpa in cpas:
                print(type(cpa).__name__)
                print(cpa.find_change_point(areas))'''


            areas = np.vstack(areas)
            hu_feas = np.vstack(hu_feas)
            lengths = np.vstack(lengths)
            delta_areas = []
            delta_hu_feas = []
            delta_lengths = []

            for j in range(areas.shape[0]):
                if j == 0:
                    delta_areas.append(areas[j+1]-areas[j])
                    delta_hu_feas.append(hu_feas[j+1]-hu_feas[j])
                    delta_lengths.append(lengths[j+1]-lengths[j])
                else:
                    delta_areas.append(areas[j]-areas[j-1])
                    delta_hu_feas.append(hu_feas[j]-hu_feas[j-1])
                    delta_lengths.append(lengths[j]-lengths[j-1])

            delta_areas = np.vstack(delta_areas)
            delta_lengths = np.vstack(delta_lengths)
            delta_hu_feas = np.vstack(delta_hu_feas)

            hu_feas = np.hstack(
                [hu_feas, delta_hu_feas, areas, delta_areas, lengths, delta_lengths])  # added in area and delta area.
            hu_feas = (hu_feas - self.hu_feas_mu) / (self.hu_feas_stds + 1e-9)

            # hu_preds = self.clf_hu.score_samples(hu_feas) #GMM
            hu_preds = self.clf_hu.predict(hu_feas)

            if self.use_colour:
                color_feas = np.hstack([np.vstack(colors_r), np.vstack(colors_g), np.vstack(colors_b)])
                color_feas = (color_feas - self.color_feas_mu) / self.color_feas_stds
                # color_preds = self.clf_color.score_samples(color_feas) #GMM
                # average_pred = ((color_preds + hu_preds) / 2) < self.combined_prob_low #GMM

                color_preds = self.clf_color.predict(color_feas)
                average_pred = ((color_preds + hu_preds) / 2) <= 0

            else:
                # average_pred = hu_preds < self.combined_prob_low #GMM
                average_pred = hu_preds <= 0  # convert -1 or 1 to boolean

            areas = np.array(areas)

            to_analyse = int(len(self.panel_masks) * 0.2)
            area_mu = areas[:to_analyse].mean()

            if not (areas > (area_mu * self.area_growth_min)).sum():
                seed_classification[rp.label] = [0] * len(seed_masks)
                continue

            img_ten_percent = int(len(seed_masks) * 0.1)
            img_start = np.argmax(areas > (area_mu * self.area_growth_min)) - img_ten_percent
            if img_start < 0:
                img_start = 0

            seed_area_mask = np.array([0] * len(seed_masks))
            seed_area_mask[img_start:] = 1

            all_preds = np.logical_and(average_pred, seed_area_mask)

            seed_classification[rp.label] = all_preds.astype('i').tolist()
        return seed_classification
