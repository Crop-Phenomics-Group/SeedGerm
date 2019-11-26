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
from keras.models import load_model
# Imaging/vision imports.
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imageio import imread
from tqdm import tqdm
import keras
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.utils import to_categorical
from skimage.transform import resize
import matplotlib

matplotlib.use('TkAgg')


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)





import keras
from keras import initializers
from keras import layers
from keras.utils.anchors import AnchorParameters
from keras import assert_training_model


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default regression submodel.
    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.
    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.
    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.
    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.
    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.
    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for object detection.
    The default submodels contains a regression submodel and a classification submodel.
    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.
    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.
    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.
    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.
    Args
        models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.
    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.
    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.
    Returns
        A tensor containing the anchors for the FPN features.
        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = None,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.
    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.
    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.
    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.
    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.
    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.
        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
















def create_unet(self, img_shape, num_class):
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


def create_object_detector():
    return


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

        # Read image file names for the experiments, sort based on image 
        # number.
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
                                                   self.exp.species.lower()])  # copy the species classifier  # type: SpeciesClassifer
        except KeyError:
            print("No species module found for %s" % (self.exp.species))
            print("ought to use default, shouldn't occur as populate species list from these modules")
            print("consider adding parameters to the config if you're confident")

    def _run_check(self):
        if not self.running:
            pass
            # do some clean up and stufffff

    def _save_init_image(self, img):
        out_f = pj(self.exp_images_dir, "init_img.jpg")
        if os.path.exists(out_f):
            return
        img_01 = imread(pj(self.exp.img_path, img)) / 255.
        fig = plt.figure()
        plt.imshow(img_01)
        fig.savefig(out_f)
        plt.close(fig)

    def _yuv_clip_image(self, img_f):
        img = imread(os.path.join(self.exp.img_path, img_f))
        img_yuv = rgb2ycrcb(img)
        mask_img = in_range(img_yuv, self.yuv_low, self.yuv_high)
        return mask_img.astype(np.bool)

    def _extract_panels(self, img, chunk_no, chunk_reverse):
        panel_data_f = os.path.join(self.exp_gzdata_dir, "panel_data.pkl")
        if os.path.exists(panel_data_f):
            with open(panel_data_f, 'rb') as fh:
                try:
                    self.panel_list = pickle.load(fh)
                    # print "CONTINUING FOR TESTING..."
                    return
                except EOFError:
                    print("pickle is broken")

        mask_img = self._yuv_clip_image(img)
        mask_img = remove_small_objects(
            fill_border(mask_img, 10, fillval=False),
            min_size=1024
        )
        mask_img_cleaned_copy = mask_img.copy()
        mask_img = erosion(binary_fill_holes(mask_img), disk(7))

        obj_only_mask = np.logical_and(mask_img, np.logical_not(mask_img_cleaned_copy))

        # get labels
        l, n = measurements.label(mask_img)

        rprops = regionprops(l, coordinates='xy')

        def get_mask_objects(idx, rp):
            tmp_mask = np.zeros(mask_img.shape)
            tmp_mask[l == rp.label] = 1

            both_mask = np.logical_and(obj_only_mask, tmp_mask)
            both_mask = remove_small_objects(both_mask)
            # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            # ax[0].imshow(tmp_mask)
            # ax[1].imshow(both_mask)
            # fig.savefig(pj(self.exp_images_dir, "mask_img_{}.jpg".format(idx)))
            # plt.close(fig)
            _, panel_object_count = measurements.label(both_mask)  # , return_num=True)
            return panel_object_count

        rprops = [(rp, get_mask_objects(idx, rp)) for idx, rp in enumerate(rprops)]
        rprops = sorted(rprops, key=itemgetter(1), reverse=True)

        # Check the panel has seeds in it

        panels = [(rp, rp.centroid[0], rp.centroid[1]) for rp, _ in rprops[:self.exp.panel_n]]

        # sort panels based on y first, then x
        panels = sorted(panels, key=itemgetter(1))
        panels = chunks(panels, chunk_no)
        panels = [sorted(p, key=itemgetter(2), reverse=chunk_reverse) for p in panels]
        print(panels)
        panels = list(chain(*panels))

        # set mask, where 1 is top left, 2 is top right, 3 is middle left, etc
        panel_list = []  # List[Panel]
        for idx in range(len(panels)):
            rp, _, _ = panels[idx]
            new_mask = np.zeros(mask_img.shape)
            new_mask[l == rp.label] = 1
            panel_list.append(Panel(idx + 1, new_mask.astype(np.bool), rp.centroid, rp.bbox))
        self.panel_list = panel_list  #type: List[Panel]
        self.rprops = rprops

        with open(panel_data_f, "wb") as fh:
            pickle.dump(self.panel_list, fh)

    def _save_contour_image(self, img):
        fig = plt.figure()
        img_01 = self.all_imgs_list[0]
        plt.imshow(img_01)

        img_full_mask = np.zeros(img_01.shape[:-1])
        for p in self.panel_list:
            plt.annotate(str(p.label), xy=p.centroid[::-1], color='r', fontsize=20)
            min_row, min_col, max_row, max_col = p.bbox
            img_full_mask[min_row:max_row, min_col:max_col] += p.mask_crop

        self.panels_mask = img_full_mask.astype(np.bool)

        out_f = pj(self.exp_images_dir, "img_panels.jpg")
        if os.path.exists(out_f):
            return

        plt.contour(img_full_mask, [0.5], colors='r')
        fig.savefig(out_f)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(self.panels_mask)
        fig.savefig(pj(self.exp_images_dir, "panels_mask.jpg"))
        plt.close(fig)

    def _gmm_update_gmm(self, gmm, y):
        y_pred = gmm.predict(y)

        # previously used 1 / # examples, but parameters are updated too quickly
        alpha = 0.5 / float(y.shape[0])

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

    def _gmm_get_TCD(self, X, E, s, a, b):
        A = np.sum((X * E) / np.power(s, 2), axis=1)
        B = np.sum(np.power(E / s, 2))
        alpha = A / B
        print(alpha)
        all_NBD = (alpha - 1) / b

        alpha_tiled = np.repeat(alpha.reshape(alpha.shape[0], 1), 3, axis=1)

        # print blue_bg.shape, all_NBD.shape
        inner = np.power((X - (E * alpha_tiled)) / s, 2)
        all_NCD = np.sqrt(np.sum(inner, axis=1)) / b
        # plt.hist(all_NCD,bins=1000)

        TCD = np.percentile(all_NCD, 99.75)
        return TCD


    def _train_gmm_clfs(self):
        print(self.exp.bg_remover, "train")

        gmm_clf_f = pj(self.exp_gzdata_dir, "gmm_clf.pkl")
        if os.path.exists(gmm_clf_f):
            with open(gmm_clf_f, 'rb') as fh:
                self.classifiers = pickle.load(fh)
            return

        curr_img = imread(pj(self.exp.img_path, self.imgs[self.exp.start_img])) / 255.

        curr_mask = self._yuv_clip_image(self.imgs[self.exp.start_img])

        print(curr_mask)
        print(curr_mask[curr_mask is True])
        curr_mask = dilation(curr_mask, disk(2))

        bg_mask3 = np.dstack([np.logical_and(curr_mask, self.panels_mask)] * 3)
        bg_rgb_pixels = curr_img * bg_mask3

        # get all of the bg pixels
        bg_rgb_pixels = flatten_img(bg_rgb_pixels)
        bg_rgb_pixels = bg_rgb_pixels[bg_rgb_pixels.all(axis=1), :]
        bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
        bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain, replace=True)
        X = bg_rgb_pixels[bg_retain, :]

        blue_E, blue_s = X.mean(axis=0), X.std(axis=0)
        print(blue_E, blue_s)

        alpha = flatBD(X, blue_E, blue_s)
        a = np.sqrt(np.power(alpha - 1, 2) / X.shape[0])
        b = np.sqrt(np.power(flatCD(X, blue_E, blue_s), 2) / X.shape[0])
        print(a, b)

        TCD = self._gmm_get_TCD(X, blue_E, blue_s, a, b)
        print(TCD)
        print("Training GMM background remover...")
        bg_gmm = GaussianMixture(n_components=3)
        bg_gmm.fit(X)

        thresh = np.percentile(bg_gmm.score(X), 1.)
        print(thresh)

        horp_masks = []
        new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)
        print(new_E)

        self.classifiers = [bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E]

        with open(gmm_clf_f, "wb") as fh:
            pickle.dump(self.classifiers, fh)

    def _gmm_remove_background(self):

        print(self.exp.bg_remover, "remove")

        if len(os.listdir(self.exp_masks_dir)) >= ((self.exp.end_img-self.exp.start_img) - self.exp.start_img):
            return

        bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E = self.classifiers
        print(bg_gmm)

        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img-self.exp.start_img):
            return

        # 2d array of all the masks that are indexed first by image, then by panel.
        self.all_masks = []
        print(self.exp.start_img, self.exp.end_img)
        for idx in range(self.exp.start_img, self.exp.end_img):
            print(idx)

            img_f = self.imgs[idx]

            img = imread(pj(self.exp.img_path, img_f)) / 255.

            img_masks = []
            # generate the predicted mask for each panel and add them to the mask list.
            for p in self.panel_list:
                panel_img = p.get_cropped_image(img)
                pp_predicted = NCD(panel_img, new_E, blue_s, b) > TCD
                pp_predicted = pp_predicted.astype(np.bool)
                img_masks.append(pp_predicted)

            y = flatten_img(img)
            predicted = bg_gmm.score_samples(y)
            bg_retain = predicted > thresh
            y_bg = y[bg_retain, :]
            retain = random.choice(y_bg.shape[0], np.minimum(y_bg.shape[0], 50000), replace=False)
            y_bg = y_bg[retain, :]
            self._gmm_update_gmm(bg_gmm, y_bg)

            new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)

            print(new_E)

            with open(self.exp_masks_dir_frame % (idx), "wb") as fh:
                np.save(fh, img_masks)

            self.all_masks.append(img_masks)


    def _ensemble_predict(self, clfs, X):
        for clf_n, clf in clfs:
            y_pred = clf.predict(X)
        return y_pred

    def _train_clfs(self, clf_in):
        print("Classifier: ", self.exp.bg_remover)

        self.classifiers = None

        ensemble_clf_f = pj(self.exp_gzdata_dir, "ensemble_clf.pkl")
        if os.path.exists(ensemble_clf_f):
            with open(ensemble_clf_f, 'rb') as fh:
                self.classifiers = pickle.load(fh)

        if self.classifiers is not None:
            print('Loaded previous classifiers successfully')
            return

        # fig, axarr = plt.subplots(2, 3)
        # axarr = list(chain(*axarr))

        train_masks = []
        train_images = []

        train_img_ids = list(range(self.exp.start_img, self.exp.start_img + 3))
        train_img_ids += [int((self.exp.end_img+self.exp.start_img) / 2) - 1, self.exp.end_img - 2]

        for idx, img_i in enumerate(train_img_ids):
            curr_img = imread(pj(self.exp.img_path, self.imgs[img_i])) / 255.
            train_images.append(curr_img)

            curr_mask = self._yuv_clip_image(self.imgs[img_i])
            curr_mask = dilation(curr_mask, disk(3))
            train_masks.append(curr_mask.astype(np.bool))

            # axarr[idx].imshow(curr_mask)
            # axarr[idx].axis('off')

        # fig.savefig(pj(self.exp_images_dir, "train_imgs.jpg"))
        # plt.close(fig)

        all_bg_pixels = []
        all_fg_pixels = []

        for idx, (mask, curr_img) in enumerate(zip(train_masks, train_images)):
            # produce the background and foreground masks for this mask
            bg_mask3 = np.dstack([np.logical_and(mask, self.panels_mask)] * 3)
            fg_mask3 = np.dstack([np.logical_and(np.logical_not(mask), self.panels_mask)] * 3)

            bg_rgb_pixels = self._create_transformed_data(curr_img * bg_mask3)
            fg_rgb_pixels = self._create_transformed_data(curr_img * fg_mask3)

            print("Training background image shape: ", bg_rgb_pixels.shape)
            all_bg_pixels.append(bg_rgb_pixels)
            all_fg_pixels.append(fg_rgb_pixels)

        bg_rgb_pixels = np.vstack(all_bg_pixels)
        fg_rgb_pixels = np.vstack(all_fg_pixels)
        print("Training background image shape: ", bg_rgb_pixels.shape, "Training foreground image shape: ", fg_rgb_pixels.shape)

        # bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
        # bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain)

        # fg_retain = int(fg_rgb_pixels.shape[0] * 0.1)
        # fg_retain = random.choice(fg_rgb_pixels.shape[0], fg_retain)

        # bg_rgb_pixels = bg_rgb_pixels[bg_retain, :]
        # fg_rgb_pixels = fg_rgb_pixels[fg_retain, :]

        # print bg_rgb_pixels.shape, fg_rgb_pixels.shape        

        # make training data
        X = np.vstack([bg_rgb_pixels, fg_rgb_pixels])
        y = np.concatenate([
            np.zeros(bg_rgb_pixels.shape[0]),
            np.ones(fg_rgb_pixels.shape[0])
        ])

        self.classifiers = clf_in
        # self._sgd_hyperparameters(self.classifiers, ensemble_clf_f, X, y)
        self._train_clf(self.classifiers, ensemble_clf_f, X, y)

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
        train_img_ids += [int((self.exp.end_img + self.exp.start_img) / 2) - 1, int((self.exp.end_img + self.exp.start_img) / 2) - 2, int((self.exp.end_img + self.exp.start_img) / 2) - 3, int((self.exp.end_img + self.exp.start_img) / 2) - 4, self.exp.end_img - 2,self.exp.end_img - 1,self.exp.end_img - 3,self.exp.end_img - 4]

        train_images = []
        train_masks = []

        for idx, img_i in enumerate(train_img_ids):
            curr_img = imread(pj(self.exp.img_path, self.imgs[img_i])) / 255.
            curr_img = resize(curr_img, (int(curr_img.shape[0] / 2), int(curr_img.shape[1] / 2)),
                              anti_aliasing=True)
            curr_img = np.expand_dims(curr_img, axis=0)
            train_images.append(curr_img)

            curr_mask = self._yuv_clip_image(self.imgs[img_i])
            curr_mask = dilation(curr_mask, disk(3))
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
                curr_mask = self._yuv_clip_image(self.imgs[img_i])
                curr_mask = dilation(curr_mask, disk(3))
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

        for j in range(1, len(panel_images)+1):
            panel_images_j = panel_images[j]
            panel_images_j = np.vstack(panel_images_j)
            panel_masks_j = panel_masks[j]
            panel_masks_j = np.vstack(panel_masks_j)
            pshape = panel_images_j.shape[1:]
            model = create_unet(self, pshape, 2)
            callbacks = [
                EarlyStopping(patience=20, verbose=1),
                ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.0000001, verbose=1),
                ModelCheckpoint('model_panel%s.h5'%j, verbose=1, save_best_only=True, save_weights_only=False)
            ]
            model.compile(optimizer=Adam(lr=0.00001), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(panel_images_j, panel_masks_j, batch_size=2, epochs=200, callbacks=callbacks, verbose=2,
                      validation_split=0.2)
            models[j] = load_model('model_panel%s.h5'%j)

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
            fig = plt.figure()
            plt.imshow(pp_predicted)
            fig.savefig(pj(self.exp_images_dir, "panels_mask_%s.jpg" % idx))
            plt.close(fig)
            self.all_masks.append(img_masks)

            with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                np.save(fh, img_masks)

        # self.app.status_string.set("Removing background %d %%" % int(
        #     float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))



    def _train_clf(self, clf_in, ensemble_clf_f, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Shape of entire dataset: ", X.shape, y.shape)
        print("Shape of training dataset: ", X_train.shape, y_train.shape)
        print("Shape of testing dataset: ", X_test.shape, y_test.shape)

        for clf_n, clf in clf_in:
            print("Classifier: ", clf_n)
            clf.fit(X_train, y_train)
            print(clf_n, " train score: ", clf.score(X_train, y_train))
            print(clf_n, " test score: ", clf.score(X_test, y_test))

        y_pred = self._ensemble_predict(clf_in, X_test)
        print("Shape of predicted values: ", y_pred.shape)
        print("Score of the ensembled classifiers: ", accuracy_score(y_test, y_pred))

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
        # removed the random selection. Use as much data as possible.
        # get all of the bg pixels
        rgb_pixels = flatten_img(rgb_pixels)
        rgb_pixels = rgb_pixels[rgb_pixels.all(axis=1), :]
        # bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
        # bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain, replace=False)
        # bg_rgb_pixels = bg_rgb_pixels[bg_retain, :]

        return rgb_pixels

    def _remove_background(self):

        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img - self.exp.start_img):
            return

        self.all_masks = []

        print("Removing background...")

        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img_masks = []
            # can we load the mask, if we stop part way through training for some reason.
            try:
                start = time.time()
                img_masks = np.load(self.exp_masks_dir_frame % idx, allow_pickle=True)
            except Exception as e:
                img = self.all_imgs_list[idx]
                for p in self.panel_list:
                    start = time.time()
                    panel_img = p.get_cropped_image(img)
                    end = time.time()
                    print(end - start, "CROPPING TIME")
                    start = time.time()
                    pp_predicted = self._ensemble_predict(self.classifiers, flatten_img(panel_img))
                    end = time.time()
                    print(end - start, "ENSEMBLE PREDICT TIME")
                    pp_predicted.shape = p.mask_crop.shape
                    pp_predicted = pp_predicted.astype(np.bool)
                    img_masks.append(pp_predicted)

                with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                    np.save(fh, img_masks)

            # self.app.status_string.set("Removing background %d %%" % int(
            #     float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))
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

        # for i in range(self.exp.start_img, self.exp.start_img + retain):
        #     data = np.load(self.exp_masks_dir_frame % i, allow_pickle=True)     #UNCOMMENT THIS IF PROBLEMS ARISE
        #     init_masks.append(data)
        self.panel_l_rprops = []

        for idx, panel in enumerate(self.panel_list):

            # :10 for tomato, :20 for corn/brassica
            mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
            mask_med = clear_border(np.median(mask_med, axis=2)).astype(np.bool)
            mask_med = remove_small_objects(mask_med)

            # label features in an array using the default structuring element which is a cross.
            labelled_array, num_features = measurements.label(mask_med)
            print(num_features)
            rprops = regionprops(labelled_array)
            print(len(rprops))

            all_seed_rprops = []  # type: List[SeedPanel]
            for rp in rprops:
                all_seed_rprops.append(
                    SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
                              rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent, rp.convex_area))

            # Get maximum number of seeds
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            in_mask = find_closest_n_points(pts, self.exp.seeds_n)

            # if we've got less seeds than we should do, should we throw them away?
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

            # Remove extra 'seeds' (QR labels) from boundary
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            xy_range = get_xy_range(labelled_array)
            in_mask = find_pts_in_range(pts, xy_range)

            # if we've got less seeds than we should do, should we throw them away?
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

            # need to update pts if we have pruned.
            pts = np.vstack([el.centroid for el in all_seed_rprops])

            pts_order = order_pts_lr_tb(pts, self.exp.seeds_n, xy_range, self.exp.seeds_col_n, self.exp.seeds_row_n)

            print("all seed rprops length: " + str(len(all_seed_rprops)))

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

            # we add an array of labels and the region proprties for each panel.
            print("all seed rprops length: " + str(len(all_seed_rprops)))
            self.panel_l_rprops.append((labelled_array, all_seed_rprops))

        for idx in range(0, len(self.all_masks)):
            self.panel_l_rprops_1 = []
            for ipx, panel in enumerate(self.panel_list):
                # :10 for tomato, :20 for corn/brassica
                # mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
                print(idx, ipx)
                mask_med = init_masks[idx][ipx]
                # fig = plt.figure()
                # plt.imshow(mask_med)
                # if ipx == 1:
                    # fig.savefig(pj(self.exp_images_dir, "init_mask_med_{}.jpg".format(idx)))
                    # plt.close(fig)
                # mask_med = clear_border(np.median(mask_med, axis=2)).astype(np.bool)
                # mask_med = clear_border(mask_med).astype(np.bool)
                # fig = plt.figure()
                # plt.imshow(mask_med)
                # fig.savefig(pj(self.exp_images_dir, "mask_med_after_border_clear.jpg"))
                # plt.close(fig)
                mask_med = remove_small_objects(mask_med)

                # label features in an array using the default structuring element which is a cross.
                labelled_array, num_features = measurements.label(mask_med)
                # fig = plt.figure()
                # plt.imshow(labelled_array)
                # fig.savefig(pj(self.exp_images_dir, "labelled_array.jpg"))
                # plt.close(fig)
                print('Number of features: ', num_features)
                rprops = regionprops(labelled_array, coordinates='xy')
                print('Length of rprops: ', len(rprops))

                all_seed_rprops = []  # type: List[SeedPanel]
                for rp in rprops:
                    all_seed_rprops.append(
                        SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
                                  rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent, rp.convex_area))

                # Get maximum number of seeds
                if all_seed_rprops == []:
                    break
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                in_mask = find_closest_n_points(pts, self.exp.seeds_n)

                # if we've got less seeds than we should do, should we throw them away?
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

                # Remove extra 'seeds' (QR labels) from boundary
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                xy_range = get_xy_range(labelled_array)
                in_mask = find_pts_in_range(pts, xy_range)

                # if we've got less seeds than we should do, should we throw them away?
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

                # need to update pts if we have pruned.
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

                # we add an array of labels and the region properties for each panel.
                print("Number of seeds identified: " + str(len(all_seed_rprops)))
                self.panel_l_rprops_1.append((labelled_array, all_seed_rprops))

                axarr[ipx].imshow(mask_med)
                axarr[ipx].annotate(str(panel.label), xy=(20, 10), color='r', fontsize=28)

                if idx == 0:        # This is here to prevent red numbers being created every iteration, either do it once or update the locations of them
                    for rp in all_seed_rprops:
                        axarr[ipx].annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r', fontsize=16)

                axarr[ipx].axis('off')

            c = str(idx)
            filename = "seeds_labelled" + c + ".png"                          #UNCOMMENT IF WANT MASKS
            fig.savefig(pj(self.exp_images_dir, filename))
            plt.close(fig)
            self.all_rprops.append(self.panel_l_rprops_1)

    def _generate_statistics(self):
        l_rprops_f = pj(self.exp_gzdata_dir, "l_rprops_data.pkl")
        if os.path.exists(l_rprops_f):
            with open(l_rprops_f, 'rb') as fh:
                self.all_rprops = pickle.load(fh)
        for j in range(len(self.all_rprops)):
            print(j+1)
            x = self.all_rprops[j]
            n_seeds = 0
            for p in range(len(x)):
                n_seeds += len(x[p][1])
            X_stats = np.zeros((n_seeds, 10))
            counter = 0
            for i in range(len(x)):
                x0 = x[i][1]
                for k in range(len(x0)):
                    X_stats[counter, :] = [i+1, k+1, x0[k].area, x0[k].eccentricity, x0[k].extent,
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
        stats_over_time = pd.DataFrame(seed_stats, columns=['Image Index', 'Panel Number', 'Seed Number', 'Seed Area', 'Seed Eccentricity', 'Seed Extent', 'Seed Major Axis Length', 'Seed Minor Axis Length', 'Seed Perimeter', 'Seed Solidity', 'Seed Convex Area'])
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

        if len(glob.glob(pj(self.exp_results_dir, "germ_panel_*.json"))) >= self.exp.panel_n:
            print("Already analysed data")
            # print "Continuing for testing..."
            return

        if self.all_masks is None:
            self.all_masks = []
            for i in range(self.exp.start_img, self.exp.end_img):
                data = np.load(self.exp_masks_dir_frame % (i), allow_pickle=True)
                self.all_masks.append(data)

        #evaluate each panel separately.
        for panel_idx, panel_object in enumerate(self.panel_list):
            try:
                print("panel %d" % panel_idx)

                # this is the tuple of the labelled arrays genereated from mesurements.
                # label and the regionprops generated by the measurements.regionprops
                panel_labels, panel_regionprops = self.panel_l_rprops[panel_idx]

                # extract all the masks for the specific panel, for every image.
                p_masks = np.array([el[panel_idx] for el in self.all_masks])

                self.spp_processor.use_colour = self.exp.use_colour

                panel_germ = self.spp_processor._classify(
                    panel_idx,
                    panel_object,
                    self.all_imgs[self.exp.start_img:self.exp.end_img],
                    panel_labels,
                    panel_regionprops,
                    p_masks
                )

                print("panel germ length " + str(len(panel_germ)))

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
                    curr_seed[:idx - 1] = 0
                    curr_seed[idx - 1:] = 1
                    break
                idx += 1
            if idx >= (curr_seed.shape[0] - win):
                curr_seed[:] = 0
        return germ.sum(axis=0), germ

    def _analyse_results(self, proprtions):
        all_germ = []
        for i in range(self.exp.panel_n):
            # if the with fails to open, we should not perform operations on germ_d.
            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (i))) as fh:
                germ_d = json.load(fh)

                # ensure the germ_d isn't empty.
                if len(germ_d) == 0:
                    continue

                germinated = []
                for j in range(1, len(germ_d.keys()) + 1):
                    germinated.append(germ_d[str(j)])

                germinated = np.vstack(germinated)
                all_germ.append(germinated)

        p_totals = []
        for i in range(self.exp.panel_n):
            l, rprop = self.panel_l_rprops[i]
            p_totals.append(len(rprop))

        print(len(all_germ))
        if len(all_germ) == 0:
            raise Exception("Germinated seeds found is 0. Try changing YUV values.")

        print(p_totals)

        cum_germ_data = []
        for germ in all_germ:
            cum_germ_data.append(self._get_cumulative_germ(germ, win=1)[0])

        initial_germ_time_data = []
        for germ in all_germ:
            cols, rows = germ.shape
            init_germ_time = []
            for m in range(cols):
                for n in range(rows):
                    if germ[m, n]:
                        # For adding offset
                        init_germ_time.append(n + self.exp.start_img)
                        break
            initial_germ_time_data.append(init_germ_time)

        cum_germ_data = np.vstack(cum_germ_data).T.astype('f')

        print(len(cum_germ_data))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12., 10.))

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
                range(self.exp.start_img, n_frames),
                cum_germ_data[:, idx] / float(p_totals[idx]), label="Genotype" + str(idx + 1)
            )
        ax1.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])

        if has_date:
            # Sort out xtick labels in hours
            xtick_labels = []
            for val in ax1.get_xticks():
                if int(val) >= (self.exp.end_img-self.exp.start_img):
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
            cum_germ = cum_germ_data[:, idx].copy().ravel()
            cum_germ /= p_totals[idx]

            germ_pro_total = cum_germ[-1]
            prop_idxs = []
            for pro in proprtions:
                if (cum_germ > pro).any():
                    pos_idx = np.argmax(cum_germ >= pro)

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

        ax3.boxplot(initial_germ_time_data, vert=False)
        ax3.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])
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

        print(cum_germ_data[-1, :] / np.array(p_totals))

        ax4.barh(np.arange(self.exp.panel_n) + 0.75, cum_germ_data[-1, :] / np.array(p_totals), height=0.5)
        ax4.set_yticks(range(1, 1 + self.exp.panel_n))
        ax4.set_ylim([0.5, self.exp.panel_n + .5])
        ax4.set_xlim([0., 1.])
        ax4.set_ylabel("Panel number")
        ax4.set_xlabel("Germinated proportion")
        ax4.set_title("Proportion germinated")

        fig.savefig(pj(self.exp_results_dir, "results.jpg"))
        plt.close(fig)

        img_index = np.arange(n_frames) + self.exp.start_img

        if has_date:
            times_index = []
            for _i in img_index:
                curr_dt = s_to_datetime(self.imgs[_i])
                times_index.append(hours_between(start_dt, curr_dt))

            times_index = np.array(times_index).reshape(-1, 1)
            cum_germ_data = np.hstack([times_index, cum_germ_data])

        print(cum_germ_data.shape)

        df = pd.DataFrame(data=cum_germ_data)
        df.index = img_index

        if has_date:
            df.columns = ["Time"] + [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', 1:] = p_totals
        else:
            df.columns = [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', :] = p_totals

        print(df.columns)

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

        img_f = imread(self.all_imgs[self.exp.start_img])
        f_masks = np.load(self.exp_masks_dir_frame % (self.exp.start_img), allow_pickle=True)

        img_l = imread(self.all_imgs[-1])
        l_masks = np.load(self.exp_masks_dir_frame % ((self.exp.end_img-self.exp.start_img) - 1), allow_pickle=True)

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

            # print prop_idxs

            p_f_img = self.panel_list[p_idx].get_bbox_image(img_f)
            p_f_mask = f_masks[p_idx]

            p_l_img = self.panel_list[p_idx].get_bbox_image(img_l)
            p_l_mask = l_masks[p_idx]

            f_rgb_mu = p_f_img[p_f_mask].mean(axis=0)
            l_rgb_mu = p_l_img[p_l_mask].mean(axis=0)
            f_rgb_mu = tuple(np.round(f_rgb_mu).astype('i'))
            l_rgb_mu = tuple(np.round(l_rgb_mu).astype('i'))

            # print "init_rgb", f_rgb_mu
            # print "germ_rgb", l_rgb_mu

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

                self._extract_panels(self.imgs[self.exp.start_img], self.core.chunk_no, self.core.chunk_reverse)

                # self.app.status_string.set("Saving contour image")

                self._save_contour_image(self.imgs[self.exp.start_img])

                # self.app.status_string.set("Training background removal clfs")

                if self.exp.bg_remover == 'UNet':
                    # self.app.status_string.set("Removing background")
                    self._train_unet()
                elif self.exp.bg_remover == "SGD":
                    self._train_clfs([("sgd", SGDClassifier(max_iter=1000, tol=None, random_state=0))])
                    # self.app.status_string.set("Removing background")
                    self._remove_background()
                elif self.exp.bg_remover == "GMM":
                    self._train_gmm_clfs()
                    # self.app.status_string.set("Removing background")
                    self._gmm_remove_background()
                else:
                    print(".... unknown BG classifier")

                # self.app.status_string.set("Labelling seeds")
                self._label_seeds()

                # self.app.status_string.set("Generating statistics")
                self._generate_statistics()

                # self.app.status_string.set("Performing classification")
                self._perform_classification()

                # self.app.status_string.set("Analysing results")

                self._analyse_results(self.core.proportions)

                # self.app.status_string.set("Quantifying initial seed data")
                self._quantify_first_frame(self.core.proportions)

                # self.app.status_string.set("Finished processing")
                # self.exp.status = "Finished"

                print()
                print(time.time() - start)
                print()

                print("Finished")

            except Exception as e:
                raise e
                #self.exp.status = "Error"
                #self.app.status_string.set("Error whilst processing")
                #print("Exception args: " + str(e.args))

            self.running = False

            self.core.stop_processor(self.exp.eid)

    def die(self):
        self.running = False
