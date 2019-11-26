# -*- coding: utf-8 -*-

"""
Created on Tue Jan 12 14:31:56 2016

@author: dty09rcu
"""

import json
import tkinter as Tkinter
from itertools import chain
from operator import itemgetter
from tkinter import messagebox

import matplotlib.pyplot as plt
import seaborn as sns
import skimage
from imageio import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects, erosion, disk
from skimage.transform import rescale

from helper.experiment import Experiment
from helper.functions import *
from helper.panel_segmenter import fill_border

sns.set_style("white")


class YUVRanges(Tkinter.Toplevel):
    def __init__(self, app, exp):
        Tkinter.Toplevel.__init__(self)

        self.maxsize(width=1260, height=720)
        self.app = app
        self.exp = exp  # type: Experiment
        self.yuv_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges.json"
        )

        self.title("Set YUV ranges")
        self.resizable(width=False, height=False)
        self.iconbitmap('.\logo.ico')

        start_img = int(self.exp.start_img)
        end_img = int(self.exp.end_img)

        img_path = self.exp.img_path
        imgs = get_images_from_dir(img_path)

        img_01 = imread(os.path.join(img_path, imgs[start_img]))
        img_01 = skimage.img_as_ubyte(rescale(img_01, 0.25))
        img_02 = imread(os.path.join(img_path, imgs[start_img + int((end_img - start_img) / 3)]))
        img_02 = skimage.img_as_ubyte(rescale(img_02, 0.25))
        img_03 = imread(os.path.join(img_path, imgs[start_img + int(2 * (end_img - start_img) / 3)]))
        img_03 = skimage.img_as_ubyte(rescale(img_03, 0.25))
        img_04 = imread(os.path.join(img_path, imgs[end_img - 1]))
        img_04 = skimage.img_as_ubyte(rescale(img_04, 0.25))

        self.img_01_yuv = rgb2ycrcb(img_01)
        self.img_01_rgb = img_01.copy() / 255.
        self.img_02_yuv = rgb2ycrcb(img_02)
        self.img_02_rgb = img_02.copy() / 255.
        self.img_03_yuv = rgb2ycrcb(img_03)
        self.img_03_rgb = img_03.copy() / 255.
        self.img_04_yuv = rgb2ycrcb(img_04)
        self.img_04_rgb = img_04.copy() / 255.

        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(2, 2, 1)
        self.ax1 = self.fig.add_subplot(2, 2, 2)
        self.ax2 = self.fig.add_subplot(2, 2, 3)
        self.ax3 = self.fig.add_subplot(2, 2, 4)

        for tic in self.ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False


        for tic in self.ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax1.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax1.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax2.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax2.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax3.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        for tic in self.ax3.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            tic.tick1line.set_visible = False
            tic.tick2line.set_visible = False
            tic.label1.set_visible = False
            tic.label2.set_visible = False

        self.tk_fig_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.tk_fig_canvas.draw()
        self.tk_fig_canvas_widget = self.tk_fig_canvas.get_tk_widget()

        self.img_plot = self.ax.imshow(img_01)
        self.img_plot1 = self.ax1.imshow(img_02)
        self.img_plot2 = self.ax2.imshow(img_03)
        self.img_plot3 = self.ax3.imshow(img_04)

        self.y_label = Tkinter.Label(master=self, text="Y")
        self.u_label = Tkinter.Label(master=self, text="U")
        self.v_label = Tkinter.Label(master=self, text="V")

        scale_length = 255

        self.y_low = Tkinter.Scale(
            self,
            label="Low",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )
        self.y_high = Tkinter.Scale(
            self,
            label="High",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )

        self.u_low = Tkinter.Scale(
            self,
            label="Low",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )
        self.u_high = Tkinter.Scale(
            self,
            label="High",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )

        self.v_low = Tkinter.Scale(
            self,
            label="Low",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )
        self.v_high = Tkinter.Scale(
            self,
            label="High",
            from_=0,
            to=255,
            orient=Tkinter.HORIZONTAL,
            length=scale_length,
            command=self._handle_slide
        )

        self.y_high.set(255)
        self.u_high.set(255)
        self.v_high.set(255)

        self.save_button = Tkinter.Button(
            self,
            text="Save values",
            command=self._save_values,
        )

        self.bind('<Return>', self._save_values)

        self.y_label.grid(in_=self, column=2, row=1, )
        self.y_low.grid(in_=self, column=2, row=2, )
        self.y_high.grid(in_=self, column=2, row=3, )

        self.u_label.grid(in_=self, column=2, row=4, )
        self.u_low.grid(in_=self, column=2, row=5, )
        self.u_high.grid(in_=self, column=2, row=6, )

        self.v_label.grid(in_=self, column=2, row=7, )
        self.v_low.grid(in_=self, column=2, row=8, )
        self.v_high.grid(in_=self, column=2, row=9, )

        self.save_button.grid(in_=self, column=2, row=12, )

        self.tk_fig_canvas_widget.grid(
            in_=self,
            column=1,
            row=1,
            # columnspan=3,
            rowspan=12,
            sticky='news'
        )

        self._check_yuv_set()

    def _handle_slide(self, event):
        self.yuv_low = np.array([
            self.y_low.get(),
            self.u_low.get(),
            self.v_low.get()
        ])
        self.yuv_high = np.array([
            self.y_high.get(),
            self.u_high.get(),
            self.v_high.get(),
        ])

        mask = in_range(self.img_01_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot.set_data(self.img_01_rgb * mask_stack)
        mask = in_range(self.img_02_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot1.set_data(self.img_02_rgb * mask_stack)
        mask = in_range(self.img_03_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot2.set_data(self.img_03_rgb * mask_stack)
        mask = in_range(self.img_04_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot3.set_data(self.img_04_rgb * mask_stack)
        self.fig.canvas.draw()

    def _check_yuv_set(self):
        if os.path.exists(self.yuv_json_file):
            with open(self.yuv_json_file) as fh:
                yuv_ranges = json.load(fh)

            low = yuv_ranges['low']
            high = yuv_ranges['high']

            self.y_low.set(low[0])
            self.u_low.set(low[1])
            self.v_low.set(low[2])

            self.y_high.set(high[0])
            self.u_high.set(high[1])
            self.v_high.set(high[2])

    def _save_values(self, event=None):
        yuv_ranges = {'low': self.yuv_low.tolist(), 'high': self.yuv_high.tolist()}
        yes_overwrite = True
        if os.path.exists(self.yuv_json_file):
            yes_overwrite = messagebox.askyesno(
                "",
                "YUV values set, are you sure you want to overwrite?"
            )

        if yes_overwrite:
            print("setting yuv")
            self.exp.yuv_ranges_set = True
            with open(self.yuv_json_file, "w+") as fh:
                json.dump(yuv_ranges, fh)
            self.destroy()


class YUVPanelRanges(Tkinter.Toplevel):
    def __init__(self, app, exp, idx_p=0):
        Tkinter.Toplevel.__init__(self)

        self.maxsize(width=1260, height=720)
        self.app = app
        self.exp = exp  # type: Experiment
        self.idx_p = idx_p
        self.yuv_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges.json"
        )

        self.title("Set YUV ranges")
        self.resizable(width=False, height=False)
        self.iconbitmap('.\logo.ico')

        data = json.load(open("config.json"))
        self.chunk_no = data["chunk_no"]
        self.chunk_reverse = data["chunk_reverse"]

        start_img = int(self.exp.start_img)
        end_img = int(self.exp.end_img)

        img_path = self.exp.img_path
        imgs = get_images_from_dir(img_path)

        self.img_01 = imread(os.path.join(img_path, imgs[start_img]))
        self.img_01 = skimage.img_as_ubyte(self.img_01)
        self.img_02 = imread(os.path.join(img_path, imgs[start_img + int((end_img - start_img) / 3)]))
        self.img_02 = skimage.img_as_ubyte(self.img_02)
        self.img_03 = imread(os.path.join(img_path, imgs[start_img + int(2 * (end_img - start_img) / 3)]))
        self.img_03 = skimage.img_as_ubyte(self.img_03)
        self.img_04 = imread(os.path.join(img_path, imgs[end_img - 1]))
        self.img_04 = skimage.img_as_ubyte(self.img_04)

        with open(self.yuv_json_file) as fh:
            yuv_ranges = json.load(fh)

        self.yuv_low = yuv_ranges['low']
        self.yuv_high = yuv_ranges['high']

        def _yuv_clip_image(img):
            img_yuv = rgb2ycrcb(img)
            mask_img = in_range(img_yuv, self.yuv_low, self.yuv_high)
            return mask_img.astype(np.bool)

        mask_img = _yuv_clip_image(self.img_01)
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
            _, panel_object_count = measurements.label(both_mask)  # , return_num=True)
            return panel_object_count

        rprops = [(rp, get_mask_objects(idx, rp)) for idx, rp in enumerate(rprops)]
        rprops = sorted(rprops, key=itemgetter(1), reverse=True)

        # Check the panel has seeds in it

        panels = [(rp, rp.centroid[0], rp.centroid[1]) for rp, _ in rprops[:self.exp.panel_n]]

        # sort panels based on y first, then x
        panels = sorted(panels, key=itemgetter(1))
        panels = chunks(panels, self.chunk_no)
        panels = [sorted(p, key=itemgetter(2), reverse=self.chunk_reverse) for p in panels]
        print(panels)
        panels = list(chain(*panels))

        # set mask, where 1 is top left, 2 is top right, 3 is middle left, etc
        panel_list = []  # List[Panel]
        for idx in range(len(panels)):
            rp, _, _ = panels[idx]
            new_mask = np.zeros(mask_img.shape)
            new_mask[l == rp.label] = 1
            panel_list.append(rp.bbox)
        self.panel_list = panel_list  # type: List[Panel]

        self.idx_max = len(panel_list)

        print(self.panel_list)

        def _panel_yuv(self, p):
            self.idx = self.idx_p
            p = self.panel_list[idx_p]
            print(p)
            self.indicator = 0
            img_01 = self.img_01[p[0]:p[2], p[1]:p[3]]
            img_02 = self.img_02[p[0]:p[2], p[1]:p[3]]
            img_03 = self.img_03[p[0]:p[2], p[1]:p[3]]
            img_04 = self.img_04[p[0]:p[2], p[1]:p[3]]
            self.img_01_yuv = rgb2ycrcb(img_01)
            self.img_01_rgb = img_01.copy() / 255.
            self.img_02_yuv = rgb2ycrcb(img_02)
            self.img_02_rgb = img_02.copy() / 255.
            self.img_03_yuv = rgb2ycrcb(img_03)
            self.img_03_rgb = img_03.copy() / 255.
            self.img_04_yuv = rgb2ycrcb(img_04)
            self.img_04_rgb = img_04.copy() / 255.

            self.fig = plt.Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(2, 2, 1)
            self.ax1 = self.fig.add_subplot(2, 2, 2)
            self.ax2 = self.fig.add_subplot(2, 2, 3)
            self.ax3 = self.fig.add_subplot(2, 2, 4)

            for tic in self.ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax1.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax1.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax2.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax2.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax3.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            for tic in self.ax3.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                tic.tick1line.set_visible = False
                tic.tick2line.set_visible = False
                tic.label1.set_visible = False
                tic.label2.set_visible = False

            self.tk_fig_canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.tk_fig_canvas.draw()
            self.tk_fig_canvas_widget = self.tk_fig_canvas.get_tk_widget()

            self.img_plot = self.ax.imshow(img_01)
            self.img_plot1 = self.ax1.imshow(img_02)
            self.img_plot2 = self.ax2.imshow(img_03)
            self.img_plot3 = self.ax3.imshow(img_04)

            self.y_label = Tkinter.Label(master=self, text="Y")
            self.u_label = Tkinter.Label(master=self, text="U")
            self.v_label = Tkinter.Label(master=self, text="V")

            scale_length = 255

            self.y_low = Tkinter.Scale(
                self,
                label="Low",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )
            self.y_high = Tkinter.Scale(
                self,
                label="High",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )

            self.u_low = Tkinter.Scale(
                self,
                label="Low",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )
            self.u_high = Tkinter.Scale(
                self,
                label="High",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )

            self.v_low = Tkinter.Scale(
                self,
                label="Low",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )
            self.v_high = Tkinter.Scale(
                self,
                label="High",
                from_=0,
                to=255,
                orient=Tkinter.HORIZONTAL,
                length=scale_length,
                command=self._handle_slide
            )

            self.y_high.set(255)
            self.u_high.set(255)
            self.v_high.set(255)

            self.save_button = Tkinter.Button(
                self,
                text="Save values",
                command=self._save_values_panel,
            )

            self.bind('<Return>', self._save_values_panel)

            self.y_label.grid(in_=self, column=2, row=1, )
            self.y_low.grid(in_=self, column=2, row=2, )
            self.y_high.grid(in_=self, column=2, row=3, )

            self.u_label.grid(in_=self, column=2, row=4, )
            self.u_low.grid(in_=self, column=2, row=5, )
            self.u_high.grid(in_=self, column=2, row=6, )

            self.v_label.grid(in_=self, column=2, row=7, )
            self.v_low.grid(in_=self, column=2, row=8, )
            self.v_high.grid(in_=self, column=2, row=9, )

            # self.save_button.grid(in_=self, column=2, row=12, )

            self.tk_fig_canvas_widget.grid(
                in_=self,
                column=1,
                row=1,
                # columnspan=3,
                rowspan=12,
                sticky='news'
            )

            self._check_yuv_set(self.idx)

        _panel_yuv(self, panel_list[0])

    def _handle_slide(self, event):
        self.yuv_low = np.array([
            self.y_low.get(),
            self.u_low.get(),
            self.v_low.get()
        ])
        self.yuv_high = np.array([
            self.y_high.get(),
            self.u_high.get(),
            self.v_high.get(),
        ])

        mask = in_range(self.img_01_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot.set_data(self.img_01_rgb * mask_stack)
        mask = in_range(self.img_02_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot1.set_data(self.img_02_rgb * mask_stack)
        mask = in_range(self.img_03_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot2.set_data(self.img_03_rgb * mask_stack)
        mask = in_range(self.img_04_yuv, self.yuv_low, self.yuv_high)
        mask_stack = np.dstack([mask * 3]).astype(np.bool)
        self.img_plot3.set_data(self.img_04_rgb * mask_stack)
        self.fig.canvas.draw()

    def _check_yuv_set(self, idx):
        new_path = os.path.join(
            self.exp.exp_path,
            "yuv_ranges_{}.json".format(idx)
        )
        if os.path.exists(new_path):
            with open(new_path) as fh:
                yuv_ranges = json.load(fh)

            low = yuv_ranges['low']
            high = yuv_ranges['high']
            self.yuv_low = yuv_ranges['low']
            self.yuv_high = yuv_ranges['high']

            self.y_low.set(low[0])
            self.u_low.set(low[1])
            self.v_low.set(low[2])

            self.y_high.set(high[0])
            self.u_high.set(high[1])
            self.v_high.set(high[2])

    def _save_values_panel(self, event=None):
        yuv_ranges = {'low': self.yuv_low.tolist(), 'high': self.yuv_high.tolist()}
        yes_overwrite = True
        new_path = os.path.join(
            self.exp.exp_path,
            "yuv_ranges_{}.json".format(self.idx_p + 1)
        )

        print(new_path, self.idx_p)

        if os.path.exists(new_path):
            yes_overwrite = messagebox.askyesno(
                "",
                "YUV values set, are you sure you want to overwrite?"
            )

        if yes_overwrite:
            print("setting yuv")
            with open(new_path, "w+") as fh:
                json.dump(yuv_ranges, fh)
            self.destroy()

        print(self.idx_p + 1, self.idx_max)

        if self.idx_p + 1 < self.idx_max:
            YUVPanelRanges.__init__(self, app=self.app, exp=self.exp, idx_p=self.idx_p + 1)
