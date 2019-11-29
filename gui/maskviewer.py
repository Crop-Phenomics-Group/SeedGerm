#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import tkinter as Tkinter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from helper.functions import get_images_from_dir

pj = os.path.join


class MaskViewer(Tkinter.Toplevel):
    
    def __init__(self, exp):
        Tkinter.Toplevel.__init__(self)
        
        self.title("View masks")
        self.iconbitmap(sys._MEIPASS + '.\logo.ico')
        
        self.exp = exp

        self.imgs = get_images_from_dir(self.exp.img_path)
        
        self.exp_masks_dir = pj(self.exp.exp_path, "masks")
        self.exp_masks_dir_frame = pj(self.exp_masks_dir, "frame_%d.npy")
        self.exp_results_dir = pj(self.exp.exp_path, "results")
        
        self.showing_scatter = False
        self.panel_seed_idxs = None
        self.scatter = None
        
        scatter_f = pj(self.exp_results_dir, "panel_seed_idxs.json")
        if os.path.exists(scatter_f):
            with open(scatter_f) as fh:
                self.panel_seed_idxs = json.load(fh)
            self.showing_scatter = True
        
        self.num_images = len(self.imgs)
        self.curr_panel = 0        
        self.pan_btns = []
        
        self.playing = False
        
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)

        for tic in self.ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        
        for tic in self.ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        
        self.tk_fig_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.tk_fig_canvas.draw()
        self.tk_fig_canvas_widget = self.tk_fig_canvas.get_tk_widget()
        
        self.fn_label_var = Tkinter.StringVar()
        self.fn_label_var.set("")
        self.filename_label = Tkinter.Label(
            master=self,
            textvariable=self.fn_label_var
        )        
        
        #img = imread(img_path)
        self.curr_img = self.exp.start_img
        print(self.curr_img)
        self.img_plot = self.ax.imshow(self._get_image(self.curr_img))
        
        self.panel_frame = Tkinter.Frame(self)
        
        self._add_panel_selectors()
        
        self.prev_button = Tkinter.Button(
            master=self,
            text="Previous",
            command=self._prev_click
        )
        self.play_stop_button = Tkinter.Button(
            master=self,
            text="Start",
            command=self._play_stop_click
        )
        self.next_button = Tkinter.Button(
            master=self,
            text="Next",
            command=self._next_click
        )        
        
        self.filename_label.grid(
            in_=self,
            column=1,
            row=0,
            columnspan=3,
            sticky='news'
        )        
        self.tk_fig_canvas_widget.grid(
            in_=self,
            column=1,
            row=1,
            columnspan=3,
            sticky='news'
        )
        self.prev_button.grid(
            in_=self,
            column=1,
            row=2,
            #sticky='
        )
        self.play_stop_button.grid(
            in_=self,
            column=2,
            row=2,
        )
        self.next_button.grid(
            in_=self,
            column=3,
            row=2,
        )
        self.panel_frame.grid(
            in_=self,
            column=4,
            row=1,
            rowspan=3,
        )

    def _panel_btns(self, idx):       
        if idx == self.curr_panel:
            return

        self._stop()    
    
        self.pan_btns[self.curr_panel].config(relief=Tkinter.RAISED)
        self.pan_btns[idx].config(relief=Tkinter.SUNKEN)    
        self.curr_panel = idx
        
        self.ax = self.fig.add_subplot(111)
        self.curr_img = self.exp.start_img
        this_img = self._get_image(self.curr_img)
        m, n = this_img.shape
        self.img_plot = self.ax.imshow(this_img)
        
        # Clear any points that are there currently
        self._add_scatter_points(m, n) 
        
        self.fig.canvas.draw()

    def _add_panel_selectors(self):
        n = self.exp.panel_n
        for i in range(n):
            pan_txt = "Panel %d" % (i + 1)
            self.pan_btns.append(
                Tkinter.Button(
                    master=self.panel_frame,
                    text=pan_txt,
                    command=lambda i=i: self._panel_btns(i))
                )
            self.pan_btns[-1].pack()
        self.pan_btns[0].config(relief=Tkinter.SUNKEN)
        
    def _get_image(self, idx):
        self.fn_label_var.set("%d: %d" % (self.curr_panel + 1, idx))
        masks = np.load(self.exp_masks_dir_frame % idx, allow_pickle=True)
        return masks[self.curr_panel]
    
    def _stop(self):    
        self.play_stop_button.config(text="Start")
        self.playing = False
    
    def _next_image(self):
        if self.playing:
            if self.curr_img == (self.num_images - 1):
                self._stop()
                return
            self._next_click()
            self.after(10, self._next_image)
    
    def _play_stop_click(self):
        if self.playing:
            self._stop()
        else:
            self.playing = True
            self.play_stop_button.config(text="Stop")
            self._next_image()
            
    def _add_scatter_points(self, m, n):
        if not self.showing_scatter:
            self.ax.set_xlim(0, n)
            self.ax.set_ylim(m, 0)
            return
    
        if self.scatter != None:
            self.scatter.remove()
            del self.scatter
            
        self.scatter = None
        
        ys, xs = [], []
        for i, y, x in self.panel_seed_idxs[str(self.curr_panel)]:
            if i == -1:
                continue
            if self.curr_img >= i: 
                ys.append(y)
                xs.append(x)
        if len(ys) and len(xs):
            self.scatter = self.ax.scatter(xs, ys, c='r', s=40)
            self.ax.set_xlim(0, n)
            self.ax.set_ylim(m, 0)

    def _prev_click(self):
        self.curr_img -= 1
        if self.curr_img < self.exp.start_img:
            self.curr_img = self.num_images - 1
            
        this_img = self._get_image(self.curr_img)
        m, n = this_img.shape
        self.img_plot.set_data(this_img)
        self._add_scatter_points(m, n)
        self.fig.canvas.draw()
    
    def _next_click(self):
        self.curr_img += 1
        if self.curr_img > (self.num_images - 1):
            self.curr_img = self.exp.start_img

        this_img = self._get_image(self.curr_img)
        m, n = this_img.shape
        self.img_plot.set_data(this_img)
        self._add_scatter_points(m, n)
        self.fig.canvas.draw()
