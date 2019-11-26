# -*- coding: utf-8 -*-

import os
import tkinter as Tkinter

import matplotlib.pyplot as plt
from imageio import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

pj = os.path.join


class ImageViewer(Tkinter.Toplevel):
    
    def __init__(self, directory, images):
        Tkinter.Toplevel.__init__(self)
        
        self.title("Image viewer")
        self.iconbitmap('.\logo.ico')
        
        self.directory = directory
        self.images = images
        
#        self.directory = "../../data/5022_Test_Rep4_CAM01_panel1/"
#        self.images = os.listdir(self.directory)
#        self.images = sorted(self.images, key=lambda s: int(re.findall('ID-(\d+)_', s)[0]))
#        
#        self.images = self.images[-100:]        
        
        self.num_images = len(self.images)
        
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
        self.curr_img = 0
        self.img_plot = self.ax.imshow(self._get_image(self.curr_img))
        
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
        

    def _get_image(self, idx):
        self.fn_label_var.set(self.images[idx])
        return imread(pj(self.directory, self.images[idx]))
    
    def _stop(self):
        self.play_stop_button.config(text="Start")
        self.playing = False
    
    def _next_image(self):
        if self.playing:
            if self.curr_img == (self.num_images - 1):
                self._stop()
                return
            self._next_click()
            self.after(34, self._next_image)
    
    def _play_stop_click(self):
        if self.playing:
            self._stop()
        else:
            self.playing = True
            self.play_stop_button.config(text="Stop")
            self._next_image()

    def _prev_click(self):
        self.curr_img -= 1
        if self.curr_img < 0:
            self.curr_img = self.num_images - 1
        self.img_plot.set_data(self._get_image(self.curr_img))
        self.fig.canvas.draw()
    
    def _next_click(self):
        self.curr_img += 1
        if self.curr_img > (self.num_images - 1):
            self.curr_img = 0
        self.img_plot.set_data(self._get_image(self.curr_img))
        self.fig.canvas.draw()