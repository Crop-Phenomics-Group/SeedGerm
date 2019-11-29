# -*- coding: utf-8 -*-

import os
import sys
import tkinter as Tk

import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from helper.experiment import Experiment

pj = os.path.join
sns.set_style("white")


class ViewResults(Tk.Toplevel):
    
    def __init__(self, exp):
        Tk.Toplevel.__init__(self)

        self.iconbitmap(sys._MEIPASS + '.\logo.ico')

        self.exp = exp #type: Experiment
        self.exp_results_graph = pj(self.exp.get_results_dir(), "results.jpg")
        
        self.graph_img = imread(self.exp_results_graph)

        self.fig = plt.Figure(facecolor='white', figsize=(10., 8.))
        
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.graph_img)
        
        self.ax.axis('off')

        self.tk_fig_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.tk_fig_canvas.draw()
        self.tk_fig_canvas_widget = self.tk_fig_canvas.get_tk_widget()

        self.tk_toolbar = NavigationToolbar2Tk(self.tk_fig_canvas, self)
        self.tk_toolbar.update()

        self.tk_fig_canvas_widget.pack(side=Tk.BOTTOM, fill=Tk.X)
        self.tk_toolbar.pack(side=Tk.TOP, fill=Tk.X)