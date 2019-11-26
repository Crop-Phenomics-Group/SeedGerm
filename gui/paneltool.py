#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import tkinter as Tkinter

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon


class PanelPolygon:
    def __init__(self):
        self.pts = []
        self.ax_points = []
        self.ax_lines = []
        self.curr_pt = None
        self.last_pt = None
        
    def get_mpl_poly(self):
        pts = np.array(self.pts)
        return Polygon(pts, True)

class PanelTool(Tkinter.Toplevel):
    
    def __init__(self, experiment=None):
        Tkinter.Toplevel.__init__(self)
        
        self.experiment = experiment
        self.n_panels = experiment.panel_n
        
        self.all_polygons = []
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        self.last_down_time = None
        self.pts = []
        self.ax_points = []
        self.ax_lines = []
        self.curr_pt = None
        self.last_pt = None
        
        self.title("Panel tool")
        self.resizable(width=False, height=False)
        self.iconbitmap('.\logo.ico')
        
        self.fig = plt.Figure(figsize=(8., 8.))
        self.ax = self.fig.add_subplot(111)
        self.default_xlim = self.ax.get_xlim()
        self.default_ylim = self.ax.get_ylim()

        self.tk_fig_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.tk_fig_canvas.show()
        self.tk_fig_canvas_widget = self.tk_fig_canvas.get_tk_widget()        

        self.canvas = self.fig.canvas
        
        for tic in self.ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        
        for tic in self.ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

        img_path = os.path.join(
            "../../data/",
            "5022_Test_Rep4_CAM01_ID-01_Date-3-7-2015_14-41.jpg"
        )
        img = imread(img_path)
        self.ax.imshow(img)
        
        self.right_frame = Tkinter.Frame(master=self)
        
        self.pan_btns = []
        self.cur_btn = 0
        self.curr_poly = None

        for i in range(self.n_panels):
            pan_txt = "Panel %d" % (i + 1)
            self.pan_btns.append(
                Tkinter.Button(
                    master=self.right_frame, 
                    text=pan_txt, 
                    command=lambda i=i: self.panel_btns(i))
                )
            self.pan_btns[-1].pack()
        
        self.pan_btns[0].config(relief=Tkinter.SUNKEN) 
        
        self._zoom_init(base_scale=1.1)
        self._pan_init()
        self._labelling_init()
        
        self.tk_fig_canvas_widget.grid(
            in_=self,
            column=1,
            row=1,
            #columnspan=3,
            sticky='news'
        )
        self.right_frame.grid(
            in_=self,
            column=2,
            row=1,
            sticky='news'
        )
        
    def panel_btns(self, idx):
        # if idx == cur_btn:
        #     return
    
        self.pan_btns[self.cur_btn].config(relief=Tkinter.RAISED)
        
        self.pan_btns[idx].config(relief=Tkinter.SUNKEN)    
        self.old_poly = self.all_polygons[self.cur_btn]
        # zp.update_poly_vals(old_poly)    
        
        cur_btn = idx
        curr_poly = self.all_polygons[idx]
        #zp.set_new_poly(curr_poly)
        print(cur_btn)
        
    def _zoom_init(self, base_scale=2.0):
        def zoom(event):
            if event.inaxes != self.ax: 
                return
                
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            x_low = xdata - new_width * (1-relx)
            x_high = xdata + new_width * (relx)
            y_low = ydata - new_height * (1-rely)
            y_high = ydata + new_height * (rely)

            if x_low < self.default_xlim[0]:
                x_low = self.default_xlim[0]
            if x_high > self.default_xlim[1]:
                x_high = self.default_xlim[1]
                
            # y upper and lower reversed as image not plot and coordinate 
            # system is different
            if y_low > self.default_ylim[0]:
                y_low = self.default_ylim[0]
            if y_high < self.default_ylim[1]:
                y_high = self.default_ylim[1]

            self.ax.set_xlim([x_low, x_high])
            self.ax.set_ylim([y_low, y_high])
            self.canvas.draw()

        self.canvas.mpl_connect('scroll_event', zoom)

        return zoom
    
    def _pan_init(self):
        def onPress(event):
            if event.inaxes != self.ax: 
                return
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            self.canvas.draw()

        def onMotion(event):
            if self.press is None: 
                return
            if event.inaxes != self.ax: 
                return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress            
            self.cur_xlim -= dx
            self.cur_ylim -= dy

#            if self.cur_xlim[0] < self.default_xlim[0]:
#                self.cur_xlim[0] = self.default_xlim[0]
#            if self.cur_xlim[1] > self.default_xlim[1]:
#                self.cur_xlim[1] = self.default_xlim[1]
#                
#            # y upper and lower reversed as image not plot and coordinate 
#            # system is different
#            if self.cur_ylim[0] > self.default_ylim[0]:
#                self.cur_ylim[0] = self.default_ylim[0]
#            if self.cur_ylim[1] < self.default_ylim[1]:
#                self.cur_ylim[1] = self.default_ylim[1]            

            self.ax.set_xlim(self.cur_xlim)
            self.ax.set_ylim(self.cur_ylim)

            self.canvas.draw()

        # attach the call back
        self.canvas.mpl_connect('button_press_event',onPress)
        self.canvas.mpl_connect('button_release_event',onRelease)
        self.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion
    
    def _labelling_init(self):
        def onPress(event):
            self.last_down_time = time.time()
            
        def onRelease(event):
            xdata, ydata = event.xdata, event.ydata
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()
            
            if event.button == 3:
                if self.ax.lines:
                    line = self.ax_lines[-1][0]
                    del_idx = self.ax.lines.index(line)
                    del self.ax_lines[-1]
                    del self.ax.lines[del_idx]
                    
                if self.ax.collections:
                    pt = self.ax_points[-1]
                    del_idx = self.ax.collections.index(pt)
                    del self.ax_points[-1]
                    del self.ax.collections[del_idx]
                    
                if self.pts:
                    del self.pts[-1]
                
                if self.pts:
                    self.last_pt = self.pts[-1]
                else:
                    self.last_pt = None
                
                self.canvas.draw()
                return
                
            if xdata is None and ydata is None:
                return          
                
            if time.time() - self.last_down_time > 0.12:
                self.last_down_time = None
                return
            
            self.last_down_time = None            
            
            self.ax_points.append(self.ax.scatter(xdata, ydata))
            self.ax.set_xlim(self.cur_xlim)
            self.ax.set_ylim(self.cur_ylim)
            self.canvas.draw()

            self.curr_pt = (xdata, ydata)
            self.pts.append(self.curr_pt)
                        
            if len(self.pts) == 1:
                self.last_pt = self.curr_pt
                return
        
            xs = [self.last_pt[0], self.curr_pt[0]]
            ys = [self.last_pt[1], self.curr_pt[1]]
            
            self.ax_lines.append(self.ax.plot(xs, ys, 'r'))
            
            self.last_pt = self.curr_pt
            
            self.canvas.draw()

        # attach the call back
        self.canvas.mpl_connect('button_press_event',onPress)
        self.canvas.mpl_connect('button_release_event',onRelease)
        
