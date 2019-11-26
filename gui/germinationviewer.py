# -*- coding: utf-8 -*-

import tkinter as Tkinter

from gui.maskviewer import MaskViewer


class GerminationViewer(MaskViewer):
    
    def __init__(self, exp):
        Tkinter.Toplevel.__init__(self)        
        MaskViewer.__init__(self, exp)
        
        self.title("Lal")
        self.iconbitmap('.\logo.ico')