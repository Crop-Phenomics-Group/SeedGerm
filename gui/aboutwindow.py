# -*- coding: utf-8 -*-

import tkinter as Tkinter
import sys


about_text = """
SeedGerm - Beta release

Property of: Crop Phenomics Group, Earlham Institute, Norwich Research Park, UK

Authors:  Joshua Colmer, Aaron Bostrom, Ji Zhou, Danny Websdale, Thomas Le Cornu, Joshua Ball
"""


class AboutWindow(Tkinter.Toplevel):
    

    
    def __init__(self, exp):
        Tkinter.Toplevel.__init__(self)
        self.title("About SeedGerm")
        self.resizable(width=False, height=False)
        self.wm_geometry("420x250")
        self.iconbitmap('.\logo.ico')

        photo = Tkinter.PhotoImage(file=("./icon.gif"))
        w = Tkinter.Label(self, image=photo)
        w.photo = photo
        w.pack()

        self.msg = Tkinter.Message(self, text=about_text)
        self.msg.pack()
        
      
        