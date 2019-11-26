#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tkinter as Tkinter
from tkinter import filedialog
from tkinter import messagebox
from tinydb import where
from helper.functions import slugify, get_images_from_dir
from helper.experiment import Experiment

pj = os.path.join


class AddExperiment(Tkinter.Toplevel):
    def __init__(self, app):
        Tkinter.Toplevel.__init__(self)

        self.app = app
        self.db = Experiment.database

        self.title("Add experiment")
        self.resizable(width=False, height=False)
        self.iconbitmap('.\logo.ico')

        self.name_label = Tkinter.Label(
            master=self,
            text="Experiment name: ",
            anchor=Tkinter.W
        )
        self.name_entry = Tkinter.Entry(master=self)

        self.dir_label = Tkinter.Label(
            master=self,
            text="Image directory: ",
            anchor=Tkinter.W
        )
        self.dir_entry = Tkinter.Entry(master=self)
        self.dir_button = Tkinter.Button(
            master=self,
            text="...",
            padx=5,
            pady=0,
            command=self._get_exp_dir
        )

        self.species_label = Tkinter.Label(
            master=self,
            text="Species: ",
            anchor=Tkinter.W
        )
        self.species_var = Tkinter.StringVar(self)
        self.species = self.app.core.species_classes.keys()
        self.species_var.set(list(self.species)[0])  # default value
        self.species_options = Tkinter.OptionMenu(
            self,
            self.species_var,
            *self.species
        )

        self.bg_rm_label = Tkinter.Label(
            master=self,
            text="BG remover: ",
            anchor=Tkinter.W
        )
        self.removers_var = Tkinter.StringVar(self)
        self.removers = ['GMM', 'SGD', 'UNet']
        self.removers_var.set(self.removers[0])  # default value
        self.bg_rm_options = Tkinter.OptionMenu(
            self,
            self.removers_var,
            *self.removers
        )

        self.panel_num_label = Tkinter.Label(
            master=self,
            text="Number of panels: ",
            anchor=Tkinter.W
        )
        self.panel_num_entry = Tkinter.Entry(master=self)

        self.seeds_per_panel_row_label = Tkinter.Label(
            master=self,
            text="Rows: ",
            anchor=Tkinter.W
        )
        self.seeds_per_panel_row_entry = Tkinter.Entry(master=self)

        self.seeds_per_panel_col_label = Tkinter.Label(
            master=self,
            text="Cols: ",
            anchor=Tkinter.W
        )
        self.seeds_per_panel_col_entry = Tkinter.Entry(master=self)

        self.start_image_label = Tkinter.Label(
            master=self,
            text="Start image: ",
            anchor=Tkinter.W
        )
        self.start_image_entry = Tkinter.Entry(master=self)
        self.end_image_label = Tkinter.Label(
            master=self,
            text="End image: ",
            anchor=Tkinter.W
        )
        self.end_image_entry = Tkinter.Entry(master=self)

        self.use_colour_label = Tkinter.Label(
            master=self,
            text="Use Colour: ",
            anchor=Tkinter.W
        )
        self.use_colour = Tkinter.IntVar(self)
        self.use_colour_box = Tkinter.Checkbutton(master=self, variable=self.use_colour)

        self.cancel_button = Tkinter.Button(
            master=self,
            text="Cancel",
            command=self._cancel,
            padx=16
        )

        self.add_button = Tkinter.Button(
            master=self,
            text="Add",
            command=self._add,
        )

        self.name_label.grid(
            in_=self,
            column=1,
            row=1,
            sticky='news'
        )

        self.name_entry.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=1,
            sticky='ew'
        )

        self.dir_label.grid(
            in_=self,
            column=1,
            row=2,
            sticky='news'
        )

        self.dir_entry.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=2,
            sticky='ew'
        )
        self.dir_button.grid(
            in_=self,
            column=3,
            row=2,
            sticky='ew'
        )

        self.species_label.grid(
            in_=self,
            column=1,
            row=3,
            sticky='news'
        )

        self.species_options.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=3,
            sticky='ew'
        )

        self.panel_num_label.grid(
            in_=self,
            column=1,
            row=4,
            sticky='news'
        )

        self.panel_num_entry.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=4,
            sticky='new'
        )

        self.seeds_per_panel_row_label.grid(
            in_=self,
            column=1,
            row=5,
            sticky='news'
        )
        self.seeds_per_panel_row_entry.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=5,
            sticky='ew'
        )

        self.seeds_per_panel_col_label.grid(
            in_=self,
            column=3,
            row=5,
            sticky='news'
        )
        self.seeds_per_panel_col_entry.grid(
            in_=self,
            column=4,
            columnspan=1,
            row=5,
            sticky='ew'
        )

        self.start_image_label.grid(
            in_=self,
            column=1,
            row=6,
            sticky='news',
        )
        self.start_image_entry.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=6,
            sticky='ew'
        )
        self.end_image_label.grid(
            in_=self,
            column=3,
            row=6,
            sticky='news',
        )
        self.end_image_entry.grid(
            in_=self,
            column=4,
            columnspan=1,
            row=6,
            sticky='ew'
        )

        self.bg_rm_label.grid(
            in_=self,
            column=1,
            row=9,
            sticky='news'
        )

        self.bg_rm_options.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=9,
            sticky='ew'
        )

        self.use_colour_label.grid(
            in_=self,
            column=1,
            row=12,
            sticky='news'
        )

        self.use_colour_box.grid(
            in_=self,
            column=2,
            columnspan=1,
            row=12,
            sticky='ew'
        )

        self.cancel_button.grid(
            in_=self,
            column=2,
            row=13,
            sticky='e'
        )

        self.add_button.grid(
            in_=self,
            column=3,
            columnspan=1,
            row=13,
            sticky='ew'
        )

        for i in range(1, 14):
            self.grid_rowconfigure(i, pad=5)
        self.grid_columnconfigure(2, minsize=200)

    def _get_exp_dir(self):
        self.dir_entry.delete(0, 'end')
        self.dir_entry.insert(0, filedialog.askdirectory())
        self.dir_entry.xview_moveto(1)
        self.lift()

    def _warning_conditions(self, warn_conds):
        for cond, msg in warn_conds:
            if cond:
                messagebox.showwarning(
                    "Add experiment",
                    msg
                )
                return False
                # if we reach here then everything was okay
        return True

    def _is_int(self, n):
        try:
            int(n)
            return True
        except:
            return False

    def _add(self):
        name = self.name_entry.get()
        dir_ = self.dir_entry.get()
        panel_n = self.panel_num_entry.get()
        seeds_row_n = self.seeds_per_panel_row_entry.get()
        seeds_col_n = self.seeds_per_panel_col_entry.get()
        species = self.species_var.get()
        bg_remover = self.removers_var.get()
        start_img = self.start_image_entry.get()
        end_img = self.end_image_entry.get()
        if len(end_img) < 1:
            end_img = '-1'
        if len(start_img) < 1:
            start_img = '-1'
        use_colour = self.use_colour.get()
        pre_conditions = [
            (len(name) < 1, "Experiment name is too short"),
            (len(dir_) < 1, "Need to enter a directory"),
            (len(panel_n) < 1, "Need to enter the number of panels"),
            (len(seeds_col_n) < 1, "Need to enter the number of seeds per column"),
            (len(seeds_row_n) < 1, "Need to enter the number of seeds per row"),
            (len(start_img) < 1, "Need to enter start image index"),
            (len(end_img) < 1, "Need to enter end image index")
        ]

        if not self._warning_conditions(pre_conditions):
            self.app.lift()
            return

        post_conditions = [
            (self.db.get(where("name") == name) is not None,
             "Experiment with this name already exists"),
            (not os.path.exists(dir_), "Cannot find experiment directory"),
            (len(os.listdir(dir_)) < 1,
             "Directory does not contain any files"),
            (not self._is_int(panel_n), "Panel value needs to be an integer"),
            (not self._is_int(seeds_row_n), "Seeds row value needs to be an integer"),
            (not self._is_int(seeds_col_n), "Seeds column value needs to be an integer"),
            (not self._is_int(start_img), "Start image index needs to be an integer"),
            (not self._is_int(end_img), "End image index needs to be an integer")
        ]

        if not self._warning_conditions(post_conditions):
            self.app.lift()
            return

        panel_n = int(panel_n)
        seeds_col_n = int(seeds_col_n)
        seeds_row_n = int(seeds_row_n)
        start_img = int(start_img)
        end_img = int(end_img)

        imgs = get_images_from_dir(dir_)
        if end_img == -1:
            end_img = len(imgs)
        if start_img == -1:
            start_img = 0
        if not imgs:
            messagebox.showwarning("Warning",
                                     "Cannot find images in directory: {}".format(dir_))
            self.app.lift()
            return

        exp_path = "./data/experiments/%s" % (slugify(name))
        exp = Experiment(name=name,
                         img_path=dir_,
                         panel_n=panel_n,
                         seeds_col_n=seeds_col_n,
                         seeds_row_n=seeds_row_n,
                         species=species,
                         exp_path=exp_path,
                         start_img=start_img,
                         end_img=end_img,
                         bg_remover=bg_remover,
                         use_colour=use_colour,
                         panel_labelled=False,
                         _yuv_ranges_set=False,
                         _status="")
        exp.create_directories()
        exp.insert_into_database()
        self.app.experiments.append(exp)

        self.destroy()
        self.app._populate_experiment_table()
        self.app.lift()

    def _cancel(self):
        self.destroy()
