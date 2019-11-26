# -*- coding: utf-8 -*-

import shutil
import os
import tkinter as Tkinter
from tkinter import ttk
import collections
from tkinter import messagebox
from tkinter import filedialog
import glob
import sys
from matplotlib import pyplot as plt
import numpy as np
import json
import _thread
from helper.functions import get_images_from_dir
from gui.paneltool import PanelTool
from gui.addexperiment import AddExperiment
from gui.imageviewer import ImageViewer
from gui.viewresults import ViewResults
from gui.yuvranges import YUVRanges, YUVPanelRanges
from gui.aboutwindow import AboutWindow
from gui.maskviewer import MaskViewer
from gui.germinationviewer import GerminationViewer
from helper.experiment import Experiment


class Application(Tkinter.Tk):

    def __init__(self):
        Tkinter.Tk.__init__(self)
        self.db = Experiment.database

        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.title("SeedGerm - Beta Release")
        self.resizable(width=False, height=False)
        # self.iconbitmap(sys._MEIPASS + '.\logo.ico')
        self.iconbitmap('.\logo.ico')
        self.exp_treeview_ids = {}
        self._experiments = None

        self.BLUE = "#a0a0ff"
        self.GREEN = "#80ff80"
        self.RED = "#ff8080"

        # Define and build main application menu.
        self.menu = Tkinter.Menu(self)
        menu_config = collections.OrderedDict([
            ('File', [
                ('command', 'Add experiment', self._add_experiment),
                ('separator', None, None),
                ('command', 'Quit', self._quit)
            ]),
            ('Help', [
                ('command', 'Documentation', self._documentation),
                ('separator', None, None),
                ('command', 'About', self._about)
            ])
        ])
        self._build_menu(self.menu, "main_menu_item", menu_config)
        self.configure(menu=self.menu)

        self.tree_columns = [
            "Name",
            "Species",
            "# panels",
            "# images",
            "Status"
        ]
        self.treeview = ttk.Treeview(
            master=self,
            columns=self.tree_columns,
            show="headings",
            height=15,
            selectmode="browse"
        )

        for col in self.tree_columns:
            self.treeview.heading(col, text=col.title())
            self.treeview.column(col, anchor="center")

        self._populate_experiment_table()

        self.treeview.column("Name", width=250)
        self.treeview.column("Species", width=100)
        self.treeview.column("# panels", width=100)
        self.treeview.column("# images", width=100)
        self.treeview.column("Status", width=150)

        self.vsb_1 = ttk.Scrollbar(
            master=self,
            orient="vertical",
            command=self.treeview.yview
        )
        self.hsb_1 = ttk.Scrollbar(
            master=self,
            orient="horizontal",
            command=self.treeview.xview
        )

        self.treeview.configure(
            yscrollcommand=self.vsb_1.set,
            xscrollcommand=self.hsb_1.set
        )

        self.status_string = Tkinter.StringVar()
        self.info_label = Tkinter.Label(
            master=self,
            textvariable=self.status_string,
            height=2,
            justify=Tkinter.LEFT,
            anchor=Tkinter.W,
        )

        self.treeview.grid(
            in_ = self,
            column = 1,
            row = 1,
            sticky='news'
        )
        self.vsb_1.grid(
            in_ = self,
            column = 2,
            row = 1,
            sticky = 'news'
        )
        self.hsb_1.grid(
            in_ = self,
            column = 1,
            #columnspan=2,
            row = 2,
            sticky = 'news'
        )
        self.info_label.grid(
            in_ = self,
            column = 1,
            columnspan=2,
            row = 4,
            sticky = 'news'
        )

        # bind the table menu to right-click on the treeview
#        self.treeview.bind("<Button-3>", self._table_menu_right_click)
        if sys.platform == 'darwin':
            self.treeview.bind("<Button-2>", self._table_menu_right_click)
        else:
            self.treeview.bind("<Button-3>", self._table_menu_right_click)

        self.treeview.bind("<Double-1>", self._treeview_dbl)

        # Define and build the right-click Table Menu
        self.table_menu = Tkinter.Menu(self, tearoff=0)
        table_menu_config = [
            ('command', 'Set YUV ranges', self._set_yuv_ranges),
            ('separator', None, None),
            ('command', 'Set YUV_panel ranges', self._set_yuv_panel_ranges),
            ('separator', None, None),
            ('command', 'Process images', self._process_images),
            ('separator', None, None),
            ('command', 'View results', self._view_results),
            ('command', 'View images', self._view_images),
            ('command', 'View seed masks', self._view_seed_masks),
#            ('command', 'View germination', self._view_algo_desc),
            ('separator', None, None),
            ('command', 'Save results', self._save_results),
            ('command', 'Save masks', self._save_masks),
            ('separator', None, None),
#            ('command', 'Edit', self._edit_exp),
            ('command', 'Reset', self._reset_exp),
            ('separator', None, None),
            ('command', 'Delete', self._del_exp),
            ('separator', None, None),
            ('command', 'Cancel', None),
        ]
        self._menu_commands(self.table_menu, table_menu_config)

        self.db_updated = False
        self._refresh_exp()


    @property
    def experiments(self):
        #when we init the app, we want to constuct the experiment objects once.
        #these can be extracted back to maps later when we update.
        if self._experiments is None:
            self._experiments = [Experiment(**x) for x in self.db.all()]

        return self._experiments

    def _refresh_exp(self):
        if Experiment.updated:
            print("refreshing GUI")
            self._populate_experiment_table()
            Experiment.updated = False
        self.after(100, self._refresh_exp)

    def _populate_experiment_table(self):
        self.treeview.delete(*self.treeview.get_children())

        if not self.experiments:
            return

        # Populate the table with saved experiments.
        for exp in self.experiments:

            n_imgs = -1
            if not os.path.exists(exp.img_path):
                print("Can't find this path...",  exp.img_path)
                messagebox.showwarning("Experiment problem",
                    "Can't find directory {} for experiment {}.".format(exp.img_path, exp.name))
                exp.status = "Error"
            else:
                imgs = get_images_from_dir(exp.img_path)
                n_imgs = len(imgs)
                if not n_imgs:
                    messagebox.showwarning("Experiment problem",
                                             "Problem with images for experiment {}.".format(exp.name))
                    exp.status = "Error"

            values = [
                exp.name,
                exp.species,
                exp.panel_n,
                n_imgs,
                exp.status
            ]
            e_item = self.treeview.insert('', 'end', values=values)
            self.exp_treeview_ids[exp.eid] = e_item

    def _menu_commands(self, menu_obj, commands):
            for t, l, fn in commands:
                if t == "command":
                    menu_obj.add_command(label=l, command=fn)
                elif t == "separator":
                    menu_obj.add_separator()
                else:
                    print("unknown menu type")

    def _build_menu(self, menu, menu_item_name, menu_config):
        for idx, (key, value) in enumerate(menu_config.items()):
            # Create menu item key and object, and set as class variable.
            menu_item_key = "%s_%d" % (menu_item_name, idx)
            new_menu_obj = Tkinter.Menu(menu, tearoff=0)
            self.__dict__[menu_item_key] = new_menu_obj

            # Add a cascade with key as the label.
            menu.add_cascade(label=key, menu=new_menu_obj)
            self._menu_commands(new_menu_obj, value)

    def _table_menu_right_click(self, event):
        """ Get the item that has been right-clicked and display the
        relevant menu.
        """
        if len(self.treeview.get_children()) == 0:
            # we don't want to do anything if no files are imported
            return
        # post the event to the table menu
        item_id = self.treeview.identify_row(event.y)
        # user clicked on header and not an element
        if not len(item_id):
            return
        self.treeview.selection_set(item_id)
        self.table_menu.post(event.x_root, event.y_root)

    def set_core(self, core):
        self.core = core

    def _get_exp(self):
        item = self.treeview.selection()
        index = self.treeview.index(item)
        return self.experiments[index]

    def _set_yuv_ranges(self):
        exp = self._get_exp()
        self.yuv_ranges = YUVRanges(self, exp)

    def _set_yuv_panel_ranges(self):
        exp = self._get_exp()
        self.yuv_panel_ranges = YUVPanelRanges(self, exp)

    def _process_images(self):
        exp = self._get_exp()



        # If the user hasn't labelled panels for this experiment we can't
        # process the experiment
        if not exp.yuv_ranges_set:
            messagebox.showwarning(
                "",
                "YUV ranges have not been set for this experiment."
            )
            return

        self.info_label.config(background=self.GREEN)

        exp.status = "Running"
        self.core.start_processor(exp)

    def _save_results(self):
        exp = self._get_exp()

        exp_results_dir = os.path.join(exp.exp_path, "results")

        if len(glob.glob(os.path.join(exp_results_dir, "*.csv"))) < 3:
            messagebox.showwarning(
                "",
                "Need results to save, process images first."
            )
            return

        sev_dir = filedialog.askdirectory()
        if not len(sev_dir):
            return

        self.core.zip_results(exp, sev_dir)

    def _treeview_dbl(self, event):
        self._view_results()

    def _view_results(self):
        exp = self._get_exp()
        if not os.path.exists(exp.get_results_graph()):
            print("Results not available.")
            return
        self.vr = ViewResults(exp)

    def _view_images(self):
        exp = self._get_exp()
        imgs = get_images_from_dir(exp.img_path)
        self.image_viewer = ImageViewer(exp.img_path, imgs[exp.start_img:exp.end_img])

    def _view_seed_masks(self):
        self.mask_viewer = MaskViewer(self._get_exp())

    def _view_algo_desc(self):
        self.germ_viewer = GerminationViewer(self._get_exp())

#    def _label_panels(self):
#        tkMessageBox.showinfo("", "Panel tool not supported.")
#        exp = self._get_exp()
#        print exp
#        self.panel_tool = PanelTool(exp)

#    def _edit_exp(self):
#        print "edit exp"

    #this destroys the experiment path of data. but keeps the experiment.
    # will need to remake all the YUVs everything.
    def _reset_exp(self):
        exp = self._get_exp()
        yes_remove = messagebox.askyesno(
            "",
            "Are you sure you want to reset this experiment?"
        )

        if yes_remove:
            exp.status = "Reset"
            exp.yuv_ranges_set = False
            exp.reset()

    @staticmethod
    def _save_masks_function(exp, sev_dir):
        scatter_f = os.path.join(exp.get_results_dir(), "panel_seed_idxs.json")
        if not os.path.exists(scatter_f):
            return

        with open(scatter_f) as fh:
            panel_seed_idxs = json.load(fh)
            exp_masks_dir_frame = os.path.join(exp.get_masks_dir(), "frame_%d")

            for idx in range(exp.start_img, len(get_images_from_dir(exp.img_path))):
                masks = np.load(exp_masks_dir_frame % idx)
                for curr_panel, mask in enumerate(masks):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.imshow(mask)

                    m,n = mask.shape
                    ys, xs = [], []
                    for i, y, x in panel_seed_idxs[str(curr_panel)]:
                        if i == -1:
                            continue
                        if idx >= i:
                            ys.append(y)
                            xs.append(x)
                    if len(ys) and len(xs):
                        ax.scatter(xs, ys, c='r', s=40)
                        ax.set_xlim(0, n)
                        ax.set_ylim(m, 0)

                    fig.canvas.draw()
                    fig.savefig(os.path.join(sev_dir, 'panel_%d_frame_%d.jpg' % (curr_panel, idx)))
                    plt.close(fig)

        messagebox.showinfo(
            "",
            "Masks Finished Saving."
        )

    def _save_masks(self):
        exp = self._get_exp()
        exp_results_dir = os.path.join(exp.exp_path, "results")
        if len(glob.glob(os.path.join(exp_results_dir, "*.csv"))) < 3:
            messagebox.showwarning(
                "",
                "Need results to save, process images first."
            )
            return

        sev_dir = filedialog.askdirectory()
        if not len(sev_dir):
            return

        _thread.start_new_thread(Application._save_masks_function, (exp, sev_dir))


    def _del_exp(self):
        exp = self._get_exp()
        yes_remove = messagebox.askyesno(
            "",
            "Are you sure you want to delete this experiment?"
        )

        if yes_remove:
            shutil.rmtree(exp.exp_path)
            self.db.remove(eids=[exp.eid])
            Experiment.updated = True
            self._experiments.remove(exp)
            self._populate_experiment_table()

    def _add_experiment(self):
        self.add_experiment = AddExperiment(self)

    def _open_panel_tool(self):
        self.panel_tool = PanelTool()

    def _documentation(self):
        print("_documentation method")

    def _about(self):
        self.about_window = AboutWindow(self)

    def _quit(self):
        self.core.die()
        self.quit()
        self.destroy()
