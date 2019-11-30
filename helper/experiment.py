import os
import shutil

from tinydb import TinyDB


class Experiment(object):
    db_path = './data/experiment_list.json'
    sub_directories = ['masks', 'gzdata', 'results', 'images']

    database = TinyDB(db_path)
    updated = False

    def __init__(self, name=None, exp_path=None, img_path=None, panel_n=None, seeds_col_n=None, seeds_row_n=None,
                 species="Brassica", start_img=0, end_img=None, bg_remover="GMM", panel_labelled=False,
                 _yuv_ranges_set=False,
                 _eid=None, _status="", use_colour=False, use_delta=False):
        self.name = name
        self.img_path = img_path
        self.panel_n = panel_n
        self.seeds_col_n = seeds_col_n
        self.seeds_row_n = seeds_row_n
        self.species = species
        self.start_img = start_img
        self.end_img = end_img
        self.bg_remover = bg_remover
        self.panel_labelled = panel_labelled
        self._yuv_ranges_set = _yuv_ranges_set
        self._eid = _eid
        self._status = _status
        self.use_colour = use_colour
        self.use_delta = use_delta

        self.exp_path = exp_path  # "./data/experiments/%s" % (slugify(self.name))

        if self.exp_path is not None:
            self.create_directories()

    @property
    def seeds_n(self):
        return self.seeds_col_n*self.seeds_row_n

    @property
    def status(self):
        return self._status

    @property
    def yuv_ranges_set(self):
        return self._yuv_ranges_set

    @property
    def eid(self):
        return self._eid

    # for some reason tinyDB is indexed from 1....

    @yuv_ranges_set.setter
    def yuv_ranges_set(self, value):
        if self._yuv_ranges_set is not value:
            self._yuv_ranges_set = value
            self.database.update(vars(self), eids=[self.eid])
            Experiment.updated = True

    @status.setter
    def status(self, value):
        if self._status is not value:
            self._status = value
            self.database.update(vars(self), eids=[self.eid])
            Experiment.updated = True

    @eid.setter
    def eid(self, value):
        if self._eid is not value:
            self._eid = value
            self.database.update(vars(self), eids=[self.eid])
            # don't need to update GUI

    def get_results_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[2])

    def get_images_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[3])

    def get_masks_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[0])

    def get_results_graph(self):
        return os.path.join(self.get_results_dir(), "results.jpg")

    def create_directories(self):
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
            for sub_dir in Experiment.sub_directories:
                os.makedirs(os.path.join(self.exp_path, sub_dir))

    def reset(self):
        # delete the tree and rebuild the dirs.
        if os.path.exists(self.exp_path):
            shutil.rmtree(self.exp_path)
            self.create_directories()

    def insert_into_database(self):
        self.eid = Experiment.database.insert(vars(self))
