# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 10:20:38 2017

@author: lecornt
"""

import glob
import os
import datetime
import compileall
import shutil


pj = os.path.join

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

now = datetime.datetime.now()
NOW_STR = now.strftime("%Y%m%d_%H%M%S")
RELEASE_DIR = "Release/"

BUILD_DIR = pj(RELEASE_DIR, "build_{}".format(NOW_STR))
os.makedirs(BUILD_DIR)

DIRS_CREATE = [
    "brain",
    "data",
    "data/experiments",
    "gui",
    "helper"
]

for d in DIRS_CREATE:
    os.makedirs(pj(BUILD_DIR, d))

FILES_COPY = [
    "gui/logo.gif",
]

for f in FILES_COPY:
    shutil.copyfile(f, pj(BUILD_DIR, f))

FILES_CREATE = [
    "data/run.log",
    "data/experiment_list.json",
]

for f in FILES_CREATE:
    touch(pj(BUILD_DIR, f))

PYC_FILE_DIRS = [
    ".",
    "brain",
    "gui",
    "helper",
]

for fdir in PYC_FILE_DIRS:
    compileall.compile_dir(fdir, maxlevels=0, force=True)

    pyc_files = glob.glob(pj(fdir, "*.pyc"))
    for pycf in pyc_files:
        shutil.copy(pycf, pj(BUILD_DIR, pycf))

os.remove(pj(BUILD_DIR, "build_app.pyc"))
