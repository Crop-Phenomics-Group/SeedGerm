#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" germapp.py - Main script for runnning the germination scoring application.

Starts the core thread for heavy computational processing and the GUI for user
interaction.

hello world
"""

import matplotlib
matplotlib.use('Agg')
import sys
import time
import os

if not os.path.exists("./data/"):
    os.makedirs("./data/")

from gui.application import Application
from brain.core import Core


def main():

    # Redirect stdout/stderr to file
    with open("./data/run.log", "a+") as log_fh:
        #sys.stdout = log_fh
        #sys.stderr = log_fh
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        seper = "-" * 19
        print("\n\n%s\n%s\n%s\n" % (seper, now, seper))

        # Create the GUI application and the brain core.
        app = Application()
        core = Core()

        # Ensure the app and core have a handle of each other.
        app.set_core(core)
        core.set_gui(app)

        # Start the brain core running.
        core.start()

        #core.initialise_gui()

        # Run the application.
        app.mainloop()

if __name__ == "__main__":
    main()
