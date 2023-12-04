#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coded this in the bus on my smartphone, probably full of typos.
Must check tomorrow...
This is supposed to be a commandline interface to run the checks e.g. from within R.
"""
__author__ = "Manuel"
__date__ = "Mon Dec  4 21:12:52 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
# Append current file directory to import from other Python scripts
import os, sys, argparse

dir_py = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_py)

from model import SimpleCNN
from apply_model import ImageChecker

#-----------------------------------------------------------------------------|
#> Get arguments
def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("folder", help = "Folder to check", type = str)

    # Optional arguments
    parser.add_argument("-t", "--threshold",
                        help = "Probability threshold.",
                            type = float, default = None)
    parser.add_argument("-o", "--outfile", help = "Output .csv filepath.",
                        type = "str", default = None)
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

# debug mode
if False:
    saved_args = \
    "C:\\Users\\Manuel\\Nextcloud\\Masterarbeit\\py3\\vrs\\train_settings.pkl"
    with open(saved_args, "rb") as f:
        args = pickle.load(f)
    args.wd = "home"

folder = args.folder
threshold = args.threshold
outfile = args.outfile

#-----------------------------------------------------------------------------|
#> Run
# Instantiate ImageChecker
dir_main = os.path.dirname(dir_py)

im = ImageChecker(os.path.join(dir_main, "mod", "trained_model_sd.pt"))
probabilities = im.check_folder(
    folder,
    confidence_threshold = threshold,
    output_file = outfile
    )

print(probabilities)
