#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:16:35 2023
"""
__author__ = "Manuel"
__date__ = "Mon Dec  4 11:16:35 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
# Append current file directory to import from other Python scripts
import os, sys

dir_py = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_py)

from model import SimpleCNN
from apply_model import ImageChecker

#-----------------------------------------------------------------------------|
#> Example
# Instantiate ImageChecker
im = ImageChecker("L:/poppman/cnn/trained_model_sd.pt")

# Run check on a directory
probabilities = im.check_folder("L:/poppman/cnn/Maps/NOTOK")
