#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:34:01 2023

Create data set from known corrupt images and known good images. Make sure the
sampling is balanced and potentially unusual cases are present. However, note
that the images should be completely good or completely corrupt in order to
get accurate estimates of model accuracy and a good training result.
"""
__author__ = "Manuel"
__date__ = "Tue Nov 21 09:34:01 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
import os
import numpy as np
import rasterio as rio
from rasterio import windows
from itertools import product

#-----------------------------------------------------------------------------|
#> Create data sets (tiles from valid and from corrupt images)
corrupt_data = {
    "trn" : [
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Viola hirta_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acer campestre_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aconitum lycoctonum_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acrocephalus palustris_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acrocephalus scirpaceus_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aglais io_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aglais urticae_zh.tif"
        ],
    "tst" : [
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Strix aluco_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Sylvia atricapilla_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Sylvia borin_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Sympetrum sanguineum_zh.tif"
        ]
    }

good_data = {
    "trn" :[
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Natrix natrix_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Abramis brama_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acicula lineata_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acronicta megacephala_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Acupalpus flavicollis_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aegopinella nitens_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aegopinella pura_zh.tif"
        ],
    "tst" : [
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Aglia tau_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Agonum viduum_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Sphecodes puncticeps_zh.tif",
        "L:/brunp/shared/for_Manuel/NSDM_maps/rcp45_2075/Spirodela polyrhiza_zh.tif"
        ]
    }

output_filename = "{0}_tile_{1}-{2}.tif"

#-----------------------------------------------------------------------------|
#> Set directories and name patterns
# Main directory for image tiles (training and test data)
dir_main = "L:/poppman/cnn256"

out_main = dir_main + "/{}"
dir_good = dir_main + "/{}/0"
dir_bad = dir_main + "/{}/1"
tile_size = 128

#-----------------------------------------------------------------------------|
#> Functions to extract image tiles
def get_tiles(ds, width = tile_size, height = tile_size):
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(
        col_off = 0, row_off = 0, width = ncols, height = nrows
        )
    
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off = col_off,
            row_off = row_off,
            width = width,
            height = height
            ).intersection(big_window)
        
        transform = windows.transform(window, ds.transform)
        yield window, transform

#-----------------------------------------------------------------------------|
#> Create data sets
for d, s in zip(["trn", "trn", "tst", "tst"], ["0", "1", "0", "1"]):
    os.makedirs(os.path.join(dir_main, d, s), exist_ok = True)

for ds in ["trn", "tst"]:
    good, bad = good_data[ds], corrupt_data[ds]
    out_good = [dir_good.format(ds)] * len(good)
    out_bad = [dir_bad.format(ds)] * len(bad)
    data = good + bad
    paths = out_good + out_bad
    
    for in_path, out_path in zip(data, paths):
        with rio.open(in_path) as inds:
            tile_width, tile_height = tile_size, tile_size
        
            meta = inds.meta.copy()
        
            for window, transform in get_tiles(inds):
                if window.width == tile_size and window.height == tile_size:
                    meta["transform"] = transform
                    meta["width"], meta["height"] = window.width, window.height
                    outpath = os.path.join(
                        out_path,
                        output_filename.format(
                            os.path.splitext(os.path.basename(in_path))[0],
                            int(window.col_off),
                            int(window.row_off)
                            )
                        )
                    
                    ods = inds.read(window = window)
                    if np.sum(ods) < 4177920:
                        with rio.open(outpath, "w", **meta) as outds:
                            outds.write(ods)
