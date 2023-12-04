#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:00:47 2023

Define ImageChecker class which is used to apply a trained model to predict
whether an image or a set of images contains typical write error patterns
associated with OOM issue.
"""
__author__ = "Manuel"
__date__ = "Fri Dec  1 16:00:47 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
import os, sys, torch
from torch import device, cuda
from torchvision import transforms
import numpy as np
import rasterio as rio
from rasterio import windows
from glob import glob
from PIL import Image
import warnings
from model import SimpleCNN

# Append current file directory to import from other Python scripts
dir_py = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_py)

from model import SimpleCNN

#-----------------------------------------------------------------------------|
#> Classes
class ImageChecker:
    def __init__(self, model_file):
        # Load the saved model
        dev = device("cuda" if cuda.is_available() else "cpu")
        save_pth = torch.load(model_file, map_location = dev)
        self.model = SimpleCNN()
        self.model.load_state_dict(save_pth)
        self.to_tensor = transforms.ToTensor()
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Image size settings
        self.h = 128
        self.w = 128
    
    def _create_window(self, r, c):
        '''
        Create a window to read a sub-image from a raster file.

        Parameters
        ----------
        r : int
            Row offset.
        c : int
            Column offset.

        Returns
        -------
        wdw : windows.Window
            Window to read.

        '''
        wdw = windows.Window(
            col_off = c, row_off = r, width = self.w, height = self.h
            )
        
        return wdw
    
    def _arrange_grid(self, nrow, ncol):
        '''
        Create a diagonal line of (as well as possible) evenly spaced windows
        on a raster image.

        Parameters
        ----------
        nrow : int
            Number or rows of the raster image.
        ncol : int
            Number of columns of the raster image.

        Returns
        -------
        windows : list
            List of windows.Window.

        '''
        n_x = nrow // self.w
        n_y = nrow // self.h
        
        remainder_x = nrow % self.w
        remainder_y = nrow % self.h
        
        spacing_x = remainder_x // n_x
        spacing_y = remainder_y // n_y
        n_windows = min(n_x, n_y)
        
        step_size_x = max(1, n_x // n_windows)
        indices_x = list(range(n_x))[::step_size_x][:n_windows]
        step_size_y = max(1, n_y // n_windows)
        indices_y = list(range(n_y))[::step_size_y][:n_windows]
        
        left = [i * self.w + i * spacing_x for i in indices_x]
        bottom = [i * self.h + i * spacing_y for i in indices_y]
        
        windows = [self._create_window(r, c) for r, c in zip(left, bottom)]
        
        return windows
    
    def check_image(
            self,
            path,
            output = "probability",
            confidence_threshold = None
            ):
        '''
        Check an image using

        Parameters
        ----------
        path : str
            Image path.
        output : str, optional
            One in "binary", "probability", or "tensor". For binary, the output
            is 0 or 1, indicating whether the value is below or above the
            threshold set via the "confidence_threshold" argument (which must
            be set to use this mode). For "probability", the method returns the
            maximum probability value (across all sampled sub-images). For
            "tensor", the entire propability tensor is returned.
            The default is "probability".
        confidence_threshold : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            See parameter "output".

        '''
        tiles = []
        
        with rio.open(path) as inds:
            nrow, ncol = inds.shape
            windows = self._arrange_grid(nrow, ncol)
            
            for wdw in windows:
                win = inds.read(window = wdw)
                im = Image.fromarray(np.squeeze(win, axis = 0))
                
                # Training was done on INT8U tiles, which is, thus, what the
                # model expects
                im = im.convert(mode = "L", colors = 256)
                
                if len(set(im.getextrema())) > 1:
                    tiles.append(self.to_tensor(im))
        
        tiles = tuple(tiles)
        tiles = torch.cat(tiles, dim = 0)
        tiles = tiles.unsqueeze(1)
        resp = self.model(tiles)
        
        if output == "binary":
            if torch.max(resp) > confidence_threshold:
                return 1
            else:
                return 0
        
        elif output == "probability":
            # The "worst" tiles can be plotted. Requires matplotlib import.
            #max_index = torch.argmax(resp).item()
            #plt.imshow(tiles[max_index].squeeze())
            #plt.show()
            return torch.max(resp).item()
        
        elif output == "tensor":
            return resp
        
        else:
            raise Exception(f"No valid action: output = {output}.")
    
    def check_folder(
            self,
            folder,
            output_file = None,
            return_dict = True,
            confidence_threshold = None,
            **kwargs
            ):
        '''
        Search a folder for .tif images and apply the check_image method.

        Parameters
        ----------
        folder : str
            Directory path.
        output_file : str, optional
            Path to write the output as text. The default is None.
        return_dict : bool
            Whether to return a dictionary with the output files.
            The default is True.
        confidence_threshold : float, optional
            A confidence threshold in [0, 1] to decide whether an image is ok
            or flawed and needs inspection. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        bad_files : list
            Responses from check_image. Either a list of images which are wrong
            with a probability higher than the threshold, or a list of
            probabilities for all images within the directory.
        
        Note
        ----
        This method is not (yet) prepared to pass all the arguments possible in
        check_image.

        '''
        files = glob(os.path.join(folder, "*.tif"))
        print(f"Found {len(files)} files.")
        
        bad_files = []
        
        if confidence_threshold is not None:
            file_list = []
        else:
            file_list = files
        
        for file in files:
            prob = self.check_image(
                file, confidence_threshold = confidence_threshold
                )
            
            if confidence_threshold is not None:
                if prob > confidence_threshold:
                    file_list += [file]
                    bad_files += [prob]
            
            else:
                bad_files += [prob]
        
        if return_dict:
            out = {f : v for f, v in zip(file_list, bad_files)}
            bad_files = [f"{f},{v}" for f, v in zip(file_list, bad_files)]
        
        else:
            out = bad_files
        
        if isinstance(output_file, str):
            parent = os.path.dirname(output_file)
            if not os.path.isdir(parent):
                output_file = os.path.join(os.path.getcwd())
                
                try:
                    with open(output_file, "w") as f:
                        f.write("\n".join(bad_files))
                
                except FileNotFoundError:
                    warnings.warn(
                        f"Cannot write to {output_file}.\n" +
                        f"{parent} is not a directory."
                        )
        
        return out