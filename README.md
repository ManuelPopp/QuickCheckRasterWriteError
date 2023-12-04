# Welcome to QuickCheckRasterWriteError
## About
In this repository, we share code to train a simple CNN on valid and corrupt raster files. We then use the trained model to check folders containing .tif files for raster write errors associated with OOM issues.
## Installation
- Download this repository as a compressed archive.
- Unzip the archive to some place on your harddrive.
- Optional: Create and activate a virtual environment in Python
- Install the required Python modules (requirements.txt), e.g. using pip as
`pip install requirements.txt`
- Create a new Python script similar to `/py3/Examples.py`:
```python
#-----------------------------------------------------------------------------|
#> Import modules
# Append current file directory to import from other Python scripts
import os, sys

dir_py = "/path/to/main/py3"
sys.path.append(dir_py)

from model import SimpleCNN
from apply_model import ImageChecker

#-----------------------------------------------------------------------------|
#> Example
# Instantiate ImageChecker with the path to a trained model
im = ImageChecker("/home/poppman/main/mod/trained_model_sd.pt")

# Run check on a directory
probabilities = im.check_folder("/home/poppman/downloads/Maps/NOTOK")
```
The above code will return a dictionary of probabilities for the files within the given folder to be corrupt (default).
If you want a list of probabilities instead, use option
```python
return_dict = False
```
In order to also save the output as a .csv table, set a valid file path:
```python
output_file = "/path/to/my/file.csv"
```
