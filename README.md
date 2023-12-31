# Welcome to QuickCheckRasterWriteError
## About
In this repository, we share code to train a simple CNN on valid and corrupt raster files. We then use the trained model to check folders containing .tif files for raster write errors associated with OOM issues.
## Installation
- Download this repository as a compressed archive.
- Unzip the archive to some place on your harddrive.
- Optional: Create and activate a virtual environment in Python
- Install the required Python modules (requirements.txt), e.g. using pip as
`pip install -r requirements.txt`
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
## Commandline interface
The `interface.py` script was added to make the main functionalities accessible from a terminal or from within other languages such as R via system calls.

Here is an example:
```console
cd ./py3
python3 interface.py /directory/containing/tiff/rasters --outfile /path/to/some/table.csv
```
Note that in Windows, the equivalent to `python3` would be `py`.

To call it from within an R session:
```R
directory = "/directory/containing/tiff/raste"
output = "/path/to/some/table.csv"
setwd("./py3")
cmd <- paste("python3", "interface.py", directory, "--outfile", output)
system(cmd)
```
