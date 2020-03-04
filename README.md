# absDL

In this repository we arranged the required elements for training a deep network to deploy single-shot 
absorption imaging in your lab.

We provide a Matlab script to assist with correct handling of the training and prediction images, and
a python script for the network training. Both will enable you to reproduce the illustration figures 
from the paper:

[Single-exposure absorption imaging of ultracold atoms using deep learning](http://arxiv.org/abs/2003.01643)

![Infrence examples](atoms_examples.png)

Here we provide hyperparameters and architecture that were used to generate the results in the paper. 
Most of them were crudely optimized for the Technion ultracold fermions lab apparatus and will need 
variation in order to fit other use-cases. We encourage you to contribute your insights to our project!

absDNN repository is provided and maintained by Yoav Sagi’s [ultracold fermions research group](https://phsites.technion.ac.il/sagi/) from the Technion – Israel Institute of Technology.


## Environment:
#### 1. Using Anaconda recipe:
Please follow the Anaconda recipe provided in this repo.
Download by clicking on "View on GitHub" on top or [here](https://github.com/absDL/absDL.github.io).

#### 2. Using pip:
* make sure that your virtual environment is installed with python 3.7
* make sure your pip is pointing on your venv interpreter: <venv_location/bin/pip>
* on the project directory type in:
```
$ pip install -r requirements.txt 
```

## Dataset:
Use the Matlab script 'dataset_extractor.m' to extract a 32bit TIFF images from your experimental setup.
In case you have the raw camera images stored as MAT files small modifications to this script would do.
Even if it not the case you can follow its lines to understand how we normalized the images and extract
training and testing sets for your data.

### Original dataset:
The dataset used in the paper can be downloaded from [here](https://www.dropbox.com/s/dg7y000rmicc8ed/single_shot_dataset.tar?dl=0). We use 
constant normalization for all frames to adjust the log images and store them in separate TIFF files.
In our experimental setup the camera is installed from beneath the vacuum chamber. Therefore, to focus 
the image on the free-falling atoms, the imaging system was manually shifted to follow the gravitational-
induced position of the atoms. This introduced a variation of the noise patterns between sets of images, 
which was otherwise quite limited.

## Training:
You can simply run 'main.py' after referencing to the datasets with and without atoms in 'wAtoms_ds.txt'
and 'woAtoms_ds.txt', respectively.
The main script includes an argument parser to allow variation of the options and parameters from within the terminal.
You can view the different options by running main.py with the -h flag (or --help).  
