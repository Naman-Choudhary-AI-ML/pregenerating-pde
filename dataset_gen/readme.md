# Dataset Intructions

## Intro
The NS_datasets, and CE_datasets folder contain datasets for Lid driven cavity flow, and Flow past an object

Different types of datasets across families:

1. Lid driven cavity flow
2. Flow past an object inside a cylinder (meaning top and bottom walls have no slip)
3. Flow past an object externally (meaning top and bottom walls are free flowing)

## Data Gen Setup
To run the datagen code, in each folder for each PDE and setup, we have a main.py file, which needs to be run to develop datasets.
You will need to first have openfoam installed, and working, either through apptainer or docker or any other way.
In case of apptainer run
apptainer exec openfoam.sif bash
to use openfoam

Then go inside the required folder and run python main.py

Once data is generated, to restructure data correctly and add additional channels needed, you will find a ipynb file in each folder.
Inside that, you should be able to restructure openfoam data, into row major format, and add gemeotry mask, SDF channels as well. It also contains code to visualize the data.
