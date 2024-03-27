Welcome to Radiomics4!


In this repository we have all the corresponding data and needed to use Pyradiomics and execute the module I built (main.py).


How to run the model:
Open a project in python, add the main function to your own project.
In order to run the program you need to download the following libraries:
NumPy
Panda
sklearn
pickle
The import commands are already in the code.


Stages for execution:

  --------------------------------

MRI Images – Data acquisition -> Image Pre – processing & Segmentation:

in order to have an MRI image and a corresponding mask file, you need to use “ImageJ” software that can be downloaded online.
Each image needs to be uploaded, select the area of the tumor and export it as a mask file with the exact name of the image.

the path to each image and mask should be added to a csv file in order to use pyradiomics with the following command:

pyradiomics <path/to/input> -o results.csv -f csv

(Change path to input to the appropriate csv file from before and results.csv to a place you want the output to be saved. Combine all of the data extractions to one file.)

the output file is the csv file that the system needs in order to examine its attributes and decide whether the tumor has BRCA attributes.

* if you want to use the files I have put in the repositury, this stage is not necessary.


  --------------------------------

PyRadiomics-Feature Extraction ->Correlation with gene expression -> Classification and Prediction:

Add to the CSV file from before a column named: “targets” which are the groups in this experiment, 1 or 0 ( in our case, 1 stands for BRCA mutation).

Later, use the module as is. You need to change to the correct path to the input files in your computer. All the needed files can be taken from the GitHub. You can choose how many features you want to consider as important.

There is a possibility to save the module after the training. This features are commented and need to be uncommented if you want to use them.



