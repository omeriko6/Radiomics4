In this repository we have all the corresponding files needed to use Pyradiomics and execute the module I built (main.py).

Stages for execution:

MRI Images – Data acquisition -> Image Pre – processing & Segmentation:

in order to have an MRI image and a corresponding mask file, you need to use “ImageJ” software that can be downloaded online.
each image needs to be uploaded, select the area of the tumor and export it as a mask file with the exact name of the image.
the path to each image and mask should be added to a csv file in order to use pyradiomics with this command:

pyradiomics <path/to/image> <path/to/segmentation> -o results_testing.csv -f csv


the output file is the csv file that the system needs in order to examine its attributes and decide whether the tumor has BRCA attributes.

* if you want to use the files I have put in the repositury, this stage is not necessary.

PyRadiomics-Feature Extraction ->Correlation with gene expression -> Classification and Prediction:

this stage is the work of the module I built, it needs the results_training.csv in the repository and a results_testing.csv in order to examine the output of the module.
the module also outputs the features that the system has detected as important to the prediction.


