# JSDML
The Jensen-Shannon Divergence and Machine Learning Toolbox

Code written by  Nick Carrara (and including several routines written by G. Ver Steeg). 
Based on: “On the Upper Limit of Separability”, Nick Carrara, Jesse Ernst, arXiv:1708.09449 [hep-ex].

(with some code borrowed from Greg Ver Steeg's NPEET package (https://github.com/gregversteeg/NPEET))

The JSDML is a python package that implements a neural network using Keras/Tensorflow for feature selection and also calculates MI or the JSD between features and class labels using methods developed by Kraskov (https://arxiv.org/pdf/cond-mat/0305641.pdf) and modified code from NPEET (https://github.com/gregversteeg/NPEET).

Documentation is provided at https://github.com/AlbanyHEP/JSDML/blob/master/JSDML_Documentation.pdf.  For users who wish to get started quickly without wading through the main documentation will find a quick start guide and examples in (https://github.com/AlbanyHEP/JSDML/tree/master/Quick_Start_Guide).  

# SUSY Example

In order to run the example included in the main part of JSDML.py, you will need to first download the SUSY data set from the UCI repository (https://archive.ics.uci.edu/ml/datasets/SUSY).  Then you will need to run the SUSYParser.py script which will break up the data into separate signal and background files 'SUSYSignal.csv' and 'SUSYBackground.csv' respectively.
