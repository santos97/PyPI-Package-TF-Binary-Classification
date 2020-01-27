# PyPI-Package-TF-Binary-Classification
This package enables developers to train and test a ready model for binary classification using single line of terminal caoomand. 
#TensorFLow-Binary-Image-Classifier

A Python package to get train and test a model for binary classification.

Usage

Following query on terminal will allow you to TRAIN the data. Here c1 and c2 are two categories and has SAME folder name of the data. p is path of folder containing train data image folders. e is number of epoches EX:

[Train_data] / 
        [not_human] , [human]

train -c1 not_human -c2 human -p c:/downloads/train_data/ -e 3

Following query on terminal will allow you to TEST the data. Here c1 and c2 are two categories and has SAME folder name of the data. p is path of folder containing test data image folder. EX:

[Test_data] / 
        [folder]

test -c1 not_human -c2 human -p c:/downloads/test_data/ 
