# Comparing-node-embedding-methods-and-classifiers-for-predicting-disease-genes
This repository contains the Python code of the work done for the project of a course called "Learning from Networks" (Master Degree in Data Science). 

The aim of the work has been to compare different kinds of node embedders (factorization-based and random-walk-based) along with different classifiers (Random Forest, AdaBoost, MLP, CNN) to predict gene-disease associations.
The code uses karateclub (https://github.com/benedekrozemberczki/karateclub) for the embedders, Sklearn for the Machine Learning and MLP classifiers, and Pytorch for the CNN classifier.
Everything runs on the CPU apart from the CNN that can use cuda. The dataset used is DisGeNet (smaller version), which can be found at https://snap.stanford.edu/biodata/datasets/10012/10012-DG-AssocMiner.html.
This work is inspired by https://ieeexplore.ieee.org/document/8983134.
