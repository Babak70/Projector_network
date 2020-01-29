# Projector_network
Data and source code for the article "Competing neural networks for the robust control of non-linear systems" currently under review and available on arXiv (https://arxiv.org/abs/1907.00126).


# Description


# Software-requirements
The neural network (NN) training has been implemented via a python-based costum script using Tensorflow. Any stable version of tensorflow (strictly below 2.0) and python 3 are good to use (we have used Python 3.7.3 and Tensorflow version 1.13.1). The outputs of the NN are subsequently processed by a matlab-based script. For an authomatic calling of the MATLAB-engine within python, the python package "MATLAB Engine API" should be pre-installed. Straight-forward instructions on how to install the required packages can be found here: (https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Our version of Matlab is R2019a.


# Hardware-requirements
Our costum-scripts have been executed on an NVIDIA GTX 1080 Titan GPU. Be advised that execution of the code without GPU-backed machinces (CPU only) considerably increases the convergence time.


# Data repository

Two versions of dataset one with 1k examples and the other with 20k examples are provided which can be found here and here.


# How to run the codes?




# Reference
If you use this code, please cite the following paper:
arXiv version: https://arxiv.org/abs/1907.00126
  

