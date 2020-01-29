# Projector_network
Data and source code for the article "Competing neural networks for the robust control of non-linear systems" currently under review and available on arXiv (https://arxiv.org/abs/1907.00126).


# Description


# Software-requirements
The neural network (NN) training has been implemented via a python-based costum script using the Tensorflow library. Any stable versions of tensorflow (strictly below 2.0) and python 3 are good to use (we have used Python 3.7.3 and Tensorflow version 1.13.1). Since the generated data by the NN is picked up by a matlab-based script subsequently, the python package "Matlab engine" should be installed. 

Matlab engine

# Hardware-requirements
Our costum-scripts have been executed on an NVIDIA GTX 1080 Titan GPU. Be advised that execution of the code without GPU-backed machinces (CPU only) considerably increases the convergence time.

# Reference
If you use this code, please cite the following paper:
arXiv version: https://arxiv.org/abs/1907.00126
  

