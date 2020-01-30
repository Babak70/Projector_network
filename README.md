# Projector_network
Data and code for the article "Competing neural networks for the robust control of non-linear systems" currently under review and available on arXiv (https://arxiv.org/abs/1907.00126).


# Description
The source code as well as the dataset accompanying the paper "Competing neural networks for the robust control of non-linear systems" by Rahmani et. al. is provided. The given scripts implement an algorithm for generating the required 2D input of a given system that would result in a desired target output. Here we showcase this proof-of-the-concept algorithm for projecting desired target images through the highly scattering medium of multi-mode optical fibers. 


# Software-requirements
The neural network (NN) training has been implemented via a python-based costume script using Tensorflow installed on Windows 10 OS. Any stable version of tensorflow (strictly below 2.0) and python 3 can be used (we have used Python 3.7.3 and Tensorflow version 1.13.1). Tensorflow can be installed straightforwardly as instructed here https://www.tensorflow.org/ . 

The outputs of the NN are subsequently processed by a matlab-based script. For an automatic calling of the MATLAB-engine within python, the python package "MATLAB Engine API" should be pre-installed. Instructions on how to install the required packages can be found here: (https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). We have used Matlab version R2019a.


# Hardware-requirements
Our costume-scripts have been executed on an NVIDIA RTX 2080 Titan GPU. Be advised that execution of the code without GPU-backed machines (CPU only) considerably increases the convergence time.


# Data repository
A dataset containing 20k examples of the EMNIST target images, initial random 2D input patterns (real and imaginary) and their corresponding output patterns is provided which can be used to train the network from the scratch. It can be downloaded from here. A smaller dataset containing 1k examples of EMNIST target images is also provided for demo. The "dataset_readme.txt" explains each one in detail.


# How to run the codes?
For training the neural network, the script "eye_train.py" should be executed. Prior to running the script, the four files containing the dataset should be placed in the same directory as the rest of the script files. Upon executing the codes, the model (D and G sub-networks) starts to be trained. On a standard PC with Core i7-9800X CPU @3.8 GHz and 32GB RAM, one round of the training takes about  1h 15mins. Once the training is finished, the matlab-script is automatically run. After execution of the matlab-script, the projected images are obtained and stored in the file "train_dataF_1". This process can be repeated multiple times. The dataset are automatically updated after each round of training. For this, the variable "number_of_iterations" in the file "eye_train.py" must be adjusted (default is 1). In addition to "train_dataF_1" which contains all projected images, some sample images are also provided in PNG formats. To train the network with the dataset of 20k EMNIST follow the instructions in the "train_readme.txt" file.

# Demo
A demo of the neural network trained with 20k EMNIST dataset together with examples of the 2D input solutions as well as their corresponding outputs (in PNG) are provided. To run the demo with the toy dataset of 1k EMNIST follow the instructions in the "Demo_readme.txt" file.

# Additional information



# Reference
If you use this code, please cite the following paper:
arXiv version: https://arxiv.org/abs/1907.00126
