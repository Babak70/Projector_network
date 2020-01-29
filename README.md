# Projector_network
Data and code for the article "Competing neural networks for the robust control of non-linear systems" currently under review and available on arXiv (https://arxiv.org/abs/1907.00126).


# Description
The source code as well as the dataset accompanying the paper "Competing neural networks for the robust control of non-linear systems" by B. Rahmani, D. Loterie, E. Kakkava, N. Borhani, U. Tegin, D. Psaltis and C. Moser is provided. The given scripts implement an algorithm for generating the required 2D input of a given system that would result in a desired target output. Here we showcase this proof-of-the-concept algorithm for projecting desired target images through the highly scattering medium of multi-mode optical fibers. 


# Software-requirements
The neural network (NN) training has been implemented via a python-based costum script using Tensorflow on Windows 10 OS. Any stable version of tensorflow (strictly below 2.0) and python 3 are good to use (we have used Python 3.7.3 and Tensorflow version 1.13.1). Tensorflow can be installed straightforwardly as intructed here https://www.tensorflow.org/ . 

The outputs of the NN are subsequently processed by a matlab-based script. For an authomatic calling of the MATLAB-engine within python, the python package "MATLAB Engine API" should be pre-installed. Straight-forward instructions on how to install the required packages can be found here: (https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Our version of Matlab is R2019a.


# Hardware-requirements
Our costum-scripts have been executed on an NVIDIA RTX 2080 Titan GPU. Be advised that execution of the code without GPU-backed machinces (CPU only) considerably increases the convergence time.


# Data repository
Two versions of dataset one with 1k examples and the other with 20k examples are provided which can be found here and here. The "dataset_readme.txt" explains each one in detail.


# How to run the codes?
Deponding on the dataset size (1k or 20k), the appropirate script "eye_train.py" should be executed. Prior to running the script, the four files containing the dataset should be placed in the same directory as the rest of script files. Executing the codes, the model (D and G sub-networks) starts to be trained. On a standard PC with Core i7-9800X CPU @3.8 GHz and 32GB RAM, the training takes about X hrs. Once the training is finished, the matlab-script is automatically run. After execution of the matlab-script the projected images are obtained and stored in the file "train_dataF_1". This process can be repeated for a second and more number of times. The dataset are automatically updated after each round. For this, the variable "number_of_iterations" in the file "eye_train.py" must be changed (default is 1). In addition to "train_dataF_1" which contains all projected images, some sample images are also provided in PNG formats.

# Demo
A demo of the trained neural network together with examples of the 2D input solutions as well as their corresponding outputs are provided for the dataset of 1k EMNIST Latin alphabet.

# Additional information



# Reference
If you use this code, please cite the following paper:
arXiv version: https://arxiv.org/abs/1907.00126
  

