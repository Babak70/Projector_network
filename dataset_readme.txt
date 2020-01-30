===========================================================================================
For training the script, 5 data files are required:

1- train_dataF.bin:    contains examples of the 2D outputs of the system. Initially it is a set of 20k, 200x200 random outputs. After the first round of training
this file is updated with projected images that are obtained using the solutions found by the network.

2- train_dataF_1.bin:  contains examples of 2D target images. The images are therefore of the size of 20k 200x200.

3- train_labelsF.bin:  contains examples of the real part of the 2D inputs solutions found by the network. Initially it is a set of 20k, 51x51 random numbers. After the first round of training
this file is updated with a solutions predicted by the network.


4- train_labelsF_1.bin: contains examples of the imaginary part of the 2D inputs solutions found by the network. Initially it is a set of 20k, 51x51 random numbers. After the first round of training
this file is updated with a solutions predicted by the network.


5- transmit.mat: contains the forward transfer function of the system (forward propagation of the light through the fiber)


===========================================================================================
For runnig the demo (pre-trained model) 2 data files are required:

1- eval_dataF.bin: contains 1k examples of 2D target images of size 200x200.

2- eval_labelsF.bin: contains 1k all-zero 2D input patterns of size 51x51 (dummy variable) should already be in demo directory (otherwise download).

3- transmit.mat: contains the forward transfer function of the system (forward propagation of the light through the fiber)

4- fiber10_train: the pre-trained model
