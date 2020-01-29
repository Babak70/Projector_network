For running the script, 4 data files are required.

1- train_dataF.bin: Contains examples of the 2D outputs of the system. Initially it is a set of 1k\20k, 200x200 random outputs. After the first round of training
this file is updated with projected images that are obtained using the solutions found by the network.

2- train_dataF_1.bin: contains examples of 2D target images. The images are therefore of the size of 1k\20k 200x200.

3- train_labelsF.bin: Contains examples of the real part of the 2D inputs solutions found by the network. Initially it is a set of 1k\20k, 51x51 random numbers. After the first round of training
this file is updated with a solutions predicted by the network.


3- train_labelsF_1.bin: Contains examples of the imaginary part of the 2D inputs solutions found by the network. Initially it is a set of 1k\20k, 51x51 random numbers. After the first round of training
this file is updated with a solutions predicted by the network.