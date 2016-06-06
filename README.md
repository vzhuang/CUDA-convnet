# CUDA-convnet

CUDA/C++ CNN (convolutional neural network) implementation using CUDA/cuBLAS.

Usage
_____

In the desired directory (cpu or gpu), simply

`make`

then run.

Convolutional Neural Networks
_____________________________

Convolutional neural networks are a variant of neural networks that utilize convolutional layers in order to capture translationally invariant patterns within the input. As it turns out, stacking together many of these convolutional layers produces an extremely powerful, nonlinear model. The effectiveness of this approach, called deep learning, has been demonstrated in many applications, including image recognition, game-playing, natural language processing, autonomous driving, and more.

CNNs are trained using the standard backpropagation algorithm, which boils down to a series of matrix multiplications. Hence, CNNs are very amenable to GPGPU, and we seek to exploit its parallelizable nature in our implementation.


Architecture
____________

We implement the following layers:

* convolution layers
* max-pooling layers
* fully connected layers

Results
_______