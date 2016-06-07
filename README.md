# CUDA-convnet

CUDA/C++ CNN (convolutional neural network) implementation using CUDA/cuBLAS.

## Usage
_____

In the desired directory (cpu or gpu), simply

`make`

then run.

## Convolutional Neural Networks
_____________________________

Convolutional neural networks are a variant of neural networks that utilize convolutional layers in order to capture translationally invariant patterns within the input. As it turns out, stacking together many of these convolutional layers produces an extremely powerful, nonlinear model. The effectiveness of this approach, called deep learning, has been demonstrated in many applications, including image recognition, game-playing, natural language processing, autonomous driving, and more.

CNNs are trained using the standard backpropagation algorithm, which boils down to a series of matrix multiplications. Hence, CNNs are very amenable to GPGPU, and we seek to exploit its parallelizable nature in our implementation.


## Layers
____________

We implement the following layers:

* convolution layers
* max-pooling layers
* fully connected layers

## Results
_______

We test our performance on the MNIST dataset. Our following architecture achieves accuracy.

The vanilla cpu version achieves ~80% accuracy at convergence, while a simple 3 layer (convolution->pooling->fully connected) gpu network achieves ~95% accuracy. It also trains much faster [numbers needed].

```
Layer 0: 1 x 1 x 28 x 28 --> 1 x 10 x 14 x 14
Layer 1: 1 x 10 x 14 x 14 --> 1 x 10 x 7 x 7
Layer 2: 1 x 10 x 7 x 7 --> 1 x 1 x 10 x 1
Epoch loss: 0.232267
Epoch loss: 0.146917
Epoch loss: 0.125667
Epoch loss: 0.1067
Epoch loss: 0.092
Epoch loss: 0.0806833
Epoch loss: 0.0728667
Epoch loss: 0.0672167
Epoch loss: 0.0625
Epoch loss: 0.0587833
Epoch loss: 0.0563
Epoch loss: 0.0539333
Epoch loss: 0.0521167
Epoch loss: 0.0505833
Epoch loss: 0.0496333
Epoch loss: 0.0486833
Epoch loss: 0.04765
Epoch loss: 0.0466833
```
