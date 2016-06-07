# CUDA-convnet

CUDA/C++ CNN (convolutional neural network) implementation using cuBLAS.

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
  * hard padding
  * arbitrary stride, number of filters
* max-pooling layers
* fully connected layers
  * supports arbitrary number of neurons

## Results
_______

We test our performance on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The vanilla cpu version achieves ~80% accuracy at convergence, while a simple 3 layer (convolution->pooling->fully connected) gpu network achieves 96%+ accuracy. It also trains much faster compared to the CPU version. While the CPU version takes ~50 seconds to run through the entire dataset of 60,000 points, the GPU version does it in ~4 seconds.

Output for `batch_size = 1`
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

Output for `batch_size = 10`, 100 batches (not fully converged yet)
```
Layer 0: 10 x 1 x 28 x 28 --> 10 x 10 x 14 x 14
Layer 1: 10 x 10 x 14 x 14 --> 10 x 10 x 7 x 7
Layer 2: 10 x 10 x 7 x 7 --> 10 x 1 x 10 x 1
Epoch loss: 0.507805
Epoch loss: 0.255975
Epoch loss: 0.209227
Epoch loss: 0.189912
Epoch loss: 0.179246
Epoch loss: 0.17173
Epoch loss: 0.166663
Epoch loss: 0.16233
Epoch loss: 0.159164
Epoch loss: 0.156381
Epoch loss: 0.153798
Epoch loss: 0.151631
Epoch loss: 0.149532
Epoch loss: 0.147698
Epoch loss: 0.146265
Epoch loss: 0.144249
Epoch loss: 0.142266
Epoch loss: 0.140616
Epoch loss: 0.139
Epoch loss: 0.137083
Epoch loss: 0.135683
Epoch loss: 0.133884
Epoch loss: 0.132434
Epoch loss: 0.130967
Epoch loss: 0.129168
Epoch loss: 0.127168
Epoch loss: 0.125368
Epoch loss: 0.123668
Epoch loss: 0.121818
Epoch loss: 0.120085
Epoch loss: 0.118335
Epoch loss: 0.116302
Epoch loss: 0.114518
Epoch loss: 0.113052
Epoch loss: 0.111902
Epoch loss: 0.110185
Epoch loss: 0.108402
Epoch loss: 0.106952
Epoch loss: 0.105751
Epoch loss: 0.104318
Epoch loss: 0.102951
Epoch loss: 0.101401
Epoch loss: 0.100051
Epoch loss: 0.0987347
Epoch loss: 0.097368
Epoch loss: 0.0964847
Epoch loss: 0.095168
Epoch loss: 0.093968
Epoch loss: 0.093018
Epoch loss: 0.0918179
Epoch loss: 0.0907346
Epoch loss: 0.0898179
Epoch loss: 0.0888679
Epoch loss: 0.0878345
Epoch loss: 0.0870012
Epoch loss: 0.0855012
Epoch loss: 0.0843845
Epoch loss: 0.0836178
Epoch loss: 0.0826178
Epoch loss: 0.0818845
Epoch loss: 0.0812344
Epoch loss: 0.0806178
Epoch loss: 0.0797511
Epoch loss: 0.0790677
Epoch loss: 0.0785511
Epoch loss: 0.0779677
Epoch loss: 0.0771677
Epoch loss: 0.076701
Epoch loss: 0.0761177
Epoch loss: 0.075551
Epoch loss: 0.0750343
Epoch loss: 0.0745177
Epoch loss: 0.074151
Epoch loss: 0.0736676
Epoch loss: 0.0731176
Epoch loss: 0.0726343
Epoch loss: 0.0721176
Epoch loss: 0.0717176
Epoch loss: 0.0713509
Epoch loss: 0.0709843
Epoch loss: 0.0706176
Epoch loss: 0.0703176
Epoch loss: 0.0700009
Epoch loss: 0.0697176
Epoch loss: 0.0694676
Epoch loss: 0.0691009
Epoch loss: 0.0689676
Epoch loss: 0.0686176
Epoch loss: 0.0683509
Epoch loss: 0.0682176
Epoch loss: 0.0681009
Epoch loss: 0.0679342
Epoch loss: 0.0677509
Epoch loss: 0.0675675
Epoch loss: 0.0675842
Epoch loss: 0.0675009
Epoch loss: 0.0672842
Epoch loss: 0.0670842
Epoch loss: 0.0668675
Epoch loss: 0.0666842
Time: 396s
```
