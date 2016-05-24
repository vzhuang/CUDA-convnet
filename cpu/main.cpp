#include "layer.hpp"
#include "load.hpp"
#include "convnet.hpp"


int main() {
  Tensor * X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  float ** Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);
  Tensor * X_test  = load_X("../data/t10k-images.idx3-ubyte", TEST_SIZE);
  float ** Y_test  = load_Y("../data/t10k-labels.idx1-ubyte", TEST_SIZE);

  const int num_layers = 4;

  Layer ** layers = new Layer*[num_layers];

  // Convolution layer: 32 5x5 filters
  layers[0] = &ConvLayer(32, 5, 5);
  
  // ReLU activation layer
  layers[1] = &ActivationLayer();

  // 2x2 Max pooling layer
  layers[2] = &PoolingLayer(2, 2);

  // Fully connected layer
  layers[3] = &FullyConnectedLayer(10, 10);

  // Softmax output layer
  

  // Train neural network
  ConvNet net = ConvNet(layers, num_layers, X_train, Y_train);
  net.train(0.1, 2, 1);

  // std::cout << "wot m8" << std::endl;
}
