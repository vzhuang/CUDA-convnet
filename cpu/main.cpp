#include "layer.hpp"
#include "load.hpp"
#include "convnet.hpp"


int main() {
  Tensor * X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  float ** Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);
  Tensor * X_test  = load_X("../data/t10k-images.idx3-ubyte", TEST_SIZE);
  float ** Y_test  = load_Y("../data/t10k-labels.idx1-ubyte", TEST_SIZE);

  const int num_layers = 2;
  
  Layer ** layers = new Layer*[num_layers];

  layers[0] = new FullyConnectedLayer(100, 784, RELU);

  layers[1] = new FullyConnectedLayer(10, 100, SIGMOID);

  // Train neural network
  ConvNet net = ConvNet(layers, num_layers, X_train, Y_train);
  net.train(0.1, 20, 2, 1);
}
