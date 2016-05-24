#include "load.hpp"
// #include "layer.hpp"
#include "nnet.hpp"



int main() {
  float **** X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  float **   Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);
  float **** X_test  = load_X("../data/t10k-images.idx3-ubyte", TEST_SIZE);
  float **   Y_test  = load_Y("../data/t10k-labels.idx1-ubyte", TEST_SIZE);

  const int num_layers = 3;

  Layer ** layers = new Layer*[num_layers];
  ConvLayer l1 = ConvLayer(1, 2, 2);
  layers[0] = &l1;
  layers[1] = &l1;
  PoolingLayer l2 = PoolingLayer(2, 2);
  ActivationLayer l3 = ActivationLayer();
  layers[2] = &l3;

  Dimensions d = {1, 1, DIM, DIM};

  NeuralNetwork net(layers, num_layers, X_train, Y_train, &d);
  net.step();

  free_X(X_train, TRAIN_SIZE);
  free_Y(Y_train, TRAIN_SIZE);
  free_X(X_test, TEST_SIZE);
  free_Y(Y_test, TEST_SIZE);
}
