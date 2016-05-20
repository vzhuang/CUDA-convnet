#include "load.hpp"
#include "layer.hpp"

#include <iostream>


int main() {
  float *** X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  float **  Y_train = load_Y("../data/train-labels.idx1-ubyte", TEST_SIZE);
  float *** X_test  = load_X("../data/t10k-images.idx3-ubyte", TRAIN_SIZE);
  float **  Y_test  = load_Y("../data/t10k-labels.idx1-ubyte", TEST_SIZE);

  int k = 0;

  visualize(X_train, k);
  std::cout << un_hot(Y_train[k]) << std::endl;
  visualize(X_test, k);
  std::cout << un_hot(Y_test[k]) << std::endl;


  ActivationLayer layer;
  Dimensions d = {1, DIM, DIM};

  layer.forward_prop(X_train, &d, X_test, &d);

  visualize2(X_test, k);
}


