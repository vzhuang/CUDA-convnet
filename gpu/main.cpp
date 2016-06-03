#include "layer.h"
#include "load.h"
#include "utils.h"

#include <time.h>
#include <stdio.h>


void testGPU(Tensor * X_train) {
  // Index of training set to visualize
  const int k = 3;

  // Output training data
  print(X_train, k, 0);

  // Copy to GPU, init layer + memory
  Tensor * dev_X_train = toGPU(X_train);
  Tensor * dev_X_out;
  PoolingLayer l1 = PoolingLayer(2, 2);
  l1.init_mem(&dev_X_train->dims);

  // Fprop
  l1.fprop(dev_X_train, &dev_X_out);
  Tensor * X_out = toCPU(dev_X_out);
  print(X_out, k, 0);

  // Bprop 
  Tensor * dev_input_grad;
  l1.bprop(&dev_input_grad, dev_X_out, 1);
  Tensor * input_grad = toCPU(dev_input_grad);
  print(input_grad, k, 0);

  // Free layer
  l1.free_mem();
}
void testGPU2(Tensor * X_train) {
  // Index of training set to visualize
  const int k = 420;

  // Copy to GPU, init layer + memory
  Tensor * dev_X_train = toGPU(X_train);
  Tensor * dev_X_out;
  ConvLayer l2 = ConvLayer(10, 2, 2);
  l2.init_mem(&dev_X_train->dims);

  // Output training data
  print(X_train, k, 0);

  // Fprop
  l2.fprop(dev_X_train, &dev_X_out);
  Tensor * X_out = toCPU(dev_X_out);


  // print(X_out, k, 0);
  // print(X_out, k, 1);
  // print(X_out, k, 2);
  print(X_out, k, 3);

  // Tensor * temp;
  // temp = toCPU(l2.dev_weights);
  // print(temp, k, 3);
  // temp = toCPU(l2.dev_stretch_weights);
  // print(temp, k, 0);
  // temp = toCPU(l2.dev_stretch_input);
  // print(temp, k, 0);
  // temp = toCPU(l2.dev_stretch_output);
  // print(temp, k, 0);
}

int main() {
  Tensor * X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  Tensor * Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);

  clock_t start = clock();
  testGPU2(X_train);
  printf("Time: %d\n", (int) (clock() - start));

}
