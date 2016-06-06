#include "convnet.h"
#include "layer.h"
#include "load.h"
#include "utils.h"

#include <time.h>
#include <stdio.h>

void testGPU(Tensor * X_train, Tensor * Y_train) {
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
void testGPU2(Tensor * X_train, Tensor * Y_train) {
  // Index of training set to visualize
  const int k = 0;
  X_train->dims.num_images = 32;

  Tensor * dev_X_train;
  Tensor * dev_X_out;
  Tensor * X_out;
  Tensor * temp;

  // Copy to GPU
  dev_X_train = toGPU(X_train);

  // Output training data
  print(X_train, k, 0);

  // Fprop
  printf("\nFPROP\n\n");

  ConvLayer l1 = ConvLayer(2, 2, 2);
  l1.init_mem(&dev_X_train->dims);
  l1.fprop(dev_X_train, &dev_X_out);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);

  ConvLayer l2 = ConvLayer(2, 2, 2);
  l2.init_mem(&dev_X_out->dims);
  l2.fprop(dev_X_out, &dev_X_out);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);

  PoolingLayer l3 = PoolingLayer(2, 2);
  l3.init_mem(&dev_X_out->dims);
  l3.fprop(dev_X_out, &dev_X_out);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);


  temp = toCPU(l3.dev_switches_row);
  print(temp, 0, 0);
  temp = toCPU(l3.dev_switches_col);
  print(temp, 0, 0);


  printf("\nWEIGHTS+BIASES\n\n");

  temp = toCPU(l1.dev_weights);
  print(temp, 0, 0);
  temp = toCPU(l1.dev_biases);
  print(temp, 0, 0);

  temp = toCPU(l2.dev_weights);
  print(temp, 0, 0);
  temp = toCPU(l2.dev_biases);
  print(temp, 0, 0);



  printf("\nBPROP\n\n");

  l3.bprop(&dev_X_out, dev_X_out, 0.1);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);

  l2.bprop(&dev_X_out, dev_X_out, 0.1);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);

  l1.bprop(&dev_X_out, dev_X_out, 0.1);
  X_out = toCPU(dev_X_out);
  print(X_out, k, 0);



  printf("\nWEIGHTS+BIASES\n\n");

  temp = toCPU(l1.dev_weights);
  print(temp, 0, 0);
  temp = toCPU(l1.dev_biases);
  print(temp, 0, 0);

  temp = toCPU(l2.dev_weights);
  print(temp, 0, 0);
  temp = toCPU(l2.dev_biases);
  print(temp, 0, 0);



  printf("\nWEIGHTS+BIASES GRADIENTS\n\n");

  temp = toCPU(l1.dev_weights_grad);
  print(temp, 0, 0);
  temp = toCPU(l1.dev_biases_grad);
  print(temp, 0, 0);

  temp = toCPU(l2.dev_weights_grad);
  print(temp, 0, 0);
  temp = toCPU(l2.dev_biases_grad);
  print(temp, 0, 0);
}
void testGPU3(Tensor * X_train, Tensor * Y_train) {

  // Copy to GPU
  Tensor * dev_X_train = toGPU(X_train);
  Tensor * dev_Y_train = toGPU(Y_train);

  // Init convnet
  const int num_layers = 3;
  Layer ** layers = new Layer*[num_layers];
  layers[0] = new ConvLayer(2, 2, 2); 
  layers[1] = new ConvLayer(2, 2, 2); 
  layers[2] = new PoolingLayer(2, 2);
  ConvNet cnet = ConvNet(layers, num_layers, dev_X_train, dev_Y_train);

  // Train
  const float eta = 0.01;
  const int num_epochs = 3;
  const int num_batches = 1;
  const int batch_size = 32;
  const int train_size = num_batches * batch_size;

  cnet.train(eta, num_epochs, num_batches, batch_size, train_size);

  // TODO: Free layers
}

// Tests basic ANN (using only fully connected layers)
void testGPU4(Tensor * X_train, Tensor * Y_train) {
  // Copy to GPU
  Tensor * dev_X_train = toGPU(X_train);
  Tensor * dev_Y_train = toGPU(Y_train);

  const int num_layers = 2;
  
  Layer ** layers = new Layer*[num_layers];

  layers[0] = new FullyConnectedLayer(100, 784);
  layers[1] = new FullyConnectedLayer(10, 100);
  ConvNet cnet = ConvNet(layers, num_layers, dev_X_train, dev_Y_train);

  // Train
  const float eta = 0.01;
  const int num_epochs = 10;
  const int num_batches = 1;
  const int batch_size = 32;
  const int train_size = num_batches * batch_size;
  
  cnet.train(eta, num_epochs, num_batches, batch_size, train_size);
  
}

int main() {
  Tensor * X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  Tensor * Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);


  clock_t start = clock();
  testGPU4(X_train, Y_train);
  printf("Time: %d\n", (int) (clock() - start));
}
