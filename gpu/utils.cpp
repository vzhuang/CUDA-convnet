#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>


void one_hot(int k, float * arr, int size) {
  assert(k < size);
  memset(arr, 0, sizeof(float) * size);
  arr[k] = 1;
}
int un_hot(float * arr, int size) {
  for (int i = 0; i < size; i++)
    if (arr[i] == 1)
      return i;
  return -1;
}

float sigmoid(float x)
{
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

void print(Tensor * t, int n, int c) {
  int num_images = t->dims.num_images;
  int num_channels = t->dims.num_channels;
  int rows = t->dims.rows;
  int cols = t->dims.cols;

  printf("Size: %d x %d x %d x %d\n", num_images, num_channels, rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%7.3f ", t->data[((n * num_channels + c) * rows + i) * cols + j]);
    }
    printf("\n");
  }
}


Tensor * toGPU(Tensor * t) {
  assert(!t->gpu_memory);
  Tensor * dev_t = new Tensor(&(t->dims), true);

  int data_size = t->dims.num_images * t->dims.num_channels * t->dims.rows * t->dims.cols * sizeof(float);
  cudaMemcpy(dev_t->data, t->data, data_size, cudaMemcpyHostToDevice);
  return dev_t;
}
Tensor * toCPU(Tensor * dev_t) {
  assert(dev_t->gpu_memory);
  Tensor * t = new Tensor(&(dev_t->dims), false);

  int data_size = dev_t->dims.num_images * dev_t->dims.num_channels * dev_t->dims.rows * dev_t->dims.cols * sizeof(float);
  cudaMemcpy(t->data, dev_t->data, data_size, cudaMemcpyDeviceToHost);
  return t;
}

