#include "tensor.h"

#include <cstdlib>
#include <cuda_runtime.h>


// Constructors that set dimensions and allocates memory for data array
Tensor::Tensor(int num_images_, int num_channels_, int rows_, int cols_, bool gpu_memory_) {
  dims.num_images = num_images_;
  dims.num_channels = num_channels_;
  dims.rows = rows_;
  dims.cols = cols_;

  int arr_size = dims.num_images * dims.num_channels * dims.rows * dims.cols * sizeof(float);
  if (!gpu_memory_)
    data = (float*) malloc(arr_size);
  else
    cudaMalloc((void **)&data, arr_size);

  gpu_memory = gpu_memory_;
}
Tensor::Tensor(Dimensions * dims_, bool gpu_memory_) {
  dims.num_images = dims_->num_images;
  dims.num_channels = dims_->num_channels;
  dims.rows = dims_->rows;
  dims.cols = dims_->cols;

  int arr_size = dims.num_images * dims.num_channels * dims.rows * dims.cols * sizeof(float);
  if (!gpu_memory_)
    data = (float*) malloc(arr_size);
  else
    cudaMalloc((void **)&data, arr_size);

  gpu_memory = gpu_memory_;
}

// Destructor frees data array
Tensor::~Tensor() {
  free(data);
}

// Getters + setters for convenience
float Tensor::get(int a, int b, int c, int d) {
  int num_channels = dims.num_channels;
  int rows = dims.rows;
  int cols = dims.cols;

  return data[((a * num_channels + b) * rows + c) * cols + d];
}
void Tensor::set(int a, int b, int c, int d, float val) {
  int num_channels = dims.num_channels;
  int rows = dims.rows;
  int cols = dims.cols;

  data[((a * num_channels + b) * rows + c) * cols + d] = val;
}