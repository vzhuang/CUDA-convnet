#include "layer.h"

#include <algorithm>
#include <cuda_runtime.h>

    #include <stdio.h>

/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int pool_size_, int stride_) {
  pool_size = pool_size_;
  stride = stride_;
}


__global__ void PoolingLayerFpropKernel(
  float * dev_input_data, 
  float * dev_output_data, 
  float * dev_switches_row_data,
  float * dev_switches_col_data,
  int pool_size, 
  int stride,
  int input_num_channels,
  int input_rows,
  int input_cols) 
{
  int n = blockIdx.x;
  int c = blockIdx.y;
  int i = threadIdx.x;
  int j = threadIdx.y;

  int index = ((n * gridDim.y + c) * blockDim.x + i) * blockDim.y + j;

  int min_i = i * stride;
  int max_i = min_i + pool_size;
  if (input_rows < max_i)
    max_i = input_rows;

  int min_j = j * stride;
  int max_j = min_j + pool_size;
  if (input_cols < max_j)
    max_j = input_cols;

  // Find max value over the pooling area
  float max_value = -FLT_MAX;
  int max_row = -1;
  int max_col = -1;
  for (int i2 = min_i; i2 < max_i; i2++)
    for (int j2 = min_j; j2 < max_j; j2++) {
      int index2 = ((n * input_num_channels + c) * input_rows + i2) * input_cols + j2;
      float val = dev_input_data[index2];
      if (val > max_value) {
        max_value = val;
        max_row = i2;
        max_col = j2;
      }
    }
  dev_output_data[index] = max_value;
  dev_switches_row_data[index] = max_row;
  dev_switches_col_data[index] = max_col;
}
void PoolingLayer::fprop(Tensor * dev_input_, Tensor ** dev_output_) {
  dim3 dimGrid(dev_output->dims.num_images, dev_output->dims.num_channels);
  dim3 dimBlock(dev_output->dims.rows, dev_output->dims.cols);

  PoolingLayerFpropKernel<<<dimGrid, dimBlock>>>(
      dev_input_->data, 
      dev_output->data, 
      dev_switches_row->data, 
      dev_switches_col->data, 
      pool_size, 
      stride, 
      dev_input_->dims.num_channels, 
      dev_input_->dims.rows, 
      dev_input_->dims.cols);
  
  *dev_output_ = dev_output;
}

/**
 * Propagates errors through max pooling layer (i.e. to max points in prev layer)
 */
__global__ void PoolingLayerBpropKernel(
  float * dev_input_grad_data, 
  float * dev_output_grad_data,
  float * dev_switches_row_data,
  float * dev_switches_col_data,
  int input_num_channels,
  int input_rows,
  int input_cols) 
{
  int n = blockIdx.x;
  int c = blockIdx.y;
  int i = threadIdx.x;
  int j = threadIdx.y;

  int index = ((n * gridDim.y + c) * blockDim.x + i) * blockDim.y + j;

  int max_row = dev_switches_row_data[index];
  int max_col = dev_switches_col_data[index];
  int index2 = ((n * input_num_channels + c) * input_rows + max_row) * input_cols + max_col;

  dev_input_grad_data[index2] = dev_output_grad_data[index];
}
void PoolingLayer::bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta) {
  cudaMemset(dev_input_grad->data, 0, dev_input_grad->dims.num_images * dev_input_grad->dims.num_channels * dev_input_grad->dims.rows * dev_input_grad->dims.cols * sizeof(float));

  dim3 dimGrid(dev_output_grad_->dims.num_images, dev_output_grad_->dims.num_channels);
  dim3 dimBlock(dev_output_grad_->dims.rows, dev_output_grad_->dims.cols);

  PoolingLayerBpropKernel<<<dimGrid, dimBlock>>>(
      dev_input_grad->data, 
      dev_output_grad_->data, 
      dev_switches_row->data, 
      dev_switches_col->data, 
      dev_input_grad->dims.num_channels, 
      dev_input_grad->dims.rows, 
      dev_input_grad->dims.cols);
  
  *dev_input_grad_ = dev_input_grad;
}

void PoolingLayer::get_output_dims(Dimensions * input_dims, Dimensions * output_dims) {
  output_dims->num_images = input_dims->num_images;
  output_dims->num_channels = input_dims->num_channels;
  output_dims->rows = (input_dims->rows - pool_size) / stride + 1;
  output_dims->cols = (input_dims->cols - pool_size) / stride + 1;
}

void PoolingLayer::init_mem(Dimensions * input_dims) {
  Dimensions d;
  get_output_dims(input_dims, &d);
  dev_output = new Tensor(&d, true);
  dev_input_grad = new Tensor(input_dims, true);
  dev_switches_row = new Tensor(&d, true);
  dev_switches_col = new Tensor(&d, true);
}

void PoolingLayer::free_mem() {
  delete dev_output;
  delete dev_input_grad;
  delete dev_switches_row;
  delete dev_switches_col;
}

