#include "layer.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <cublas_v2.h>

#include <iostream>


/**
 * Max pooling of size by size region
 */
ConvLayer::ConvLayer(int num_filters_, int filter_size_, int stride_) {
  num_filters = num_filters_;
  filter_size = filter_size_;
  stride = stride_;

  dev_weights = new Tensor(1, num_filters, filter_size, filter_size, true);
  // dev_biases = new Tensor(1, num_filters, 1, 1, true);

  // Create cuBLAS handle for fprop
  cublasCreate(&handle);

  // Initialization . . .
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, dev_weights->data, num_filters * filter_size * filter_size);
}
ConvLayer::~ConvLayer() {
  delete dev_weights;
  // delete dev_biases;

  cublasDestroy(handle);
}


__global__ void ConvLayerStretchWeightsKernel(
  float * dev_weights_data, 
  float * dev_stretch_weights_data,
  int num_images) 
{
  int x = threadIdx.x;
  int y = blockIdx.x;

  float val = dev_weights_data[y * blockDim.x + x];
  const int step_size = blockDim.x * gridDim.x;

  for (int n = 0; n < num_images; n++)
    dev_stretch_weights_data[n * step_size + x * gridDim.x + y] = val;
}
__global__ void ConvLayerStretchInputKernel(
  float * dev_input_data, 
  float * dev_stretch_input_data,
  int input_num_images,
  int input_num_channels,
  int input_rows,
  int input_cols,
  int stride, 
  int filter_size) 
{
  int n = blockIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = blockIdx.x;

  int i = x * stride + (z / filter_size);
  int j = y * stride + (z % filter_size);

  const int stretch_index = n * blockDim.x * blockDim.y * gridDim.x + x * blockDim.y * gridDim.x + y * gridDim.x + z;
  const int input_index = n * input_num_channels * input_rows * input_cols + i * input_cols + j;
  const int step_size = input_rows * input_cols;

  dev_stretch_input_data[stretch_index] = 0;
  if (i < input_rows && j < input_cols)
    for (int c = 0; c < input_num_channels; c++)
      dev_stretch_input_data[stretch_index] += dev_input_data[input_index + c * step_size];
}
__global__ void ConvLayerUnStretchKernel(
  float * dev_stretch_output_data,
  float * dev_output_data) 
{
  int n = blockIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int c = blockIdx.x;

  const int stretch_index = n * blockDim.x * blockDim.y * gridDim.x + x * blockDim.y * gridDim.x + y * gridDim.x + c;
  const int unstretch_index = n * blockDim.x * blockDim.y * gridDim.x + c * blockDim.x * blockDim.y + x * blockDim.y + y;

  dev_output_data[unstretch_index] = dev_stretch_output_data[stretch_index];
}
void ConvLayer::fprop(Tensor * dev_input_, Tensor ** dev_output_) {
  // Stretch weights
  ConvLayerStretchWeightsKernel<<<num_filters, filter_size * filter_size>>>(
      dev_weights->data, 
      dev_stretch_weights->data, 
      dev_input_->dims.num_images);
  
  // Stretch input
  dim3 dimGrid(filter_size * filter_size, dev_input_->dims.num_images);
  dim3 dimBlock(dev_output->dims.rows, dev_output->dims.cols);
  ConvLayerStretchInputKernel<<<dimGrid, dimBlock>>>(
      dev_input_->data, 
      dev_stretch_input->data, 
      dev_input_->dims.num_images,
      dev_input_->dims.num_channels,
      dev_input_->dims.rows,
      dev_input_->dims.cols,
      stride,
      filter_size);

  // Matrix multiplication
  int m = num_filters;
  int n = dev_output->dims.rows * dev_output->dims.cols;
  int k = filter_size * filter_size;
  int lda = m;
  int ldb = k;
  int ldc = m;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int batchSize = dev_input_->dims.num_images;
  cublasStatus_t stat = cublasSgemmBatched(handle, 
                 CUBLAS_OP_N, CUBLAS_OP_N, 
                 m, n, k, 
                 &alpha, 
                 (const float **) dev_A, lda, 
                 (const float **) dev_B, ldb, 
                 &beta, 
                 dev_C, ldc,
                 batchSize);

  // Unstretch output
  dim3 dimGrid3(num_filters, dev_input_->dims.num_images);
  dim3 dimBlock3(dev_output->dims.rows, dev_output->dims.cols);
  ConvLayerUnStretchKernel<<<dimGrid3, dimBlock3>>>(
      dev_stretch_output->data, 
      dev_output->data);

  *dev_output_ = dev_output;
}


/**
 * Propagates errors through max pooling layer (i.e. to max points in prev layer)
 */
__global__ void ConvLayerBpropKernel(
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
void ConvLayer::bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta) {
  // cudaMemset(dev_input_grad->data, 0, dev_input_grad->dims.num_images * dev_input_grad->dims.num_channels * dev_input_grad->dims.rows * dev_input_grad->dims.cols * sizeof(float));

  // dim3 dimGrid(dev_output_grad_->dims.num_images, dev_output_grad_->dims.num_channels);
  // dim3 dimBlock(dev_output_grad_->dims.rows, dev_output_grad_->dims.cols);

  // ConvLayerBpropKernel<<<dimGrid, dimBlock>>>(
  //     dev_input_grad->data, 
  //     dev_output_grad_->data, 
  //     dev_switches_row->data, 
  //     dev_switches_col->data, 
  //     dev_input_grad->dims.num_channels, 
  //     dev_input_grad->dims.rows, 
  //     dev_input_grad->dims.cols);
  
  // *dev_input_grad_ = dev_input_grad;
}


void ConvLayer::get_output_dims(Dimensions * input_dims, Dimensions * output_dims) {
  output_dims->num_images = input_dims->num_images;
  output_dims->num_channels = num_filters;
  output_dims->rows = (input_dims->rows - filter_size) / stride + 1;
  output_dims->cols = (input_dims->cols - filter_size) / stride + 1;
}


void ConvLayer::init_mem(Dimensions * input_dims) {
  Dimensions d;
  get_output_dims(input_dims, &d);
  dev_output = new Tensor(&d, true);
  dev_input_grad = new Tensor(input_dims, true);

  // Weights is stretched to num_filters x (filter_size)^2
  // Input is stretched to (filter_size)^2 x (output rows * output cols)
  // CuBLAS is column major rather than row major though, hence the swap
  dev_stretch_weights = new Tensor(d.num_images, 1, filter_size * filter_size, num_filters, true);
  dev_stretch_input = new Tensor(d.num_images, 1, d.rows * d.cols, filter_size * filter_size, true);
  dev_stretch_output = new Tensor(&d, true);

  // cuBLAS batch processing
  cudaMalloc((void **)&dev_A, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_B, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_C, sizeof(float *) * input_dims->num_images);
  float **A, **B, **C;
  A = (float **) malloc(sizeof(float *) * input_dims->num_images);
  B = (float **) malloc(sizeof(float *) * input_dims->num_images);
  C = (float **) malloc(sizeof(float *) * input_dims->num_images);
  for (int i = 0; i < input_dims->num_images; i++) {
    A[i] = dev_stretch_weights->data + i * num_filters * filter_size * filter_size;
    B[i] = dev_stretch_input->data + i * d.rows * d.cols * filter_size * filter_size;
    C[i] = dev_stretch_output->data + i * d.num_channels * d.rows * d.cols;
  }
  cudaMemcpy(dev_A, A, sizeof(float *) * input_dims->num_images, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, sizeof(float *) * input_dims->num_images, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, sizeof(float *) * input_dims->num_images, cudaMemcpyHostToDevice);
  free(A);
  free(B);
  free(C);
}
void ConvLayer::free_mem() {
  delete dev_output;
  delete dev_input_grad;

  delete dev_stretch_weights;
  delete dev_stretch_input;
  delete dev_stretch_output;

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}

