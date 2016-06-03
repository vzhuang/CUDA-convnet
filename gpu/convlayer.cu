#include "layer.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <cublas_v2.h>

#include <iostream>


/**
 * Convolutes images
 */
ConvLayer::ConvLayer(int num_filters_, int filter_size_, int stride_) {
  num_filters = num_filters_;
  filter_size = filter_size_;
  stride = stride_;

  // Create cuBLAS handle for fprop
  cublasCreate(&handle);
}
ConvLayer::~ConvLayer() {
  cublasDestroy(handle);
}


// __global__ void ConvLayerStretchWeightsKernel(
//   float * dev_weights_data, 
//   float * dev_stretch_weights_data,
//   int num_images) 
// {
//   int x = threadIdx.x;
//   int y = blockIdx.x;

//   float val = dev_weights_data[y * blockDim.x + x];
//   const int step_size = blockDim.x * gridDim.x;

//   for (int n = 0; n < num_images; n++)
//     dev_stretch_weights_data[n * step_size + x * gridDim.x + y] = val;
// }
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
  int n = blockIdx.z;   // dev_input_->dims.num_images
  int c = blockIdx.y;   // dev_input_->dims.num_channels
  int s = blockIdx.x;   // filter_size * filter_size

  int x = threadIdx.x;  // dev_output->dims.rows
  int y = threadIdx.y;  // dev_output->dims.cols

  int i = x * stride + (s / filter_size);
  int j = y * stride + (s % filter_size);

  // const int stretch_index = n * blockDim.x * blockDim.y * gridDim.x * gridDim.y + x * blockDim.y * gridDim.x + y * gridDim.x + s;
  const int stretch_index = n * blockDim.x * blockDim.y * gridDim.x * gridDim.y + x * blockDim.y * gridDim.x * gridDim.y + y * gridDim.x * gridDim.y + c * gridDim.x + s;
  const int input_index = n * input_num_channels * input_rows * input_cols + c * input_rows * input_cols + i * input_cols + j;

  float val = 0;
  if (i < input_rows && j < input_cols)
    val = dev_input_data[input_index];
  dev_stretch_input_data[stretch_index] = val;
}
__global__ void ConvLayerUnStretchKernel(
  float * dev_stretch_output_data,
  float * dev_output_data, 
  float * dev_biases_data) 
{
  int n = blockIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int c = blockIdx.x;

  const int stretch_index = n * blockDim.x * blockDim.y * gridDim.x + x * blockDim.y * gridDim.x + y * gridDim.x + c;
  const int unstretch_index = n * blockDim.x * blockDim.y * gridDim.x + c * blockDim.x * blockDim.y + x * blockDim.y + y;

  dev_output_data[unstretch_index] = dev_stretch_output_data[stretch_index] + dev_biases_data[c];
}
void ConvLayer::fprop(Tensor * dev_input_, Tensor ** dev_output_) {
  // // Stretch weights
  // ConvLayerStretchWeightsKernel<<<num_filters, dev_weights->dims.num_images * filter_size * filter_size>>>(
  //     dev_weights->data, 
  //     dev_stretch_weights->data, 
  //     dev_input_->dims.num_images);
  
  // Stretch input
  dim3 dimGrid(filter_size * filter_size, dev_input_->dims.num_channels, dev_input_->dims.num_images);
  dim3 dimBlock(dev_output->dims.rows, dev_output->dims.cols, 1);
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
  int k = dev_input_->dims.num_channels * filter_size * filter_size;
  int lda = m;
  int ldb = k;
  int ldc = m;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int batchSize = dev_input_->dims.num_images;
  cublasSgemmBatched(handle, 
                     CUBLAS_OP_N, CUBLAS_OP_N, 
                     m, n, k, 
                     &alpha, 
                     (const float **) dev_A, lda, 
                     (const float **) dev_B, ldb, 
                     &beta, 
                     dev_C, ldc,
                     batchSize);

  // Unstretch output + add biases
  dim3 dimGrid2(num_filters, dev_input_->dims.num_images);
  dim3 dimBlock2(dev_output->dims.rows, dev_output->dims.cols);
  ConvLayerUnStretchKernel<<<dimGrid2, dimBlock2>>>(
      dev_stretch_output->data, 
      dev_output->data,
      dev_biases->data);

  // Save input for bprop
  prev_input = dev_input_;

  *dev_output_ = dev_output;
}


/**
 * Propagates errors and does weight updates
 */
__global__ void ConvLayerBpropKernel(
  float * dev_input_grad_data, 
  float * dev_output_grad_data,
  float * dev_weights_grad_data,
  float * dev_biases_grad_data,
  float * dev_input_data,
  float * dev_weights_data,
  int filter_size,
  int stride,
  int input_num_channels,
  int input_rows,
  int input_cols) 
{
  int n = blockIdx.y;
  int f = blockIdx.x;
  int i = threadIdx.x;
  int j = threadIdx.y;

  int output_index = ((n * gridDim.x + f) * blockDim.x + i) * blockDim.y + j;
  float output_grad_value = dev_output_grad_data[output_index];

  int min_i = i * stride;
  int max_i = min_i + filter_size;
  if (input_rows < max_i)
    max_i = input_rows;

  int min_j = j * stride;
  int max_j = min_j + filter_size;
  if (input_cols < max_j)
    max_j = input_cols;

  // Propagate for all values
  for (int i2 = min_i; i2 < max_i; i2++)
    for (int j2 = min_j; j2 < max_j; j2++) 
      for (int channel = 0; channel < input_num_channels; channel++) {
        int input_index = ((n * input_num_channels + channel) * input_rows + i2) * input_cols + j2;
        int weight_index = ((channel * filter_size + i2 - min_i) * filter_size + j2 - min_j) * gridDim.x + f;

        dev_input_grad_data[input_index] += 1; //DEBUG dev_weights_data[weight_index] * output_grad_value;
        // dev_weights_grad_data[weight_index] += dev_input_data[input_index] * output_grad_value;
      }


  // int input_index = ((n * input_num_channels + 0) * input_rows + min_i) * input_cols + min_j;
  // dev_input_grad_data[input_index] = output_grad_value;


  // Normalize weight gradients (divide by n_imgs)
  // biases
}
void ConvLayer::bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta) {
  cudaMemset(dev_input_grad->data, 0, dev_input_grad->dims.num_images * dev_input_grad->dims.num_channels * dev_input_grad->dims.rows * dev_input_grad->dims.cols * sizeof(float));

  dim3 dimGrid(num_filters, dev_output_grad_->dims.num_images);
  dim3 dimBlock(dev_output_grad_->dims.rows, dev_output_grad_->dims.cols);
  ConvLayerBpropKernel<<<dimGrid, dimBlock>>>(
      dev_input_grad->data, 
      dev_output_grad_->data, 
      dev_weights_grad->data, 
      dev_biases_grad->data, 
      prev_input->data,
      dev_weights->data,
      filter_size,
      stride,
      dev_input_grad->dims.num_channels, 
      dev_input_grad->dims.rows, 
      dev_input_grad->dims.cols);
  
  // Update weights, biases

  *dev_input_grad_ = dev_input_grad;
}


void ConvLayer::get_output_dims(Dimensions * input_dims, Dimensions * output_dims) {
  output_dims->num_images = input_dims->num_images;
  output_dims->num_channels = num_filters;
  output_dims->rows = (input_dims->rows - filter_size) / stride + 1;
  output_dims->cols = (input_dims->cols - filter_size) / stride + 1;
}


void ConvLayer::init_mem(Dimensions * input_dims) {
  // Parameters
  dev_weights = new Tensor(1, input_dims->num_channels, filter_size * filter_size, num_filters, true);
  dev_biases = new Tensor(1, num_filters, 1, 1, true);

  // Initialization . . .
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, dev_weights->data, input_dims->num_channels * num_filters * filter_size * filter_size);

  // Gradients
  dev_weights_grad = new Tensor(1, input_dims->num_channels, filter_size * filter_size, num_filters, true);
  dev_biases_grad = new Tensor(1, num_filters, 1, 1, true);

  // Output
  Dimensions d;
  get_output_dims(input_dims, &d);
  dev_output = new Tensor(&d, true);
  dev_input_grad = new Tensor(input_dims, true);

  // Input is stretched to (filter_size)^2 x (output rows * output cols)
  // Output is temporarily stored before fixing into row major format
  // CuBLAS is column major rather than row major though, hence the swap
  dev_stretch_input = new Tensor(d.num_images, 1, d.rows * d.cols, input_dims->num_channels * filter_size * filter_size, true);
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
    A[i] = dev_weights->data;
    B[i] = dev_stretch_input->data + i * d.rows * d.cols * input_dims->num_channels * filter_size * filter_size;
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
  delete dev_weights;
  delete dev_biases;

  delete dev_weights_grad;
  delete dev_biases_grad;

  delete dev_output;
  delete dev_input_grad;

  delete dev_stretch_input;
  delete dev_stretch_output;

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}

