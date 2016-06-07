#include "layer.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <cublas_v2.h>

#include <iostream>

// TODO: currently hard coded as sigmoid
__device__ float sactiv(float val) {
  return 1.0 / (1.0 + expf(-val));
}
__device__ float sderiv(float val) {
  float f = 1.0 / (1.0 + expf(-val));;
  return f * (1 - f);
}


FullyConnectedLayer::FullyConnectedLayer(int num_neurons_, int input_dim_) {  
  num_neurons = num_neurons_;
  input_dim = input_dim_;

  // Create cuBLAS context
  cublasCreate(&handle);
}

FullyConnectedLayer::~FullyConnectedLayer() {
  cublasDestroy(handle);
}


__global__
void cudaActivationKernel(float * dev_data,
			  float * dev_biases,
			  int length)
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  while (ind < length) {
    dev_data[ind] = sactiv(dev_data[ind] + dev_biases[ind]);
    ind += blockDim.x * gridDim.x;
  }
  
}

// computes error for previous layer
__global__
void cudaFCLayerBprop1Kernel(float * dev_input_grad_data,
			     float * dev_last_input_data,
			     int output_neurons) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  float a = dev_last_input_data[ind];
  dev_input_grad_data[ind] *= a * (1 - a);
}

// computes weight gradients and applies SGD updates
__global__
void cudaFCLayerBprop2Kernel(float * dev_weights_data,			     
			     float * dev_last_input_data,
			     float * dev_output_grad_data,
			     int output_neurons,
			     int input_neurons,
			     int num_images,
			     float eta) {
  int image_id = blockIdx.x / (blockDim.x / num_images);
  int block_id = blockIdx.x / num_images;
  int in_ind = block_id * blockDim.x + blockIdx.y;  
  int out_ind = blockIdx.z * blockDim.z + threadIdx.x;

  float c = eta / num_images;
  int weight_index = in_ind * output_neurons + out_ind;
  if(in_ind < input_neurons && out_ind < output_neurons) {
    // update appropriate weight with activation in prev_input_data
    dev_weights_data[weight_index] -= c * dev_last_input_data[image_id * input_neurons + in_ind] * dev_output_grad_data[image_id * output_neurons + out_ind];
    // dev_weights_data[weight_index] -= eta * dev_last_input_data[in_ind] * dev_output_grad_data[out_ind];
    
  }

  // int image_id = blockIdx.x;
  // int in_ind = blockIdx.y;  
  // int out_ind = threadIdx.x;

  // float c = eta / num_images;
  // int weight_index = in_ind * output_neurons + out_ind;
  // if(in_ind < input_neurons && out_ind < output_neurons) {
  //   // update appropriate weight with activation in prev_input_data
  //   dev_weights_data[weight_index] -= c * dev_last_input_data[image_id * input_neurons + in_ind] * dev_output_grad_data[image_id * output_neurons + out_ind];
    
  //   // dev_weights_data[weight_index] -= eta * dev_last_input_data[in_ind] * dev_output_grad_data[out_ind] / num_images;   
    
  // }
}

// computes bias gradients and applies SGD updates
__global__
void cudaFCLayerBprop3Kernel(float * dev_output_grad_data,
			     float * dev_biases_data,
			     int num_images,
			     int num_neurons,
			     float eta) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  float c = eta / num_images;
  if (ind < num_neurons) {
    // TODO: check output grad and biases are structured identically
    for (int i = 0; i < num_images; i++) {
      int ind2 = ind + i * num_neurons;
      dev_biases_data[ind] -= c * dev_output_grad_data[ind2];
    }
  }
  
}

void FullyConnectedLayer::fprop(Tensor * dev_input_, Tensor ** dev_output_) {
  // Store input for bprop
  dev_last_input = dev_input_;

  // input dimensions are fixed, so we don't have to reshape 1d tensor array  
  // Setup dev_D for cuBLAS
  float **D;
  D = (float **) malloc(sizeof(float *) * dev_input_->dims.num_images);
  for (int i = 0; i < dev_input_->dims.num_images; i++)
    D[i] = dev_input_->data + i * input_dim;
  cudaMemcpy(dev_D, D, sizeof(float *) * dev_input_->dims.num_images,
	     cudaMemcpyHostToDevice);
  free(D);

  // matrix multiply input with weight vector
  // dev_weights x dev_input_->data -> dev_output_->data
  int m = num_neurons;
  int n = 1;
  int k = input_dim;
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
		     (const float **) dev_D, ldb,
		     &beta,
		     dev_E, ldc,
		     batchSize);

  // add biases, perform activations
  int blocks = num_neurons / 1024 + 1;
  int threadsPerBlock = 1024;
  cudaActivationKernel<<<blocks, threadsPerBlock >>>(
    dev_output->data,
    dev_biases->data,
    num_neurons * dev_input_->dims.num_images);

  *dev_output_ = dev_output;
}

void FullyConnectedLayer::bprop(Tensor ** dev_input_grad_,
				Tensor * dev_output_grad_,
				float eta) 
{
  int num_images = dev_input_grad->dims.num_images;


  int b_in = input_dim / 1024 + 1;
  int t_in = 1024;
  int b_out = num_neurons / 1024 + 1;
  int t_out = 1024;
  
  // Setup dev_B for cuBLAS
  float **B;
  B = (float **) malloc(sizeof(float *) * num_images);
  for (int i = 0; i < num_images; i++)
    B[i] = dev_output_grad_->data + i * num_neurons;
  cudaMemcpy(dev_B, B, sizeof(float *) * num_images, cudaMemcpyHostToDevice);
  free(B);

  // compute error for previous layer
  // compute weights^T * output error = output_error * weights 
  int m = 1;
  int n = input_dim;
  int k = num_neurons;
  int lda = m;
  int ldb = k;
  int ldc = m;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int batchSize = dev_output_grad_->dims.num_images;
  cublasSgemmBatched(handle,
		     CUBLAS_OP_N, CUBLAS_OP_N,
		     m, n, k,
		     &alpha,
		     (const float **) dev_B, lda,
		     (const float **) dev_A, ldb,
		     &beta,
		     dev_C, ldc,
		     batchSize);  
  
  // Errors for previous layer
  cudaFCLayerBprop1Kernel<<<num_images, input_dim>>>(dev_input_grad->data,
						     dev_last_input->data,
						     num_neurons);

  // dim3 dimGrid(num_images, input_dim);
  // dim3 dimBlock(num_neurons);
  // cudaFCLayerBprop2Kernel<<<dimGrid, dimBlock>>>(dev_weights->data,
  // 						 dev_last_input->data,
  // 						 dev_output_grad_->data,
  // 						 num_neurons,
  // 						 input_dim,
  // 						 num_images,
  // 						 eta);
  
  //update weights for current layer
  dim3 dimGrid(num_images * b_in, t_in, b_out);
  dim3 dimBlock(t_out);
  cudaFCLayerBprop2Kernel<<<dimGrid, dimBlock>>>(dev_weights->data,
  						 dev_last_input->data,
  						 dev_output_grad_->data,
  						 num_neurons,
  						 input_dim,
  						 num_images,
  						 eta);
  
  // update biases for current layer
  cudaFCLayerBprop3Kernel<<<b_out, t_out>>>(dev_output_grad_->data,
					    dev_biases->data,
					    num_images,
					    num_neurons,
					    eta);

  *dev_input_grad_ = dev_input_grad;
}

void FullyConnectedLayer::get_output_dims(Dimensions * input_dims,
					  Dimensions * output_dims) {
  int num_images = input_dims->num_images;

  output_dims->num_images = num_images;
  output_dims->num_channels = 1;
  output_dims->rows = num_neurons;
  output_dims->cols = 1;
}

void FullyConnectedLayer::init_mem(Dimensions * input_dims) {
  
  // allocate memory for weights/biases
  dev_weights = new Tensor(1, input_dim, num_neurons, 1, true); // column major
  dev_biases = new Tensor(1, num_neurons, 1, 1, true);

  // initialize to random normal values
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateNormal(gen, dev_weights->data, num_neurons * input_dim,
		       0, 1);
  cudaMemset(dev_biases->data, 0, num_neurons);
  
  // Output
  Dimensions d;
  get_output_dims(input_dims, &d);
  dev_output = new Tensor(&d, true);
  dev_input_grad = new Tensor(input_dims, true);

  // cuBLAS batch processing
  cudaMalloc((void **)&dev_A, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_B, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_C, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_D, sizeof(float *) * input_dims->num_images);
  cudaMalloc((void **)&dev_E, sizeof(float *) * input_dims->num_images);
  float **A, **C, **E;
  A = (float **) malloc(sizeof(float *) * input_dims->num_images);
  C = (float **) malloc(sizeof(float *) * input_dims->num_images);
  E = (float **) malloc(sizeof(float *) * input_dims->num_images);
  for (int i = 0; i < input_dims->num_images; i++) {
    A[i] = dev_weights->data;
    C[i] = dev_input_grad->data + i * input_dim;
    E[i] = dev_output->data + i * num_neurons;
  }
  cudaMemcpy(dev_A, A, sizeof(float *) * input_dims->num_images,
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, sizeof(float *) * input_dims->num_images,
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dev_E, E, sizeof(float *) * input_dims->num_images,
	     cudaMemcpyHostToDevice);
  free(A);
  free(C);
  free(E);
}

void FullyConnectedLayer::free_mem() {
  delete dev_weights;
  delete dev_biases;

  delete dev_output;
  delete dev_input_grad;

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  cudaFree(dev_D);
  cudaFree(dev_E);
}
