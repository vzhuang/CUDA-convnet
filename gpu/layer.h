#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "utils.h"
#include <cufft.h>
#include <cublas_v2.h>
#include <float.h>
#include <stdio.h>


/**
 * Abstract class.
 * 
 * Key functions:
 *   fprop - Forward propagates input to output
 *   bprop - Backward propagates gradient, does weight updates
 *   get_output_dims - Returns output dimensions given input dimensions
 *   init_mem - Allocates memory for fprop and bprop operations + outputs
 *   free_mem - Frees memory alocated in init_mem
 *
 * Weights, biases, etc. should be allocated in the constructor, as these
 * have dimensions that do not depend on the input size. The method init_mem()
 * should be used to initialize working memory (e.g. intermediate storage) as
 * well as memory for the output. All this should be GPU memory.
 */
class Layer {
public:
  virtual void fprop(Tensor * dev_input_, Tensor ** dev_output_) = 0;
  virtual void bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta) = 0;

  virtual void get_output_dims(Dimensions * input_dims, Dimensions * output_dims) = 0;

  virtual void init_mem(Dimensions * input_dims) = 0;
  virtual void free_mem() = 0;
};



class ConvLayer : public Layer {
  int num_filters;
  int filter_size;
  int stride;
  cublasHandle_t handle;

public:
  Tensor * dev_weights;   // Stored in column-major order!
  Tensor * dev_biases;

  // Memory for stretching input and storing output for convolution as matrix multiplication
  Tensor * dev_stretch_input;
  Tensor * dev_stretch_output;
  float **dev_A, **dev_B, **dev_C;

  // Gradients for weights, biases, x
  Tensor * dev_weights_grad;
  Tensor * dev_biases_grad;
  Tensor * dev_x_grad;

  Tensor * prev_input;

  // fprop output
  Tensor * dev_output;

  // bprop output
  Tensor * dev_input_grad;
  
  ConvLayer(int num_filters_, int filter_size_, int stride_);
  ~ConvLayer();
  
  void fprop(Tensor * dev_input_, Tensor ** dev_output_);
  void bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta);
  
  void get_output_dims(Dimensions * input_dims, Dimensions * output_dims);
  
  void init_mem(Dimensions * input_dims);
  void free_mem();
};


class PoolingLayer : public Layer {
  int pool_size;
  int stride;

public: 
  // Dimensions depend on input size
  Tensor * dev_switches_row;
  Tensor * dev_switches_col;

  // fprop output
  Tensor * dev_output;

  // bprop output
  Tensor * dev_input_grad;

  PoolingLayer(int pool_size_, int stride_);
  
  void fprop(Tensor * dev_input_, Tensor ** dev_output_);
  void bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta);
  
  void get_output_dims(Dimensions * input_dims, Dimensions * output_dims);
  
  void init_mem(Dimensions * input_dims);
  void free_mem();
};

class FullyConnectedLayer : public Layer {
  int num_neurons;
  int input_dim;
  cublasHandle_t handle;

  // used for cuBLAS
  float **dev_A;  // weights
  float **dev_B;  // output_error
  float **dev_C;  // input_error
  float **dev_D;  // input_data
  float **dev_E;  // output_data


public:
  // Weights, biases
  Tensor * dev_weights; // column major
  Tensor * dev_biases;

  // fprop output
  Tensor * dev_output;

  // bprop output
  Tensor * dev_input_grad;
  

  // Use for backprop
  Tensor * dev_last_input;
  
  FullyConnectedLayer(int num_neurons_, int input_dim_);
  ~FullyConnectedLayer();
  
  void fprop(Tensor * dev_input_, Tensor ** dev_output_);
  void bprop(Tensor ** dev_input_grad_, Tensor * dev_output_grad_, float eta);
  
  void get_output_dims(Dimensions * input_dims, Dimensions * output_dims);
  
  void init_mem(Dimensions * input_dims);
  void free_mem();
};


#endif 
