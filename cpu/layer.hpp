#ifndef LAYER_H
#define LAYER_H

#include "tensor.hpp"
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <random>

#define RELU 0
#define SIGMOID 1


class Activation {
public:
  int type;
  Activation(int type_);
  float activ(float x);
  float deriv(float x);
};



class Layer {
public:
  virtual void forward_prop(Tensor * input, Tensor * output) = 0;
  virtual void back_prop(Tensor * input_error, Tensor * output_error, float eta) = 0;
  virtual void output_dim(Dimensions * input_dims, Dimensions * output_dims) = 0;
  virtual void free_layer() = 0;
};



/**
 * Implement zero-padded convolutions!
 */ 
class ConvLayer : public Layer {
public:
  int stride;
  int size; // conv filter size
  int num_filters; // number of filters
  Activation * activation;
  float *** weights;
  float * biases;
  
  ConvLayer(int num_filters_, int size_, int stride_, int act_type);
  void forward_prop(Tensor * input, Tensor * output);
  void back_prop(Tensor * input_grad, Tensor * output_grad, float eta);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
};

class PoolingLayer : public Layer {

  // Use for backprop
  Tensor switches_X;
  Tensor switches_Y;

public: 
  int pool_size;
  int stride;

  PoolingLayer(int pool_size_, int stride_);
  void forward_prop(Tensor * input, Tensor * output);
  void back_prop(Tensor * input_grad, Tensor * output_grad, float eta);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
};



class FullyConnectedLayer : public Layer {
  // Use for backprop
  Tensor * last_input;
  Tensor * last_output;
  
public:
  int num_neurons;
  int input_dim;
  Activation * activation;
  float ** weights;
  float * biases;
  // stores gradients for minibatch of images
  Tensor * weight_grads;  
  Tensor * bias_grads;

  FullyConnectedLayer(int num_neurons_, int input_dim_, int act_type);
  void forward_prop(Tensor * input, Tensor * output);
  void back_prop(Tensor * input_error, Tensor * output_error, float eta);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();

  // flatten inputs
  void flatten(Tensor * input, Tensor * reshaped);
};

#endif
