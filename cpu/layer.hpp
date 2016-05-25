#ifndef LAYER_H
#define LAYER_H

#include "tensor.hpp"
#include <random>

#define SOFTMAX 0
#define RELU 1


class Layer {
public:  
  virtual void forward_prop(Tensor * input, Tensor * output) = 0;
<<<<<<< HEAD
  virtual void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) = 0;
  virtual void output_dim(Dimensions * input_dimensions, 
			  Dimensions * output_dimensions) = 0;
=======
  virtual void back_prop(Tensor * input_grad, Tensor * output_grad) = 0;
  virtual void output_dim(Dimensions * input_dims, Dimensions * output_dims) = 0;
>>>>>>> b2051a1d3f9e7faf8009bc2e9bc88d842689d0a7
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
  void back_prop(Tensor * input_grad, Tensor * output_grad);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
};

class Activation {
public:
  int type;
  Activation(int type_);
  float activ(float x);
  float deriv(float x);
};

// class ActivationLayer : public Layer {

//   // Use for backprop
//   Tensor * last_input;

<<<<<<< HEAD
// public:
//   int type;
//   // activation types - ReLU, tanh, sigmoid?
//   ActivationLayer(int type);
//   float activation(float x);
//   float deriv(float x);
//   void forward_prop(Tensor * input, Tensor * output);
//   void back_prop(float **** input_grad, Dimensions * input_dimensions,
//       float **** output_grad, Dimensions * output_dimensions);
//   void output_dim(Dimensions * input_dimensions, 
// 		  Dimensions * output_dimensions);
//   void free_layer();
// };
=======
public: 
  // activation types - ReLU, tanh, sigmoid?
  ActivationLayer();
  void forward_prop(Tensor * input, Tensor * output);
  void back_prop(Tensor * input_grad, Tensor * output_grad);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
};
>>>>>>> b2051a1d3f9e7faf8009bc2e9bc88d842689d0a7



class PoolingLayer : public Layer {

  // Use for backprop
  Tensor switches_X;
  Tensor switches_Y;

public: 
  int pool_size;
  int stride;

  PoolingLayer(int pool_size_, int stride_);
  void forward_prop(Tensor * input, Tensor * output);
  void back_prop(Tensor * input_grad, Tensor * output_grad);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
};



class FullyConnectedLayer : public Layer {
  // Use for backprop
  Tensor * last_input;
  
public:
  int num_neurons;
  int input_dim;
  Activation * activation;
  float ** weights;
  float ** weights_grad;
  float * biases;
  float * biases_grad;

  FullyConnectedLayer(int num_neurons_, int input_dim_, int act_type);
  void forward_prop(Tensor * input, Tensor * output);
<<<<<<< HEAD
  void back_prop(Tensor * input_error, Tensor * output_error);
  void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions);
=======
  void back_prop(Tensor * input_grad, Tensor * output_grad);
  void output_dim(Dimensions * input_dims, Dimensions * output_dims);
  void free_layer();
>>>>>>> b2051a1d3f9e7faf8009bc2e9bc88d842689d0a7

  // flatten inputs
  void flatten(Tensor * input, Tensor * reshaped);
};

#endif
