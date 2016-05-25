/* 
 * Copied some of the dimension calculations from here:
 *   http://cs231n.github.io/convolutional-networks/
 */

#include "layer.hpp"
#include "utils.hpp"

#include <algorithm>
 #include <iostream>



ConvLayer::ConvLayer(int num_filters_, int size_, int stride_, int act_type) {
  num_filters = num_filters_;
  size = size_;
  stride = stride_;
  activation = new Activation(act_type);
  weights = new float **[num_filters];
  for (int f = 0; f < num_filters; f++) {
    weights[f] = new float *[size];
    for (int i = 0; i < size; i++)
      weights[f][i] = new float[size];
  }
  biases = new float[num_filters];  
  
  // Initialization
  for (int f = 0; f < num_filters; f++) 
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        weights[f][i][j] = 0.1;
  for (int f = 0; f < num_filters; f++)
    biases[f] = 1.0;
}

void ConvLayer::forward_prop(Tensor * input, Tensor * output) {
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;

  int out_num_images = num_images;
  int out_num_channels = num_filters;
  int out_dimX = (dimX - size) / stride + 1;
  int out_dimY = (dimY - size) / stride + 1;

  for (int n = 0; n < out_num_images; n++)
    for (int f = 0; f < out_num_channels; f++)
      for (int i = 0; i < out_dimX; i++) {
        int min_i = i * stride;
        int max_i = std::min(min_i + size, dimX);
        for (int j = 0; j < out_dimY; j++) {
          int min_j = j * stride;
          int max_j = std::min(min_j + size, dimY);

          // Apply convolution
          float value = biases[f];
          for (int c = 0; c < num_channels; c++)
            for (int i2 = min_i; i2 < max_i; i2++)
              for (int j2 = min_j; j2 < max_j; j2++)
                value += input->get(n, c, i2, j2) * weights[f][i2 - min_i][j2 - min_j];
          output->set(n, f, i, j, value);
        }
      }
}

void ConvLayer::back_prop(Tensor * input_error, Tensor * output_error, float eta) {
  
}

void ConvLayer::output_dim(Dimensions * input_dims, Dimensions * output_dims) {
  int num_images = input_dims->num_images;
  int num_channels = input_dims->num_channels;
  int dimX = input_dims->dimX;
  int dimY = input_dims->dimY;

  output_dims->num_images = num_images;
  output_dims->num_channels = num_filters;
  output_dims->dimX = (dimX - size) / stride + 1;
  output_dims->dimY = (dimY - size) / stride + 1;
}

void ConvLayer::free_layer()
{
  free(activation);
  for (int f = 0; f < num_filters; f++) {
    for (int i = 0; i < size; i++) {
      free(weights[f][i]);
    }
    free(weights[f]);
  }
  free(weights);
  free(biases);
}



Activation::Activation(int type_)
{
  type = type_;
}

float Activation::activ(float x)
{
  if (type == RELU) { 
    if (x > 0) {
      return x;
    }
    return 0;
  }
  else if (type == SIGMOID) {
    return sigmoid(x);
  }

  return -1;
}
  
// for backprop-ing errors
float Activation::deriv(float x)
{
  if (type == RELU) {
    if (x > 0) {
      return 1;
    }
    return 0;
  }
  else if (type == SIGMOID) { 
    return sigmoid_prime(x);
  }
  
  return -1;
}

// float ActivationLayer::ActivationLayer(int type_)
// {
//   type = type_;
// }

// float ActivationLayer::activation(float x)
// {
//   if (type == 0) { // ReLU
//     if (x > 0) {
//       return x;
//     }
//     return 0;
//   }
//   else if (type == 1) { // sigmoid
//     return sigmoid(x);
//   }
// }

// // for backprop-ing errors
// ActivationLayer::deriv(float x)
// {
//   if (type == 0) { // ReLU
//     if (x > 0) {
//       return 1;
//     }
//     return 0;
//   }
//   else if (type == 1) { // sigmoid
//     return sigmoid_prime(x)
//   }
  
// }

// void ActivationLayer::forward_prop(Tensor * input, Tensor * output)
// {
//   int num_images = input->dims->num_images;
//   int num_channels = input->dims->num_channels;
//   int dimX = input->dims->dimX;
//   int dimY = input->dims->dimY;

//   for (int n = 0; n < num_images; n++)
//     for (int c = 0; c < num_channels; c++)
//       for (int i = 0; i < dimX; i++)
//         for (int j = 0; j < dimY; j++)
//           output->set(n, c, i, j, activation(input->get(n, c, i, j)));

//   // Save for backprop
//   last_input = input;
// }

// void ActivationLayer::back_prop(Tensor * input_grad,
// 				Tensor * output_grad)  
// {
//   int num_images = output_grad->dims->num_images;
//   int num_channels = output_grad->dims->num_channels;
//   int dimX = output_grad->dims->dimX;
//   int dimY = output_grad->dims->dimY;

//   for (int n = 0; n < num_images; n++)
//     for (int c = 0; c < num_channels; c++)
//       for (int i = 0; i < dimX; i++)
//         for (int j = 0; j < dimY; j++) {
//           float s = deriv(last_input->get(n, c, i, j));
//           input_grad[n][c][i][j] = s * output_grad[n][c][i][j];
//         }

//   input_dimensions->num_images = num_images;
//   input_dimensions->num_channels = num_channels; 
//   input_dimensions->dimX = dimX;
//   input_dimensions->dimY = dimY;
// }

// void ActivationLayer::output_dim(Dimensions * input_dimensions, 
//       Dimensions * output_dimensions)
// {
//   output_dimensions->num_images = input_dimensions->num_images;
//   output_dimensions->num_channels = input_dimensions->num_channels;
//   output_dimensions->dimX = input_dimensions->dimX;
//   output_dimensions->dimY = input_dimensions->dimY;
// }

// void ActivationLayer::free_layer()
// {
  
// }



/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int pool_size_, int stride_) {
  pool_size = pool_size_;
  stride = stride_;
}

void PoolingLayer::forward_prop(Tensor * input, Tensor * output) {
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;

  int out_num_images = num_images;
  int out_num_channels = num_channels;
  int out_dimX = (dimX - pool_size) / stride + 1;
  int out_dimY = (dimY - pool_size) / stride + 1;

  // Init switches for use in backprop
  Dimensions * out_dims = new Dimensions{out_num_images, out_num_channels, out_dimX, out_dimY};
  switches_X.init_vals(out_dims);
  switches_Y.init_vals(out_dims);

  for (int n = 0; n < out_num_images; n++)
    for (int c = 0; c < out_num_channels; c++)
      for (int i = 0; i < out_dimX; i++) {
        int min_i = i * stride;
        int max_i = std::min(min_i + pool_size, dimX);
        for (int j = 0; j < out_dimY; j++) {
          int min_j = j * stride;
          int max_j = std::min(min_j + pool_size, dimY);

          // Find max value over the pooling area
          float max_value = -FLT_MAX;
          int max_X = -1;
          int max_Y = -1;
          for (int i2 = min_i; i2 < max_i; i2++)
            for (int j2 = min_j; j2 < max_j; j2++)
              if (input->get(n, c, i2, j2) > max_value) {
                max_value = input->get(n, c, i2, j2);
                max_X = i2;
                max_Y = j2;
              }
          output->set(n, c, i, j, max_value);
          switches_X.set(n, c, i, j, max_X);
          switches_Y.set(n, c, i, j, max_Y);
        }
      }
}

/**
* Propagates errors through max pooling layer (i.e. to max points in prev layer)
 */
void PoolingLayer::back_prop(Tensor * input_grad, Tensor * output_grad, float eta) {
  int num_images = output_grad->dims->num_images;
  int num_channels = output_grad->dims->num_channels;
  int dimX = output_grad->dims->dimX;
  int dimY = output_grad->dims->dimY;

  // Bprop based on switches
  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++) {
          int max_X = switches_X.get(n, c, i, j);
          int max_Y = switches_Y.get(n, c, i, j);
          input_grad->set(n, c, max_X, max_Y, output_grad->get(n, c, i, j));
        }

  // Free switches
  switches_X.free_vals();
  switches_Y.free_vals();
}

void PoolingLayer::output_dim(Dimensions * input_dims, Dimensions * output_dims) {
  int num_images = input_dims->num_images;
  int num_channels = input_dims->num_channels;
  int dimX = input_dims->dimX;
  int dimY = input_dims->dimY;

  output_dims->num_images = num_images;
  output_dims->num_channels = num_channels;
  output_dims->dimX = (dimX - pool_size) / stride + 1;
  output_dims->dimY = (dimY - pool_size) / stride + 1;
}

void PoolingLayer::free_layer()
{
  
}

/**
 * TODO: Initializing - small normal RV's?????
 */
FullyConnectedLayer::FullyConnectedLayer(int num_neurons_, int input_dim_,
					 int act_type)
{
  std::default_random_engine gen;
  std::normal_distribution<float> dist(0.0, 1.0);
  num_neurons = num_neurons_;
  input_dim = input_dim_;
  activation = new Activation(act_type);
  weights = new float*[num_neurons];
  for (int i = 0; i < num_neurons; i++) {
    weights[i] = new float[input_dim];

  }
  for (int i = 0; i < num_neurons; i++) {
    for (int n = 0; n < input_dim; n++) {
      weights[i][n] = dist(gen);
    }
  }
  biases = new float[num_neurons];
  weight_grads = new Tensor();
  bias_grads = new Tensor();
}

void FullyConnectedLayer::forward_prop(Tensor * input, Tensor * output) {
  last_input = input;
  last_output = output;
  
  // reshape input
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;

  Dimensions * reshaped_dims = new Dimensions{num_images, 1, num_channels * dimX * dimY, 1};
  Tensor * reshaped = new Tensor();
  reshaped->init_vals(reshaped_dims);
  flatten(input, reshaped);
  // input = reshaped;

  for (int i = 0; i < num_images; i++) {
    for (int n = 0; n < num_neurons; n++) {
      float sum = 0;
      for (int j = 0; j < reshaped_dims->dimX; j++) {
        sum += weights[n][j] * reshaped->get(i, 0, j, 0);
      }
      output->set(i, 0, n, 0, activation->activ(sum));
    }
  }

  reshaped->free_vals();
  delete reshaped_dims;
  delete reshaped;
}

/** Get weights/bias gradients */
void FullyConnectedLayer::back_prop(Tensor * input_error,
				    Tensor * output_error,
				    float eta)  
{
  int num_images = input_error->dims->num_images;
  int num_channels = last_input->dims->num_channels;
  int dimX = last_input->dims->dimX;
  int dimY = last_input->dims->dimY;
  int input_neurons = input_error->dims->dimX;
  int output_neurons = output_error->dims->dimX;
  // compute error for previous layer
  for (int i = 0; i < num_images; i++) {
    for (int n = 0; n < input_neurons; n++) {
      float error = 0;
      for(int o = 0; o < output_neurons; o++) {
	error += weights[o][n] * output_error->get(i, 0, o, 0);
      }
      float s = last_output->get(i, 0, n, 0);
      error *= s * (1 - s);
      // unflatten
      int y = n % dimY;
      int x = (n / dimY) % dimX;
      int c = (n / dimY) / dimX;      
      input_error->set(i, c, x, y, error); 
    }
  }

  // calculate current weight gradients
  for (int i = 0; i < num_images; i++) {
    for (int o = 0; o < output_neurons; o++) {      
      for (int n = 0; n < input_neurons; n++) {
	weight_grads->set(i, 0, o, n, last_input->get(i, 0, n, 0) *	\
			  output_error->get(i, 0, o, 0)); 
      }      
    }
  }
  // update weights
  for (int i = 0; i < num_images; i++) {
    for (int o = 0; o < output_neurons; o++) {
      for (int n = 0; n < input_neurons; n++) {
	weights[o][n] -= eta * weight_grads->get(i, 0, o, n);
      }
    }
  }  
  // calculate bias gradients (just error)
  for (int i = 0; i < num_images; i++) {
    for (int o = 0; o < output_neurons; o++) {
      bias_grads->set(i, 0, o, 0, output_error->get(i, 0, o, o) / num_images);
    }
  }
  // update biases
  for (int i = 0; i < num_images; i++) {
    for (int o = 0; o < output_neurons; o++) {
      biases[o] -= eta * bias_grads->get(i, 0, o, 0) / num_images;
    }
  }
  
}


void FullyConnectedLayer::output_dim(Dimensions * input_dims, Dimensions * output_dims) {
  // reshape input
  int num_images = input_dims->num_images;
  int num_channels = input_dims->num_channels;
  int dimX = input_dims->dimX;
  int dimY = input_dims->dimY;

  output_dims->num_images = num_images;
  output_dims->num_channels = 1;
  output_dims->dimX = num_neurons;
  output_dims->dimY = 1;
}

void FullyConnectedLayer::flatten(Tensor * input, Tensor * reshaped)
{
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;
  for (int i = 0; i < num_images; i++) {
    for (int c = 0; c < num_channels; c++) {
      for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
          int ind = ((c * dimX + x) * dimY) + y;
          reshaped->set(i, 0, ind, 0, input->get(i, c, x, y));
        }
      }
    }
  }
}

void FullyConnectedLayer::free_layer()
{
  free(activation);
  for (int n = 0; n < num_neurons; n++) {
    free(weights[n]);
  }
  weight_grads->free_vals();
  bias_grads->free_vals();
  free(weight_grads);
  free(bias_grads);
  free(weights);
  free(biases);
}
