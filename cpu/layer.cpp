/* 
 * Copied some of the dimension calculations from here:
 *   http://cs231n.github.io/convolutional-networks/
 */

#include "layer.hpp"

#include <math.h>
#include <algorithm>



ConvLayer::ConvLayer(int num_filters_, int size_, int stride_) {
  num_filters = num_filters_;
  size = size_;
  stride = stride_;
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
                value += input->vals[n][c][i2][j2] * weights[f][i2 - min_i][j2 - min_j];
          output->vals[n][f][i][j] = value;
        }
      }
}

void ConvLayer::back_prop(float **** input_grad,
        Dimensions * input_dimensions,
        float **** output_grad,
        Dimensions * output_dimensions) 
{
  
}

void ConvLayer::output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions)
{
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

  output_dimensions->num_images = num_images;
  output_dimensions->num_channels = num_filters;
  output_dimensions->dimX = (dimX - size) / stride + 1;
  output_dimensions->dimY = (dimY - size) / stride + 1;
}



/**
 * Stick to sigmoid for now
 */
ActivationLayer::ActivationLayer() {
  
}

void ActivationLayer::forward_prop(Tensor * input, Tensor * output) {
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;

  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++)
          output->vals[n][c][i][j] = 1.0 / (1.0 + exp(-input->vals[n][c][i][j]));

  // Save for backprop
  last_input = input;
}

void ActivationLayer::back_prop(float **** input_grad,
        Dimensions * input_dimensions,
        float **** output_grad,
        Dimensions * output_dimensions) 
{
  int num_images = output_dimensions->num_images;
  int num_channels = output_dimensions->num_channels;
  int dimX = output_dimensions->dimX;
  int dimY = output_dimensions->dimY;

  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++) {
          float s = 1.0 / (1.0 + exp(-last_input->vals[n][c][i][j]));
          s = s * (1-s);
          input_grad[n][c][i][j] = s * output_grad[n][c][i][j];
        }

  input_dimensions->num_images = num_images;
  input_dimensions->num_channels = num_channels; 
  input_dimensions->dimX = dimX;
  input_dimensions->dimY = dimY;
}

void ActivationLayer::output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions)
{
  output_dimensions->num_images = input_dimensions->num_images;
  output_dimensions->num_channels = input_dimensions->num_channels;
  output_dimensions->dimX = input_dimensions->dimX;
  output_dimensions->dimY = input_dimensions->dimY;
}



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
  Dimensions out_dims = {out_num_images, out_num_channels, out_dimX, out_dimY};
  switches_X.init_vals(&out_dims);
  switches_Y.init_vals(&out_dims);

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
              if (input->vals[n][c][i2][j2] > max_value) {
                max_value = input->vals[n][c][i2][j2];
                max_X = i2;
                max_Y = j2;
              }
          output->vals[n][c][i][j] = max_value;
          switches_X.vals[n][c][i][j] = max_X;
          switches_Y.vals[n][c][i][j] = max_Y;
        }
      }
}

void PoolingLayer::back_prop(float **** input_grad,
           Dimensions * input_dimensions,
           float **** output_grad,
           Dimensions * output_dimensions) 
{
  int num_images = output_dimensions->num_images;
  int num_channels = output_dimensions->num_channels;
  int dimX = output_dimensions->dimX;
  int dimY = output_dimensions->dimY;

  int input_num_images = last_input_dimensions->num_images;
  int input_num_channels = last_input_dimensions->num_channels;
  int input_dimX = last_input_dimensions->dimX;
  int input_dimY = last_input_dimensions->dimY;

  // Zero out
  for (int n = 0; n < input_num_images; n++)
    for (int c = 0; c < input_num_channels; c++)
      for (int i = 0; i < input_dimX; i++)
        for (int j = 0; j < input_dimY; j++) {
          input_grad[n][c][i][j] = 0;
        }

  // Bprop based on switches
  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++) {
          int max_X = switches_X.vals[n][c][i][j];
          int max_Y = switches_Y.vals[n][c][i][j];
          input_grad[n][c][max_X][max_Y] = output_grad[n][c][i][j];
        }  
  input_dimensions->num_images = input_num_images;
  input_dimensions->num_channels = input_num_channels; 
  input_dimensions->dimX = input_dimX;
  input_dimensions->dimY = input_dimY;
}

void PoolingLayer::output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions)
{
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

  output_dimensions->num_images = num_images;
  output_dimensions->num_channels = num_channels;
  output_dimensions->dimX = (dimX - pool_size) / stride + 1;
  output_dimensions->dimY = (dimY - pool_size) / stride + 1;
}



/**
 * TODO: Initializing - small normal RV's?????
 */
FullyConnectedLayer::FullyConnectedLayer(int num_neurons_, int input_dim_) {
  num_neurons = num_neurons_;
  input_dim = input_dim_;
  weights = new float*[num_neurons];
  for (int i = 0; i < num_neurons; i++) {
    weights[i] = new float[input_dim];
  }
  biases = new float*[num_neurons];
}

void FullyConnectedLayer::forward_prop(Tensor * input, Tensor * output) {
  // reshape input
  int num_images = input->dims->num_images;
  int num_channels = input->dims->num_channels;
  int dimX = input->dims->dimX;
  int dimY = input->dims->dimY;

  Dimensions reshaped_dims;
  reshaped_dims.num_images = num_images;
  reshaped_dims.num_channels = 1;
  reshaped_dims.dimX = num_channels * dimX * dimY;
  reshaped_dims.dimY = 1;

  Tensor * reshaped = new Tensor();
  reshaped->init_vals(&reshaped_dims);
  flatten(input, reshaped);
  // input = reshaped;

  for (int i = 0; i < num_images; i++) {
    for (int n = 0; n < num_neurons; n++) {
      float sum = 0;
      for (int j = 0; j < reshaped_dims.dimX; j++) {
        sum += weights[n][j] * reshaped->vals[i][0][j][0];
      }
      output->vals[i][0][n][0] = sum;
    }
  }

  reshaped->free_vals();
  delete reshaped;
}

/** Get weights/bias gradients */
void FullyConnectedLayer::back_prop(float **** input_grad,
            Dimensions * input_dimensions,
            float **** output_grad,
            Dimensions * output_dimensions) 
{
  
}

void FullyConnectedLayer::output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions)
{
  // reshape input
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

  output_dimensions->num_images = num_images;
  output_dimensions->num_channels = 1;
  output_dimensions->dimX = num_neurons;
  output_dimensions->dimY = 1;
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
          int ind = c * dimX * dimY + x * dimY + y;
          reshaped->vals[i][0][ind][0] = input->vals[i][c][x][y];
        }
      }
    }
  }
}
