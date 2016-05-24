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

void ConvLayer::forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions)
{
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

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
                value += input[n][c][i2][j2] * weights[f][i2 - min_i][j2 - min_j];
          output[n][f][i][j] = value;
        }
      }
  
  output_dimensions->num_images = out_num_images;
  output_dimensions->num_channels = out_num_channels; 
  output_dimensions->dimX = out_dimX;
  output_dimensions->dimY = out_dimY; 
}

void ConvLayer::back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) 
{
  
}



/**
 * Stick to sigmoid for now
 */
ActivationLayer::ActivationLayer() {
  
}

void ActivationLayer::forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions)
{
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++)
          output[n][c][i][j] = 1.0 / (1.0 + exp(-input[n][c][i][j]));

  output_dimensions->num_images = num_images;
  output_dimensions->num_channels = num_channels; 
  output_dimensions->dimX = dimX;
  output_dimensions->dimY = dimY;

  // Save for backprop
  last_input = input;
  last_input_dimensions = input_dimensions;
}

void ActivationLayer::back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) 
{
  int num_images = output_dimensions->num_images;
  int num_channels = output_dimensions->num_channels;
  int dimX = output_dimensions->dimX;
  int dimY = output_dimensions->dimY;

  for (int n = 0; n < num_images; n++)
    for (int c = 0; c < num_channels; c++)
      for (int i = 0; i < dimX; i++)
        for (int j = 0; j < dimY; j++) {
          float s = 1.0 / (1.0 + exp(-last_input[n][c][i][j]));
          s = s * (1-s);
          input_grad[n][c][i][j] = s * output_grad[n][c][i][j];
        }

  input_dimensions->num_images = num_images;
  input_dimensions->num_channels = num_channels; 
  input_dimensions->dimX = dimX;
  input_dimensions->dimY = dimY;
}



/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int pool_size_, int stride_) {
  pool_size = pool_size_;
  stride = stride_;
}

void PoolingLayer::forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions)
{
  int num_images = input_dimensions->num_images;
  int num_channels = input_dimensions->num_channels;
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;

  int out_num_images = num_images;
  int out_num_channels = num_channels;
  int out_dimX = (dimX - pool_size) / stride + 1;
  int out_dimY = (dimY - pool_size) / stride + 1;

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
          for (int i2 = min_i; i2 < max_i; i2++)
            for (int j2 = min_j; j2 < max_j; j2++)
              if (input[n][c][i2][j2] > max_value)
                max_value = input[n][c][i2][j2];
          output[n][c][i][j] = max_value;
        }
      }
  
  output_dimensions->num_images = out_num_images;
  output_dimensions->num_channels = out_num_channels; 
  output_dimensions->dimX = out_dimX;
  output_dimensions->dimY = out_dimY; 
}

void PoolingLayer::back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) 
{
  
}



/**
 * Idk how to implement this
 */
FullyConnectedLayer::FullyConnectedLayer() {

}

void FullyConnectedLayer::forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions)
{

}

void FullyConnectedLayer::back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) 
{
  
}

float * FullyConnectedLayer::flatten() {
  return NULL;
}
