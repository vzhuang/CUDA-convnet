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
  for (int i = 0; i < num_filters; i++) {
    weights[i] = new float *[size];
    for (int j = 0; j < size; j++) {
      weights[i][j] = new float[size];
    }
  }
  biases = new float[num_filters];
    
}

void ConvLayer::forward_prop(float *** input, Dimensions * input_dimensions,
      float *** output, Dimensions * output_dimensions)
{
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;
  int dimZ = input_dimensions->dimZ;

  int out_dimX = dimX * num_filters;
  int out_dimY = (dimY - size) / stride + 1;
  int out_dimZ = (dimZ - size) / stride + 1;


  // TODO


  
  output_dimensions->dimX = out_dimX;
  output_dimensions->dimY = out_dimY;
  output_dimensions->dimZ = out_dimZ;  
}

void ConvLayer::back_prop() {
  
}



/**
 * Stick to sigmoid for now
 */
ActivationLayer::ActivationLayer() {
  
}

void ActivationLayer::forward_prop(float *** input, Dimensions * input_dimensions,
      float *** output, Dimensions * output_dimensions)
{
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;
  int dimZ = input_dimensions->dimZ;

  for (int i = 0; i < dimX; i++)
    for (int j = 0; j < dimY; j++)
      for (int k = 0; k < dimZ; k++)
        output[i][j][k] = 1.0 / (1.0 + exp(-input[i][j][k]));

  output_dimensions->dimX = dimX;
  output_dimensions->dimY = dimY;
  output_dimensions->dimZ = dimZ;
}

void ActivationLayer::back_prop() {
  
}



/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int pool_size_, int stride_) {
  pool_size = pool_size_;
  stride = stride_;
}


void PoolingLayer::forward_prop(float *** input, Dimensions * input_dimensions,
      float *** output, Dimensions * output_dimensions)
{
  int dimX = input_dimensions->dimX;
  int dimY = input_dimensions->dimY;
  int dimZ = input_dimensions->dimZ;

  int out_dimX = dimX;
  int out_dimY = (dimY - pool_size) / stride + 1;
  int out_dimZ = (dimZ - pool_size) / stride + 1;

  for (int i = 0; i < out_dimX; i++)
    for (int j = 0; j < out_dimY; j++) {
      int min_j = j * stride;
      int max_j = std::min(min_j + pool_size, dimY);
      for (int k = 0; k < out_dimZ; k++) {
        int min_k = k * stride;
        int max_k = std::min(min_k + pool_size, dimZ);

        // Find max value over the pooling area
        float max_value = -FLT_MAX;
        for (int j2 = min_j; j2 < max_j; j2++)
          for (int k2 = min_k; k2 < max_k; k2++)
            if (input[i][j2][k2] > max_value)
              max_value = input[i][j2][k2];
        output[i][j][k] = max_value;
      }
    }

  output_dimensions->dimX = out_dimX;
  output_dimensions->dimY = out_dimY;
  output_dimensions->dimZ = out_dimZ;  
}

void PoolingLayer::back_prop() {
  
}



