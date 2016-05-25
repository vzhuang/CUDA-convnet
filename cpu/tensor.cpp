#include "tensor.hpp"

#include <iostream>

Tensor::Tensor() {

}

void Tensor::init_vals(Dimensions * dims_) {
  dims = dims_;
  int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;
  vals = new float[num_images * num_channels * dimX * dimY];
}

void Tensor::free_vals() {
  delete vals;
}

float Tensor::get(int a, int b, int c, int d) {
  //int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;

  return vals[((a * num_channels + b) * dimX + c) * dimY + d];
}

void Tensor::set(int a, int b, int c, int d, float val) {
  //int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;

  vals[((a * num_channels + b) * dimX + c) * dimY + d] = val;
}

void Tensor::zero_out() {
  int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;
  
  memset(vals, 0, sizeof(float) * num_images * num_channels * dimX * dimY);
}
