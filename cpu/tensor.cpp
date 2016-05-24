#include "tensor.hpp"



Tensor::Tensor() {

}

void Tensor::init_vals(Dimensions * dims_) {
  dims = dims_;
  int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;
  vals = new float ***[num_images];
  for (int i = 0; i < num_images; i++) {
    vals[i] = new float **[num_channels];
    for (int c = 0; c < num_channels; c++) {
      vals[i][c] = new float *[dimX];
      for (int x = 0; x < dimX; x++) {
        vals[i][c][x] = new float[dimY];
      }
    }
  }
}

void Tensor::free_vals() {
  int num_images = dims->num_images;
  int num_channels = dims->num_channels;
  int dimX = dims->dimX;
  int dimY = dims->dimY;
  for (int n = 0; n < num_images; n++) {
    for (int c = 0; c < num_channels; c++) {
      for (int i = 0; i < dimX; i++) {
        delete vals[n][c][i];
      }
      delete vals[n][c];
    }
    delete vals[n];
  }
  delete vals;
}