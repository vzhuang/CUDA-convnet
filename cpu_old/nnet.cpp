#include <iostream>

#include "nnet.hpp"

#include "load.hpp"




NeuralNetwork::NeuralNetwork(Layer ** layers_, int num_layers_, 
      float **** X_, float ** Y_, Dimensions * X_dim_) 
{
  layers = layers_;
  num_layers = num_layers_;
  X = X_;
  Y = Y_;

  // Copy X dimensions into input for layer 1
  input_dims = new Dimensions[num_layers + 1];
  input_dims[0].num_images = X_dim_->num_images;
  input_dims[0].num_channels = X_dim_->num_channels;
  input_dims[0].dimX = X_dim_->dimX;
  input_dims[0].dimY = X_dim_->dimY;

  // Allocate memory for workspace
  make_workspace();
}

void NeuralNetwork::make_workspace() {
  workspace = new float****[num_layers + 1];
  for (int l = 0; l < num_layers; l++) {
    // Propagate forward to get new dimensions
    layers[l]->output_dim(&input_dims[l], &input_dims[l + 1]);
  }

  for (int l = 0; l < num_layers + 1; l++) {
    int num_images = input_dims[l].num_images;
    int num_channels = input_dims[l].num_channels;
    int dimX = input_dims[l].dimX;
    int dimY = input_dims[l].dimY;

    std::cout << "Layer " << l << ": " << num_images << " x "<< num_channels << " x "<< dimX << " x "<< dimY << std::endl;

    workspace[l] = new float***[num_images];
    for (int n = 0; n < num_images; n++) {
      workspace[l][n] = new float**[num_channels];
      for (int c = 0; c < num_channels; c++) {
        workspace[l][n][c] = new float*[dimX];
        for (int i = 0; i < dimX; i++) {
          workspace[l][n][c][i] = new float[dimY];
        }
      }
    }
  }
}

void NeuralNetwork::step() {
  int k = 0;

  visualize(X, k);
  std::cout << un_hot(Y[k]) << std::endl;

  // Get dimensions of X
  int num_images = input_dims[0].num_images;
  int num_channels = input_dims[0].num_channels;
  int dimX = input_dims[0].dimX;
  int dimY = input_dims[0].dimY;

  // Copy X to workspace
  for (int n = 0; n < num_images; n++) 
    for (int c = 0; c < num_channels; c++) 
      for (int i = 0; i < dimX; i++) 
        for (int j = 0; j < dimY; j++) 
          workspace[0][n][c][i][j] = X[n][c][i][j];

  visualize3(workspace[0], 0, 0, dimX, dimY);

  // Fprop all layers
  for (int l = 0; l < num_layers; l++) {
    layers[l]->forward_prop(workspace[l], &input_dims[l], workspace[l + 1], &input_dims[l + 1]);
    visualize3(workspace[l + 1], 0, 0, input_dims[l + 1].dimX, input_dims[l + 1].dimY);
  }

  // Result is in workspace[num_layers]
}
