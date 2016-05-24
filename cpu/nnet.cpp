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

  workspace = new Tensor[num_layers + 1];
  workspace[0].init_vals(X_dim_);

  // Allocate memory for workspace
  make_workspace();
}

void NeuralNetwork::make_workspace() {
  for (int l = 0; l < num_layers; l++) {
    // Propagate forward to get new dimensions
    Dimensions * new_dims = new Dimensions();
    layers[l]->output_dim(workspace[l].dims, new_dims);
    workspace[l + 1].init_vals(new_dims);
  }

  for (int l = 0; l < num_layers + 1; l++) {
    int num_images = workspace[l].dims->num_images;
    int num_channels = workspace[l].dims->num_channels;
    int dimX = workspace[l].dims->dimX;
    int dimY = workspace[l].dims->dimY;

    std::cout << "Layer " << l << ": " << num_images << " x "<< num_channels << " x "<< dimX << " x "<< dimY << std::endl;
  }
}

void NeuralNetwork::step() {
  int k = 0;

  // Get dimensions of X
  int num_images = workspace[0].dims->num_images;
  int num_channels = workspace[0].dims->num_channels;
  int dimX = workspace[0].dims->dimX;
  int dimY = workspace[0].dims->dimY;

  // Copy X to workspace
  for (int n = 0; n < num_images; n++) 
    for (int c = 0; c < num_channels; c++) 
      for (int i = 0; i < dimX; i++) 
        for (int j = 0; j < dimY; j++) 
          workspace[0].vals[n][c][i][j] = X[n][c][i][j];

  // Display X (input)
  visualize3(workspace[0].vals, 0, 0, dimX, dimY);

  // Fprop all layers
  for (int l = 0; l < num_layers; l++) {
    layers[l]->forward_prop(&workspace[l], &workspace[l + 1]);
    visualize3(workspace[l + 1].vals, 0, 0, workspace[l + 1].dims->dimX, workspace[l + 1].dims->dimY);
  }

  // Result is in workspace[num_layers]
}
