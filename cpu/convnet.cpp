#include "convnet.hpp"
#include "utils.hpp"

#include <iostream>
#include <time.h>



ConvNet::ConvNet(Layer ** layers_, int num_layers_, 
      Tensor * X_train_, float ** Y_train_)
{
  layers = layers_;
  num_layers = num_layers_;
  X_train = X_train_;
  Y_train = Y_train_;
}

void ConvNet::make_workspace() {
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

/**
 * Implements minibatch SGD for backpropagation
 *
 * Params:
 * eta: learning rate
 * num_batches: number of batches to run
 * batch_size: number of data_points in each batch
 */
void ConvNet::train(float eta, int num_batches, int batch_size) {
  // Get dimensions of X
  const int num_channels = X_train->dims->num_channels;
  const int dimX = X_train->dims->dimX;
  const int dimY = X_train->dims->dimY;

  // Allocate memory for workspace
  workspace = new Tensor[num_layers + 1];
  Dimensions * dims = new Dimensions{batch_size, num_channels, dimX, dimY};
  workspace[0].init_vals(dims);
  make_workspace();

  // Store selected training sample indices, Y, Y_pred
  int * indices = new int[batch_size];
  float ** Y = new float*[batch_size];
  float ** Y_pred = new float*[batch_size];
  for (int i = 0; i < batch_size; i++) {
    Y[i] = new float[10];
    Y_pred[i] = new float[10];
  }

  // Size of a training image (in floats), number of training images
  const int image_size = num_channels * dimX * dimY;

  // Seed random generator...
  // srand (time(NULL));



  // Actual training
  for (int batch_index = 0; batch_index < num_batches; batch_index++) {
    // Copy training batch to workspace[0]
    for (int i = 0; i < batch_size; i++) {
      indices[i] = rand() % X_train->dims->num_images;
      memcpy(workspace[0].vals + i * image_size, 
          X_train->vals + indices[i] * image_size, image_size * sizeof(float));
    }

    // Display X (input)
    visualize4(&workspace[0], 0, 0, dimX, dimY);

    // Fprop all layers
    for (int l = 0; l < num_layers; l++) {
      layers[l]->forward_prop(&workspace[l], &workspace[l + 1]);
      visualize4(&workspace[l + 1], 0, 0, workspace[l + 1].dims->dimX, workspace[l + 1].dims->dimY);
    }

    // Fill Y, Y_pred
    for (int i = 0; i < batch_size; i++) {
      memcpy(Y[i], Y_train[indices[i]], 10 * sizeof(float));

      // workspace[-1] is batch_size x 1 x 10 x 1
      memcpy(Y_pred[i], workspace[num_layers].vals + i * 10, 
          10 * sizeof(float));
    }

    // loss calculation
    float tot_loss = loss(Y, Y_pred, batch_size);
    std::cout << "Loss: " << tot_loss << std::endl;



    // do backpropagation shit on batch

    // what's the best way to update weights?




  }



  // Free everything
  for (int i = 0; i < batch_size; i++) {
    delete Y[i];
    delete Y_pred[i];
  }
  delete indices;
  delete Y;
  delete Y_pred;
}
