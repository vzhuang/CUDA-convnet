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

void ConvNet::make_workspaces(int batch_size) {
  // Get dimensions of X
  const int num_channels = X_train->dims->num_channels;
  const int dimX = X_train->dims->dimX;
  const int dimY = X_train->dims->dimY;

  // Number of layers + 1 (for input)
  fprop_space = new Tensor[num_layers + 1];
  bprop_space = new Tensor[num_layers + 1];

  // Allocate memory for initial layer (input)
  Dimensions * dims = new Dimensions{batch_size, num_channels, dimX, dimY};
  fprop_space[0].init_vals(dims);
  bprop_space[0].init_vals(dims);

  // Propagate forward to get new dimensions and init_vals()
  for (int l = 0; l < num_layers; l++) {
    Dimensions * new_dims = new Dimensions();
    layers[l]->output_dim(fprop_space[l].dims, new_dims);

    fprop_space[l + 1].init_vals(new_dims);
    bprop_space[l + 1].init_vals(new_dims);
  }

  // Print out layer sizes
  for (int l = 0; l < num_layers + 1; l++) {
    int num_images = fprop_space[l].dims->num_images;
    int num_channels = fprop_space[l].dims->num_channels;
    int dimX = fprop_space[l].dims->dimX;
    int dimY = fprop_space[l].dims->dimY;

    std::cout << "Layer " << l << ": " << num_images << " x "<< num_channels << " x "<< dimX << " x "<< dimY << std::endl;
  } 
}

/**
 * Implements minibatch SGD for backpropagation
 *
 * Params:
 * eta: learning rate
 * num_epochs: number of times to pass through training set
 * num_batches: number of batches to run
 * batch_size: number of data_points in each batch
 */
void ConvNet::train(float eta, int num_epochs, int num_batches, int batch_size)
{
  // Allocate memory for both fprop and bprop
  make_workspaces(batch_size);

  // Memory to store training indices, Y, Y_pred
  int * indices = new int[batch_size];
  float ** Y = new float*[batch_size];
  float ** Y_pred = new float*[batch_size]; 
  for (int i = 0; i < batch_size; i++) {
    Y[i] = new float[10];
    Y_pred[i] = new float[10];
  }

  // Size of a training image (in floats), number of training images
  const int image_size = X_train->dims->num_channels * X_train->dims->dimX * X_train->dims->dimY;

  // Seed random generator...
  // srand (time(NULL));
  
  // Actual training
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch_index = 0; batch_index < num_batches; batch_index++) {
      // Zero out the workspaces before beginning
      for (int w = 0; w < num_layers + 1; w++) {
	fprop_space[w].zero_out();
	bprop_space[w].zero_out();
      }

      // Copy training batch to fprop_space[0]
      for (int i = 0; i < batch_size; i++) {
	indices[i] = rand() % X_train->dims->num_images;
	memcpy(fprop_space[0].vals + i * image_size, 
	    X_train->vals + indices[i] * image_size, image_size * sizeof(float));
      }

      // Display X (input)
      visualize4(&fprop_space[0], 0, 0, fprop_space[0].dims->dimX, fprop_space[0].dims->dimY);

      // Fprop all layers
      for (int l = 0; l < num_layers; l++) {
	layers[l]->forward_prop(&fprop_space[l], &fprop_space[l + 1]);
	visualize4(&fprop_space[l + 1], 0, 0, fprop_space[l + 1].dims->dimX, fprop_space[l + 1].dims->dimY);
      }

      // Fill Y, Y_pred
      for (int i = 0; i < batch_size; i++) {
	memcpy(Y[i], Y_train[indices[i]], 10 * sizeof(float));

	// fprop_space[-1] is batch_size x 1 x 10 x 1
	memcpy(Y_pred[i], fprop_space[num_layers].vals + i * 10, 10 * sizeof(float));
      }

      // loss calculation
      float tot_loss = loss(Y, Y_pred, batch_size);
      std::cout << "Loss: " << tot_loss << std::endl;


      // Display last gradients
      visualize4(&bprop_space[num_layers], 0, 0, bprop_space[num_layers].dims->dimX, bprop_space[num_layers].dims->dimY);

      // do backpropagation shit on batch
      for (int l = num_layers - 1; l >= 0; l--) {
	layers[l]->back_prop(&bprop_space[l], &bprop_space[l + 1], eta);
	visualize4(&bprop_space[l], 0, 0, bprop_space[l].dims->dimX, bprop_space[l].dims->dimY);
      }

    }    
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
