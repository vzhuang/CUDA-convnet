#include "convnet.h"
#include "utils.h"

#include <iostream>
#include <time.h>
#include <cuda_runtime.h>



ConvNet::ConvNet(Layer ** layers_, int num_layers_, 
      Tensor * dev_X_train_, Tensor * dev_Y_train_)
{
  layers = layers_;
  num_layers = num_layers_;
  dev_X_train = dev_X_train_;
  dev_Y_train = dev_Y_train_;
}

void ConvNet::init_mem(int batch_size) {
  Dimensions * dims = new Dimensions(batch_size, dev_X_train->dims.num_channels, 
      dev_X_train->dims.rows, dev_X_train->dims.cols);
  Dimensions * next_dims = new Dimensions();
  Dimensions * temp;

  // Allocate memory for minibatch input
  dev_X_in = new Tensor(dims, true);

  // Propagate forward to get new dimensions and init_mem()
  for (int l = 0; l < num_layers; l++) {
    layers[l]->init_mem(dims);
    layers[l]->get_output_dims(dims, next_dims);

    // Print out layer input/output
    printf("Layer %d: %d x %d x %d x %d --> %d x %d x %d x %d\n", l,
        dims->num_images, dims->num_channels, dims->rows, dims->cols,
        next_dims->num_images, next_dims->num_channels, next_dims->rows, next_dims->cols);

    temp = dims;
    dims = next_dims;
    next_dims = temp;
  }

  delete dims;
  delete next_dims;
}
void ConvNet::free_mem() { 
  delete dev_X_in;

  for (int l = 0; l < num_layers; l++)
    layers[l]->free_mem();
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
void ConvNet::train(float eta, int num_epochs, int num_batches, int batch_size,
		    int train_size)  
{
  // Allocate memory for both fprop and bprop
  init_mem(batch_size);

  // Memory to store training indices
  int * indices = new int[batch_size];
  float ** Y = new float*[batch_size];
  float ** Y_pred = new float*[batch_size]; 
  for (int i = 0; i < batch_size; i++) {
    Y[i] = new float[10];
    Y_pred[i] = new float[10];
  }

  // Size of a training image (in floats), number of training images
  const int image_size = dev_X_train->dims.num_channels * dev_X_train->dims.rows * dev_X_train->dims.cols;

  // Seed random generator...
  srand (time(NULL));
  
  // Actual training
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_loss = 0.0;

    for (int batch_index = 0; batch_index < num_batches; batch_index++) {
      // Copy training batch to dev_X_in
      for (int i = 0; i < batch_size; i++) {
        indices[i] = i; // NOT RANDOM FOR DEBUG rand() % dev_X_train->dims.num_images;
        cudaMemcpy(dev_X_in->data + i * image_size, 
                   dev_X_train->data + indices[i] * image_size, 
                   image_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
      }

      // Variables to store all the pointers
      Tensor ** input = new Tensor*[num_layers + 1];
      Tensor ** errors = new Tensor*[num_layers + 1];
      Tensor * temp;

      // Display X (input)
      temp = toCPU(dev_X_in);
      print(temp, 0, 0);

      // Fprop all layers
      input[0] = dev_X_in;
      printf("\nFPROP\n\n");
      for (int l = 0; l < num_layers; l++) {
        layers[l]->fprop(input[l], &input[l+1]);

        // Visualize
        temp = toCPU(input[l+1]);
        print(temp, 0, 0);
      }

      // calculate error at final layer
      printf("calc error\n");
      for (int i = 0; i < batch_size; i++) {
	cudaMemcpy(Y[i], dev_Y_train->data + indices[i], 10 * sizeof(float),
		   cudaMemcpyDeviceToHost);
	cudaMemcpy(Y_pred[i], input[num_layers]->data, 10 * sizeof(float),
		   cudaMemcpyDeviceToHost);
	printf("blah\n");
	for (int j = 0; j < 10; j++) {
	  cudaMemset(errors[num_layers]->data + i * 10 + j,
		     2 * (Y_pred[i][j] - Y[i][j]),
		     sizeof(float));
	}
      }
      printf("calculated error\n");
      epoch_loss += loss(Y, Y_pred, batch_size, train_size);

      // Bprop all layers
      errors[num_layers] = input[num_layers];
      printf("\nBPROP\n\n");
      for (int l = num_layers - 1; l >= 0; l--) {
        layers[l]->bprop(&errors[l], errors[l+1], eta);

        // Visualize
        temp = toCPU(errors[l]);
        print(temp, 0, 0);
      }
    } 

    std::cout << "Epoch loss: " << epoch_loss << std::endl;   
  }


  // Free everything
  delete indices;
  free_mem();
}
