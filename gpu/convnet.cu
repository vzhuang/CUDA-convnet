#include "convnet.h"
#include "utils.h"

#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>


// get errors
__global__
void cudaGetErrorKernel(
           float * dev_output_data,
           float * dev_error_data,
           int * indices,
           float * dev_Y_data,
           int * dev_loss) 
{
  extern __shared__ int loss[];

  int ind = threadIdx.x;
  int y_ind = indices[ind];

  float max_val = -FLT_MAX;
  int max_ind = -1;
  for (int i = 0; i < 10; i++) {
    float val = dev_output_data[ind * 10 + i];
    if (val > max_val) {
      max_val = val;
      max_ind = i;
    }

    dev_error_data[ind * 10 + i] = val - dev_Y_data[y_ind * 10 + i];
  }

  if (dev_Y_data[y_ind * 10 + max_ind] != 1)
    loss[ind] = 1;
  else
    loss[ind] = 0;

  __syncthreads();

  int total_loss = 0;
  if (ind == 0) {
    for (int i = 0; i < blockDim.x; i++)
      total_loss += loss[i];
    *dev_loss = total_loss; 
  }
}




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

  assert(dims->num_images == batch_size);
  assert(dims->num_channels == 1);
  assert(dims->rows == 10);
  assert(dims->cols == 1);

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
  
  // Memory to store training indices and loss
  int * indices = new int[batch_size];
  Tensor ** input = new Tensor*[num_layers + 1];
  Tensor ** errors = new Tensor*[num_layers + 1];
  errors[num_layers] = new Tensor(batch_size, 1, 10, 1, true);
  int * dev_indices;
  int * dev_loss;
  cudaMalloc((void **)&dev_indices, sizeof(int) * batch_size);
  cudaMalloc((void **)&dev_loss, sizeof(int));

  // Size of a training image (in floats), number of training images
  const int image_size = dev_X_train->dims.num_channels * dev_X_train->dims.rows * dev_X_train->dims.cols;

  // Actual training
  int trainingindex = 0;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_loss = 0.0;

    for (int batch_index = 0; batch_index < num_batches; batch_index++) {
      // Copy training batch to dev_X_in
      for (int i = 0; i < batch_size; i++) {
        indices[i] = trainingindex;
        cudaMemcpy(dev_X_in->data + i * image_size, 
                   dev_X_train->data + indices[i] * image_size, 
                   image_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        trainingindex++;
        trainingindex %= dev_X_train->dims.num_images;
      }

      // Fprop all layers
      input[0] = dev_X_in;
      for (int l = 0; l < num_layers; l++)
        layers[l]->fprop(input[l], &input[l+1]);

      // CALCULATE ERRORS (populate errors[num_layers])
      cudaMemcpy(dev_indices, indices, sizeof(int) * batch_size,
		 cudaMemcpyHostToDevice);
      cudaGetErrorKernel<<<1, batch_size, batch_size * sizeof(int)>>>(
          input[num_layers]->data,
          errors[num_layers]->data,
          dev_indices, 
          dev_Y_train->data, 
          dev_loss);      
      int loss = 0;
      cudaMemcpy(&loss, dev_loss, sizeof(int), cudaMemcpyDeviceToHost);
      epoch_loss += (float) loss / train_size;

      // Bprop all layers
      for (int l = num_layers - 1; l >= 0; l--)
        layers[l]->bprop(&errors[l], errors[l+1], eta);
    } 

    // Display loss
    std::cout << "Epoch loss: " << epoch_loss << std::endl; 

    // Display sample FOR DEBUG
    int k = 0;
    assert(k < batch_size);
    Tensor * temp;

    // printf("\nINPUT\n\n");
    // temp = toCPU(dev_X_in);
    // print(temp, k, 0);

    // printf("\nFPROP\n\n");
    // for (int l = 0; l < num_layers; l++) {
    //   temp = toCPU(input[l+1]);
    //   print(temp, k, 0);
    // }  

    // printf("\nERRORS\n\n");
    // temp = toCPU(errors[num_layers]);
    // print(temp, k, 0);

    // printf("\nBPROP\n\n");
    // for (int l = num_layers - 1; l >= 0; l--) {
    //   temp = toCPU(errors[l]);
    //   print(temp, k, 0);
    // }

    // printf("\nWEIGHTS\n\n");
    // temp = toCPU(((FullyConnectedLayer *)layers[1])->dev_weights);
    // print(temp, 0, 0);
    // print(temp, 0, 1);
    // temp = toCPU(((FullyConnectedLayer *)layers[0])->dev_weights);
    // print(temp, 0, 0);
    // print(temp, 0, 1);
  }


  // Free everything
  delete indices;
  delete input;
  delete errors[num_layers];
  delete errors;
  cudaFree(dev_indices);
  cudaFree(dev_loss);
  free_mem();
}
