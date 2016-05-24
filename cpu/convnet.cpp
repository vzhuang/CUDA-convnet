#include "convnet.hpp"

void ConvNet::ConvNet(Layer * layers, int num_layers) {
	// assume layers already initialized?
	
}

/**
 * Deallocate memory
 */
void ConvNet::free() {
  
	
}


/**
 * Implements minibatch SGD for backpropagation
 *
 * Params:
 * X: input w/ dimensions n, image size, image size
 * Y: one-hot output, i.e. dimensions, n, onehot vector size
 * eta: learning rate
 * num_epochs: number of epochs
 * batch_size: number of data_points in each batch
 */
void ConvNet::train(float *** X, float ** Y, int n, int image_size,
		    float eta, int num_epochs, int batch_size) {
  int num_batches = n / batch_size;
	for (int i = 0; i < num_epochs; i++) {
		for (int j = 0; j < num_batches; j++) {
			// forward prop batch
			
			// do backpropagation shit on batch

			// what's the best way to update weights?
		}
	}
	
}

/**
 * Returns one-hot predictions for given dataset X
 */
float ** ConvNet::predict(float *** X, int n) {
	
}

/**
 * Returns loss on dataset X, Y
 */
float ConvNet::eval(float *** X, float ** Y, int n) {
	
}
