#include "layer.hpp"

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

ConvLayer::forward_prop() {
	
}

ConvLayer::back_prop() {
	
}

/**
 * Stick to sigmoid for now
 */
ActivationLayer::ActivationLayer() {
	
}

ActivationLayer::forward_prop() {
	
}

ActivationLayer::back_prop() {
	
}

/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int size_) {
	size = size_;
}


PoolingLayer::forward_prop() {
	
}

PoolingLayer::back_prop() {
	
}

