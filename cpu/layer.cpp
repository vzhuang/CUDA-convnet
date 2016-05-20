#include "layer.hpp"

#include <math.h>



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

void ConvLayer::forward_prop(float *** input, Dimensions * input_dimensions,
			float *** output, Dimensions * output_dimensions) 
{
	
}

void ConvLayer::back_prop() {
	
}



/**
 * Stick to sigmoid for now
 */
ActivationLayer::ActivationLayer() {
	
}

void ActivationLayer::forward_prop(float *** input, Dimensions * input_dimensions,
			float *** output, Dimensions * output_dimensions) 
{
	int dimX = input_dimensions->dimX;
	int dimY = input_dimensions->dimY;
	int dimZ = input_dimensions->dimZ;

	for (int i = 0; i < dimX; i++) {
		for (int j = 0; j < dimY; j++) {
			for (int k = 0; k < dimZ; k++) {
				output[i][j][k] = 1.0 / (1.0 + exp(-input[i][j][k]));
			}
		}
	}

	output_dimensions->dimX = dimX;
	output_dimensions->dimY = dimY;
	output_dimensions->dimZ = dimZ;
}

void ActivationLayer::back_prop() {
	
}



/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int size_) {
	int size = size_;
}


void PoolingLayer::forward_prop(float *** input, Dimensions * input_dimensions,
			float *** output, Dimensions * output_dimensions) 
{
	
}

void PoolingLayer::back_prop() {
	
}



