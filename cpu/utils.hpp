#ifndef UTILS_H
#define UTILS_H

#include "tensor.hpp"

float * one_hot(int k);
int un_hot(float * arr);

void visualize4(Tensor * data, int ind1, int ind2, int dimX, int dimY);


float sigmoid(float x);
float sigmoid_prime(float x);

float loss(float ** Y, float ** Y_pred, int num_Y, int train_size);

#endif
