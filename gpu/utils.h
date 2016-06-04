#ifndef UTILS_H
#define UTILS_H


#include "tensor.h"



// One-hot an integer k, storing in array arr
void one_hot(int k, float * arr, int size);

// Un-hot the array and return the value
int un_hot(float * arr, int size);

float sigmoid(float x);
float sigmoid_prime(float x);


void print(Tensor * t, int n, int c);


Tensor * toGPU(Tensor * t);
Tensor * toCPU(Tensor * t);



#endif
