#ifndef LOAD_H
#define LOAD_H


#include "tensor.h"

#include <string>



const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int DIM = 28;


// Load X, dimensions = SIZE * 1 * DIM * DIM
Tensor * load_X(std::string filename, int SIZE);

// Loads one-hot representation of Y (0~9), dimensions = SIZE * 1 * 10 * 1
Tensor * load_Y(std::string filename, int SIZE);


#endif
