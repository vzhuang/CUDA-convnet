#include <string>

#include "tensor.hpp"


const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int DIM = 28;


Tensor * load_X(std::string filename, int SIZE);

// Loads one-hot representation of y (0~9)
float ** load_Y(std::string filename, int SIZE);
