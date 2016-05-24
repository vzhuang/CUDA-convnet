#include "utils.hpp"


float sigmoid(float x)
{
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}
