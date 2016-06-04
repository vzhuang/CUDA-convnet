#include "activation.h"

Activation::Activation(int type_)
{
  type = type_;
}

float Activation::activ(float x)
{
  if (type == RELU) { 
    if (x > 0) {
      return x;
    }
    return 0;
  }
  else if (type == SIGMOID) {
    return sigmoid(x);
  }

  return -1;
}
  
// for backprop-ing errors
float Activation::deriv(float x)
{
  if (type == RELU) {
    if (x > 0) {
      return 1;
    }
    return 0;
  }
  else if (type == SIGMOID) { 
    return sigmoid_prime(x);
  }
  
  return -1;
}
