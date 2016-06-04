#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "utils.h"

class Activation {
public:
  int type;
  Activation(int type_);
  float activ(float x);
  float deriv(float x);
};


#endif
