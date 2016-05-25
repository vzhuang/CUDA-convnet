#ifndef TENSOR_H
#define TENSOR_H

#include <string.h>

struct Dimensions {
  int num_images, num_channels, dimX, dimY;
};

/**
 * Implements a four dimensional tensor
 * Dimemsions as given in struct
 */
class Tensor {
public:
  float * vals;
  Dimensions * dims;

  // initializes to zero
  Tensor();
  void init_vals(Dimensions * dims);
  void free_vals();
  float get(int a, int b, int c, int d);              // Retrieve vals[a][b][c][d]
  void set(int a, int b, int c, int d, float val);    // Set vals[a][b][c][d] = val
  void zero_out();
};

#endif
