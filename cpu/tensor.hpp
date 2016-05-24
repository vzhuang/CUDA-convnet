#ifndef TENSOR_H
#define TENSOR_H

struct Dimensions {
  int num_images, num_channels, dimX, dimY;
};

/**
 * Implements a four dimensional tensor
 * Dimemsions as given in struct
 */
class Tensor {
public:
  float **** vals;
  Dimensions * dims;

  // initializes to zero
  Tensor();
  void init_vals(Dimensions * dims);
  void free_vals();    
};

#endif
