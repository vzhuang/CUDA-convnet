#include "layer.hpp"


class ConvNet {
  int num_layers;
  Layer ** layers;
  Tensor * X_train;
  float ** Y_train;

  Tensor * fprop_space;    // Memory for doing fprop
  Tensor * bprop_space;    // Memory for doing bprop

  void make_workspaces(int num_images);
  
public:
  ConvNet(Layer ** layers_, int num_layers_, 
      Tensor * X_train_, float ** Y_train_);

  void train(float eta, int num_batches, int batch_size);

/**
 * Returns one-hot predictions for given dataset X
 */
  // double eval(float *** X, float ** Y, int n);

/**
 * Returns loss on dataset X, Y
 */
  // double * predict(float *** X, float ** Y, int n);
};
