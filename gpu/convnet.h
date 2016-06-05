#include "layer.h"


class ConvNet {
  int num_layers;
  Layer ** layers;
  Tensor * dev_X_train;
  Tensor * dev_Y_train;


  // Memory to store minibatch
  Tensor * dev_X_in;

  void init_mem(int batch_size);
  void free_mem();
  

public:
  ConvNet(Layer ** layers_, int num_layers_, 
      Tensor * dev_X_train_, Tensor * dev_Y_train_);

  void train(float eta, int num_epochs, int num_batches, int batch_size);

/**
 * Returns one-hot predictions for given dataset X
 */
  // double eval(float *** X, float ** Y, int n);

/**
 * Returns loss on dataset X, Y
 */
  // double * predict(float *** X, float ** Y, int n);
};
