#include "layer.hpp"


class NeuralNetwork {
  int num_layers;
  Layer ** layers;
  float **** X;
  float ** Y;

  Tensor * workspace;    // Memory for doing fprop

  void make_workspace();
  
public:
  NeuralNetwork(Layer ** layers_, int num_layers_, 
      float **** X_, float ** Y_, Dimensions * X_dim_);
  void step();
};
