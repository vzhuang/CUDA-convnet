struct Dimensions {
  int num_images, num_channels, dimX, dimY;
};



class Layer {
public:
  virtual void forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions) = 0;
  virtual void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions) = 0;
  virtual void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions) = 0;
};



/**
 * Implement zero-padded convolutions!
 */ 
class ConvLayer : public Layer {
public: 
  int stride;
  int size; // conv filter size
  int num_filters; // number of filters
  float *** weights;
  float * biases;
  
  ConvLayer(int num_filters_, int size_, int stride_);
  void forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions);
  void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions);
  void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions);
};



class ActivationLayer : public Layer {

  // Use for backprop
  float **** last_input;

public: 
  // activation types - ReLU, tanh, sigmoid?
  ActivationLayer();
  void forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions);
  void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions);
  void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions);
};



class PoolingLayer : public Layer {

  // Use for backprop
  Dimensions * last_input_dimensions;
  int ***** switches;

public: 
  int pool_size;
  int stride;

  PoolingLayer(int pool_size_, int stride_);
  void forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions);
  void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions);
  void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions);
};



class FullyConnectedLayer : public Layer {
  int num_neurons;
  
public:
  float ** weights;

  FullyConnectedLayer();
  void forward_prop(float **** input, Dimensions * input_dimensions,
      float **** output, Dimensions * output_dimensions);
  void back_prop(float **** input_grad, Dimensions * input_dimensions,
      float **** output_grad, Dimensions * output_dimensions);
  void output_dim(Dimensions * input_dimensions, 
      Dimensions * output_dimensions);

  // flatten inputs
  float * flatten(); 
};
