class Layer {
	virtual void forward_prop();
	virtual void back_prop();
};

/**
 * Implement zero-padded convolutions!
 */ 
class ConvLayer : public Layer {
	int stride;
	int size; // conv filter size
	int num_filters; // number of filters

public:	
	float *** weights;
	float * biases;
	
	void ConvLayer(int num_filters_, int size_, int stride_);
	void forward_prop();
	void back_prop();
};

class ActivationLayer : public Layer {

public:	
	// activation types - ReLU, tanh, sigmoid?
	void ActivationLayer();
	void forward_prop();
	void back_prop();
};

class PoolingLayer : public Layer {
	int pool_size;	

public:	

	void PoolingLayer(int size_);
	void forward_prop();
	void back_prop();
};

class FullyConnectedLayer : public Layer {
	int num_neurons;
	
public:
	float ** weights;

	void FullyConnectedLayer();
	void forward_prop();
	void back_prop();
	// flatten inputs
	float * flatten(); 
};
