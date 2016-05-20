class Layer {
	virtual void forward_prop();
	virtual void back_prop();
};

class ConvLayer : public Layer{
public:
	int stride;
	int size; // conv filter size
	int num_filters; // number of filters

	void forward_prop();
	void back_prop();
};

class Pooling : public Layer{

	// implement max pooling?
public:	
	int pool_size
	void forward_prop();
	void back_prop();
};
