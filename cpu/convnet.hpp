#include "layer.hpp"

class ConvNet {

public:
	int num_layers;
	Layer * layers;
	
public:
	
	void init();
	void free();
	void train(float *** X, float ** Y, int n, int image_size,
			   float eta, int num_epochs, int batch_size);
	double eval(float *** X, float ** Y, int n);
	double * predict(float *** X, float ** Y, int n);
	
}
