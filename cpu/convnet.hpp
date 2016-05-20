#include "layer.hpp"

class ConvNet {

public:
	int num_layers;
	Layer * layers;
	

	void init();
	void train();
	double eval();
	double * predict();
	
}
