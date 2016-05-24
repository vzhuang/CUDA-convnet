#include "load.hpp"
#include "utils.hpp"

#include <fstream>


Tensor * load_X(std::string filename, int SIZE) {
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );
  
  unsigned char * buffer = new unsigned char[SIZE * DIM * DIM];

  Tensor * X = new Tensor();
  Dimensions * dims = new Dimensions();
  dims->num_images = SIZE;
  dims->num_channels = 1;
  dims->dimX = DIM;
  dims->dimY = DIM;
  X->init_vals(dims);

  // Read in training set
  inFile.read((char *) buffer, SIZE * DIM * DIM + 16);
  for (int n = 0; n < SIZE; n++) 
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        X->set(n, 0, i, j, (float) (buffer[n * DIM * DIM + i * DIM + j + 16]));

  delete buffer; 
  return X;
}

// Loads one-hot representation of y (0~9)
float ** load_Y(std::string filename, int SIZE) {
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );

  unsigned char * buffer = new unsigned char[SIZE + 8];
  
  float ** Y = new float*[SIZE];

  // Read in training set
  inFile.read((char *) buffer, SIZE + 8);
  for (int n = 0; n < SIZE; n++)
    Y[n] = one_hot((int) buffer[n + 8]);

  delete buffer;  
  return Y;
}
