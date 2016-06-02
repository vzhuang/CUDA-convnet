#include "load.h"
#include "utils.h"

#include <fstream>


Tensor * load_X(std::string filename, int SIZE) {
  const int buffer_size = SIZE * DIM * DIM + 16;
  unsigned char * buffer = new unsigned char[buffer_size];

  // Read in data
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );
  inFile.read((char *) buffer, buffer_size);
  inFile.close();
  
  // Copy to X
  Tensor * X = new Tensor(SIZE, 1, DIM, DIM, false);
  for (int n = 0; n < SIZE * DIM * DIM; n++) 
    X->data[n] = (float) buffer[n + 16];

  delete buffer;
  return X;
}


Tensor * load_Y(std::string filename, int SIZE) {
  const int buffer_size = SIZE + 8;
  unsigned char * buffer = new unsigned char[buffer_size];

  // Read in data
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );
  inFile.read((char *) buffer, buffer_size);
  inFile.close();

  // Copy to Y
  Tensor * Y = new Tensor(SIZE, 1, 10, 1, false);
  for (int n = 0; n < SIZE; n++)
    one_hot((int) buffer[n + 8], Y->data + n * 10, 10);

  delete buffer;  
  return Y;
}
