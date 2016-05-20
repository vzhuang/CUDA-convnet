#include "load.hpp"

#include <iostream>
#include <fstream>


float * one_hot(int k) {
  float * arr = new float[10];
  memset(arr, 0, sizeof(float) * 10);
  arr[k] = 1;
  return arr;
}
int un_hot(float * arr) {
  for (int i = 0; i < 10; i++)
    if (arr[i] == 1)
      return i;
  return -1;
}


float *** load_X(std::string filename, int SIZE) {
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );
  
  unsigned char buffer[DIM * DIM];

  float *** X = new float **[SIZE];
  for (int n = 0; n < SIZE; n++) {
    X[n] = new float *[DIM];
    for (int i = 0; i < DIM; i++)
      X[n][i] = new float[DIM];
  }

  // Read past header
  for (int i = 0; i < 4; i++)
    inFile.read((char *) buffer, 4);

  // Read in training set
  for (int n = 0; n < SIZE; n++) {
    inFile.read((char *) buffer, DIM * DIM);
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        X[n][i][j] = (float) (buffer[i * DIM + j]);
  }

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
  
  return Y;
}


void visualize(float *** X, int index) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++)
      printf("%3.f ", X[index][i][j]);
    printf("\n");
  }
}

void visualize2(float *** X, int index) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++)
      printf("%0.3f ", X[index][i][j]);
    printf("\n");
  }
}