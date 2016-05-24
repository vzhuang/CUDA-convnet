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


float **** load_X(std::string filename, int SIZE) {
  std::ifstream inFile;
  inFile.open( filename, std::ios::in|std::ios::binary );
  
  unsigned char * buffer = new unsigned char[SIZE * DIM * DIM];

  float **** X = new float ***[SIZE];
  for (int n = 0; n < SIZE; n++) {
    X[n] = new float **[1];
    X[n][0] = new float *[DIM];
    for (int i = 0; i < DIM; i++)
      X[n][0][i] = new float[DIM];
  }

  // Read in training set
  inFile.read((char *) buffer, SIZE * DIM * DIM + 16);
  for (int n = 0; n < SIZE; n++) 
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        X[n][0][i][j] = (float) (buffer[n * DIM * DIM + i * DIM + j + 16]);

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

void free_X(float **** X, int SIZE) {
  for (int n = 0; n < SIZE; n++) {
    for (int i = 0; i < DIM; i++)
      delete X[n][0][i];
    delete X[n][0];
    delete X[n];
  }
  delete X;
}
void free_Y(float ** Y, int SIZE) {
  for (int n = 0; n < SIZE; n++)
    delete Y[n];
  delete Y;
}


void visualize(float **** X, int index) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++)
      printf("%3.f ", X[index][0][i][j]);
    printf("\n");
  }
}

void visualize2(float **** X, int index) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++)
      printf("%0.3f ", X[index][0][i][j]);
    printf("\n");
  }
}

void visualize3(float **** data, int ind1, int ind2, int dimX, int dimY) {
  for (int i = 0; i < dimX; i++) {
    for (int j = 0; j < dimY; j++)
      printf("%3.f ", data[ind1][ind2][i][j]);
    printf("\n");
  } 
}
