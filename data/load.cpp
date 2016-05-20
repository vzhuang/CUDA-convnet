#include <iostream>
#include <fstream>
#include <string>
using namespace std;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int DIM = 28;


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


float *** load_X(string filename, int SIZE) {
  ifstream inFile;
  inFile.open( filename, ios::in|ios::binary );
  
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
float ** load_Y(string filename, int SIZE) {
  ifstream inFile;
  inFile.open( filename, ios::in|ios::binary );

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


int main() {
  float *** X_train = load_X("train-images.idx3-ubyte", TRAIN_SIZE);
  float **  Y_train = load_Y("train-labels.idx1-ubyte", TEST_SIZE);
  float *** X_test  = load_X("t10k-images.idx3-ubyte", TRAIN_SIZE);
  float **  Y_test  = load_Y("t10k-labels.idx1-ubyte", TEST_SIZE);

  int k = 420;
  visualize(X_train, k);
  cout << un_hot(Y_train[k]) << endl;
  visualize(X_test, k);
  cout << un_hot(Y_test[k]) << endl;
}
