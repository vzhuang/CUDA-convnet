#include <iostream>
#include <fstream>
using namespace std;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int DIM = 28;

unsigned int X_train[TRAIN_SIZE][DIM][DIM]; 
unsigned int Y_train[TRAIN_SIZE];
unsigned int X_test[TEST_SIZE][DIM][DIM]; 
unsigned int Y_test[TEST_SIZE];


void load_X_train() {
  ifstream inFile;
  inFile.open( "train-images.idx3-ubyte", ios::in|ios::binary );
  
  unsigned char buffer[DIM * DIM];

  // Read past header
  for (int i = 0; i < 4; i++)
    inFile.read((char *) buffer, 4);

  // Read in training set
  for (int n = 0; n < TRAIN_SIZE; n++) {
    inFile.read((char *) buffer, DIM * DIM);
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        X_train[n][i][j] = (unsigned int) (buffer[i * DIM + j]);
  }
}

void load_Y_train() {
  ifstream inFile;
  inFile.open( "train-labels.idx1-ubyte", ios::in|ios::binary );

  unsigned char buffer[TRAIN_SIZE + 8];

  // Read in training set
  inFile.read((char *) buffer, TRAIN_SIZE + 8);
  for (int n = 0; n < TRAIN_SIZE; n++)
    Y_train[n] = (unsigned int) buffer[n + 8];
}

void load_X_test() {
  ifstream inFile;
  inFile.open( "t10k-images.idx3-ubyte", ios::in|ios::binary );

  unsigned char buffer[DIM * DIM];

  // Read past header
  for (int i = 0; i < 4; i++)
    inFile.read((char *) buffer, 4);

  // Read in test set
  for (int n = 0; n < TEST_SIZE; n++) {
    inFile.read((char *) buffer, DIM * DIM);
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        X_test[n][i][j] = (unsigned int) (buffer[i * DIM + j]);
  }
}

void load_Y_test() {
  ifstream inFile;
  inFile.open( "t10k-labels.idx1-ubyte", ios::in|ios::binary );
  
  unsigned char buffer[TEST_SIZE + 8];

  // Read in test set
  inFile.read((char *) buffer, TRAIN_SIZE + 8);
  for (int n = 0; n < TEST_SIZE; n++)
    Y_test[n] = (unsigned int) buffer[n + 8];
}


// void visualize(unsigned int X[][DIM][DIM], int index) {
//   for (int i = 0; i < DIM; i++) {
//     for (int j = 0; j < DIM; j++)
//       printf("%3d ", X[index][i][j]);
//     printf("\n");
//   }
// }

int main() {
  load_X_train();
  load_Y_train();
  load_X_test();
  load_Y_test();

  // int k = 420;
  // visualize(X_train, k);
  // cout << Y_train[k] << endl;
  // visualize(X_test, k);
  // cout << Y_test[k] << endl;
}
