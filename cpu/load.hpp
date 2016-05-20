#include <string>


const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int DIM = 28;


float * one_hot(int k);
int un_hot(float * arr);

float *** load_X(std::string filename, int SIZE);

// Loads one-hot representation of y (0~9)
float ** load_Y(std::string filename, int SIZE);

void visualize(float *** X, int index);
void visualize2(float *** X, int index);
