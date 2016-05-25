#include "utils.hpp"

#include <math.h>
#include <iostream>


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

void visualize4(Tensor * data, int ind1, int ind2, int dimX, int dimY) {
  for (int i = 0; i < dimX; i++) {
    for (int j = 0; j < dimY; j++)
      printf("%0.3f ", data->get(ind1, ind2, i, j));
    printf("\n");
  } 
}



float sigmoid(float x)
{
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

// float loss(float ** Y, float ** Y_pred, int num_Y) {
//   float loss = 0.0;

//   const float eps = 1e-15;
//   for (int i = 0; i < num_Y; i++) {
//     float sum = 0.0;
//     for (int j = 0; j < 10; j++) {
//       float val = Y_pred[i][j];

//       // Clip to [eps, 1 - eps]
//       if (val < eps)
//         val = eps;
//       else if (val > 1 - eps)
//         val = 1-eps;

//       sum += val;
//     }

//     for (int j = 0; j < 10; j++) {
//       float val = Y_pred[i][j];

//       // Clip to [eps, 1 - eps]
//       if (val < eps)
//         val = eps;
//       else if (val > 1 - eps)
//         val = 1-eps;

//       val /= sum;

//       loss += -Y[i][j] * log(val);
//     }
//   }

//   return loss / num_Y;
// }

float loss(float ** Y, float ** Y_pred, int num_Y) {
  float loss = 0.0;

  for (int i = 0; i < num_Y; i++) {
    float max_val = Y_pred[i][0];
    int max_ind = 0;
    for (int j = 1; j < 10; j++) {
      float val = Y_pred[i][j];

      if (val > max_val) {
        max_val = val;
        max_ind = j;
      }
    }

    if (Y[i][max_ind] != 1)
      loss++;
  }

  return loss;
}
