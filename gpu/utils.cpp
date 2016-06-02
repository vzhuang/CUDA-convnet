#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>


void one_hot(int k, float * arr, int size) {
  assert(k < size);
  memset(arr, 0, sizeof(float) * size);
  arr[k] = 1;
}
int un_hot(float * arr, int size) {
  for (int i = 0; i < size; i++)
    if (arr[i] == 1)
      return i;
  return -1;
}



void print(Tensor * t, int n) {
  int num_images = t->dims.num_images;
  int num_channels = t->dims.num_channels;
  int rows = t->dims.rows;
  int cols = t->dims.cols;

  printf("Size: %d x %d x %d x %d\n", num_images, num_channels, rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%3.f ", t->data[n * num_channels * rows * cols + i * rows + j]);
    }
    printf("\n");
  }
}
