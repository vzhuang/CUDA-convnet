#include "layer.h"
#include "load.h"
#include "utils.h"


int main() {
  Tensor * X_train = load_X("../data/train-images.idx3-ubyte", TRAIN_SIZE);
  Tensor * Y_train = load_Y("../data/train-labels.idx1-ubyte", TRAIN_SIZE);

  print(X_train, 0);

  X_train->dims.num_images=1;
  Tensor * X_out;
  PoolingLayer l1 = PoolingLayer(2, 2);
  l1.init_mem(&X_train->dims);
  l1.fprop(X_train, &X_out);

  print(X_out, 0);
}
