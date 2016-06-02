#include "layer.h"

#include <algorithm>



/**
 * Max pooling of size by size region
 */
PoolingLayer::PoolingLayer(int pool_size_, int stride_) {
  pool_size = pool_size_;
  stride = stride_;
}

void PoolingLayer::fprop(Tensor * input_, Tensor ** output_) {
  for (int n = 0; n < output->dims.num_images; n++)
    for (int c = 0; c < output->dims.num_channels; c++)
      for (int i = 0; i < output->dims.rows; i++) {
        int min_i = i * stride;
        int max_i = std::min(min_i + pool_size, input_->dims.rows);
        for (int j = 0; j < output->dims.cols; j++) {
          int min_j = j * stride;
          int max_j = std::min(min_j + pool_size, input_->dims.cols);

          // Find max value over the pooling area
          float max_value = -FLT_MAX;
          int max_X = -1;
          int max_Y = -1;
          for (int i2 = min_i; i2 < max_i; i2++)
            for (int j2 = min_j; j2 < max_j; j2++)
              if (input_->get(n, c, i2, j2) > max_value) {
                max_value = input_->get(n, c, i2, j2);
                max_X = i2;
                max_Y = j2;
              }
          output->set(n, c, i, j, max_value);
          switches_X->set(n, c, i, j, max_X);
          switches_Y->set(n, c, i, j, max_Y);
        }
      }

  *output_ = output;
}

/**
* Propagates errors through max pooling layer (i.e. to max points in prev layer)
 */
void PoolingLayer::bprop(Tensor ** input_grad, Tensor * output_grad, float eta) {
  // int num_images = output_grad->dims->num_images;
  // int num_channels = output_grad->dims->num_channels;
  // int rows = output_grad->dims->rows;
  // int cols = output_grad->dims->cols;

  // // Bprop based on switches
  // for (int n = 0; n < num_images; n++)
  //   for (int c = 0; c < num_channels; c++)
  //     for (int i = 0; i < rows; i++)
  //       for (int j = 0; j < cols; j++) {
  //         int max_X = switches_X.get(n, c, i, j);
  //         int max_Y = switches_Y.get(n, c, i, j);
  //         input_grad->set(n, c, max_X, max_Y, output_grad->get(n, c, i, j));
  //       }

  // // Free switches
  // switches_X.free_vals();
  // switches_Y.free_vals();
}

void PoolingLayer::get_output_dims(Dimensions * input_dims, Dimensions * output_dims) {
  output_dims->num_images = input_dims->num_images;
  output_dims->num_channels = input_dims->num_channels;
  output_dims->rows = (input_dims->rows - pool_size) / stride + 1;
  output_dims->cols = (input_dims->cols - pool_size) / stride + 1;
}

void PoolingLayer::init_mem(Dimensions * input_dims) {
  Dimensions d;
  get_output_dims(input_dims, &d);
  output = new Tensor(&d, false);
  switches_X = new Tensor(&d, false);
  switches_Y = new Tensor(&d, false);
}

void PoolingLayer::free_mem() {
  delete output;
  delete switches_X;
  delete switches_Y;
}

