#ifndef TENSOR_H
#define TENSOR_H



/**
 * Convenience struct for passing around dimensions
 */
struct Dimensions {
  int num_images;
  int num_channels;
  int rows;
  int cols;

  Dimensions() {};
  Dimensions(int num_images_, int num_channels_, int rows_, int cols_) {
    num_images = num_images_;
    num_channels = num_channels_;
    rows = rows_;
    cols = cols_;
  }
};


/**
 * Convenience struct for passing around data + dimensions
 */
struct Tensor {
  float * data;
  Dimensions dims;
  bool gpu_memory;

  // Constructors that set dimensions and allocates memory for data array
  Tensor(int num_images_, int num_channels_, int rows_, int cols_, bool gpu_memory_);
  Tensor(Dimensions * dims_, bool gpu_memory_);

  // Destructor frees data array
  ~Tensor();

  // Getters + setters for convenience (TEST ONLY, CPU MEMORY ONLY)
  float get(int a, int b, int c, int d);              // Retrieve vals[a][b][c][d]
  void set(int a, int b, int c, int d, float val);    // Set vals[a][b][c][d] = val
};


#endif
