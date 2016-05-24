NVCC            ?= $(CUDA_BIN_PATH)/nvcc

all: cpu

cpu: layer.o load.o main.o nnet.o
	$(NVCC) -o cpu
