NVCC            ?= $(CUDA_BIN_PATH)/nvcc

all: cpu

cpu: layer.o load.o main.o convnet.o
	$(NVCC) -o cpu
