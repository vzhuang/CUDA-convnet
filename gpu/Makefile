CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-7.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
      LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS       := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif


TARGETS = convnet

all: $(TARGETS)

convnet: main.o convlayer.o fclayer.o poolinglayer.o tensor.o load.o layer.o activation.o utils.o
	$(CC) $< -o $@ main.o convlayer.o fclayer.o poolinglayer.o tensor.o load.o layer.o activation.o utils.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp -lcurand

main.o: main.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

convlayer.o: convlayer.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

fclayer.o: fclayer.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

poolinglayer.o: poolinglayer.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

tensor.o: tensor.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c

load.o: load.cpp
	$(CC) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

layer.o: layer.cpp
	$(CC) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

activation.o: activation.cpp 
	$(CC) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

utils.o: utils.cpp 
	$(CC) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
