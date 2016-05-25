NVCC            ?= $(CUDA_BIN_PATH)/nvcc
CC = /usr/bin/g++

OBJS = layer.o main.o convnet.o tensor.o load.o utils.o 
TARGETS = cpu

all: $(TARGETS)

cpu: $(OBJS)
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -std=c++11 -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

layer.cpp: layer.hpp utils.hpp
convnet.cpp: convnet.hpp utils.hpp
tensor.cpp: tensor.hpp utils.hpp 
load.cpp: load.hpp utils.hpp
utils.cpp: utils.hpp
