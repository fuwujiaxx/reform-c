GPU = 0
CUDNN = 0
OPENCV = 0
DEBUG = 0

ARCH = 	-gencode arch=compute_30, code=sm_30 \
		-gencode arch=compute_35, code=sm_35 \
		-gencode arch=compute_50, code=[sm_50, compute_50] \
		-gencode arch=compute_52, code=[sm_52, compute_52]



EXEC = reform-c

CC = gcc
CXX = g++
NVCC = nvcc

CUDA_INCLUDE = /usr/local/cuda-11.8/targets/x86_64-linux/include/

LDFLAGS = -lm -pthread
COMMON = -Iinclude/ -Isrc/