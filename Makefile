#Makefile for DBSCAN final project

#Compile for the compute capability of the GPU that you are using.
#Compute capability: K80: 37, P100: 60, V100: 70, A100: 80
#Example compilation for A100

CC=70

all: baseline

baseline:
	nvcc -O3 -arch=compute_$(CC) -code=sm_$(CC) -lcuda -lineinfo -Xcompiler -fopenmp dbscan.cu -o dbscan
