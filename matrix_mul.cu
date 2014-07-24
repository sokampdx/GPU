/*
Kam Pui So (Anthony)
CS510 GPU
Project Group A

Application:
Matrix Multiplication based on CUDA TOOLKIT Documentation

This version of matrix multiplication does not use share memory.
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define SCALE 3.14159
#define MAX 9
#define REPEAT 10 

//global
const int TESTSIZE[] = {1, 5, 7, 11, 13, 16, 23, 32, 64};


// Row major matrix struct
typedef struct {
	int width;
	int height;
	float* elements;
} matrix;


// block size struct
typedef struct {
	int x;
	int y;
} blocksize;


// forward declaration
// matrix multiplication kernel
__global__ void matrixMultiplyKernel (const matrix, const matrix, matrix);


// print error code
void printError(char * message, cudaError_t error) {
	char errorString[255];
	strcpy(errorString, cudaGetErrorString(error));
	if (strcmp(errorString, "no error") == 1)
		printf("%s: %s\n", message, cudaGetErrorString(error));
}


// Host code - matrix multiplication
// AxB = C
// block size is determine at runtime
void matrixMultiplyHost (const matrix A, const matrix B, matrix C, const blocksize dimension) {
	// variable declaration
	matrix A_device, B_device, C_device;
	size_t size;
	cudaError_t err;

	// load A and B to device memory
	A_device.width = A.width;
	A_device.height = A.height;
	size = A.width * A.height * sizeof(float);
	err = cudaMalloc(&A_device.elements, size);
	printError("CUDA malloc A", err);
	err = cudaMemcpy(A_device.elements, A.elements, size, cudaMemcpyHostToDevice);
	printError("Copy A to device", err);

	B_device.width = B.width;
	B_device.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&B_device.elements, size);
	printError("CUDA malloc B", err);
	err = cudaMemcpy(B_device.elements, B.elements, size, cudaMemcpyHostToDevice);
	printError("Copy B to device", err);


	// allocate C in device memory
	C_device.width = C.width;
	C_device.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&C_device.elements, size);
	printError("CUDA malloc C", err);

	// invoke kernel
	dim3 dimBlock(dimension.x, dimension.y);
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y -1) / dimBlock.y);
	matrixMultiplyKernel<<<dimGrid, dimBlock>>>(A_device, B_device, C_device);
	err = cudaThreadSynchronize();
	printError("Run kernel", err);

	// read C back from device memory
	err = cudaMemcpy(C.elements, C_device.elements, size, cudaMemcpyDeviceToHost);
	printError("Copy C off of device", err);

	// free device memory 
	cudaFree(A_device.elements);
	cudaFree(B_device.elements);
	cudaFree(C_device.elements);
}


// Kernel code - matrix multiplication
// AxB = C
__global__ void matrixMultiplyKernel (const matrix A, const matrix B, matrix C) {
	// each thread compute one element in C
	int height = A.height;
	int width = B.width;
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// make sure we have a valid matrix C element
	if ((row > A.height) || (col > B.width)) return;

	float value = 0.0;
	int limit = A.width;

	for (int k = 0; k < A.width; ++k) {
		value += A.elements[row + A.width + k] * B.elements[k * B.width + col];
	}
	C.elements[row * C.width + col] = value;

}



// create random matrix
void createRandomMatrix(matrix randomMatrix) {
	int height = randomMatrix.height;
	int width = randomMatrix.width;

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			randomMatrix.elements[i * width + j] = 1.123;
//			randomMatrix.elements[i * width + j] = ((float) rand()) / 1.123;
} 


// print matrix
void printMatrix(const matrix sourceMatrix) {
	int height = sourceMatrix.height;
	int width = sourceMatrix.width;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			printf("%.2f ", sourceMatrix.elements[i * width + j]);
		}
		printf("\n");
	}
	printf("------------------------\n");
}


// print result
void printResult(const timeval start, const timeval end, const blocksize testSize) {
	printf("Result (x y micro-second), %d, %d, %ld\n", testSize.x, testSize.y, ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec )));
}



// main function
int main (int argc, char* argv[]) {
	matrix A, B, C;
	blocksize currentSize;
	int i = 0;
	int x, y;
	struct timeval start, end;

	// initialize random seed
	srand(time(NULL));

	// setup the matrices
	A.height = atoi(argv[1]);
	A.width = atoi(argv[2]);
	A.elements = (float*) malloc(A.width * A.height *sizeof(float));

	B.height = A.width;
	B.width = atoi(argv[3]);
	B.elements = (float*) malloc(B.width * B.height *sizeof(float));

	C.height = A.height;
	C.width = B.width;
	C.elements = (float*) malloc(C.width * C.height *sizeof(float));

	// create random matrix for calculation
	createRandomMatrix(A);
	createRandomMatrix(B);
	printMatrix(A);
	printMatrix(B);

	// main loop for testingg (randomly picking x & y)
	while (i < REPEAT) {
		x = rand() % MAX;
		y = rand() % MAX;
		currentSize.x = TESTSIZE[x];
		currentSize.y = TESTSIZE[y];


		// call host code
		gettimeofday(&start, NULL);
		matrixMultiplyHost(A, B, C, currentSize);
		gettimeofday(&end, NULL);
		printResult(start, end, currentSize);
		printMatrix(C);

		++i;
	}

	// free memory
	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}



