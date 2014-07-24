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


#define SCALE 3.14159
#define MAX = 9
#define REPEAT = 2000

//global
const int TESTSIZE[MAX] = {1, 5, 7, 11, 13, 16, 23, 32, 64};


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
void printError(const string, const cudaError);



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
	dim3 dimGrid(ceil(B.width / dimBlock.x), ceil(A.height / dimBlock.y));
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
	if ((row < height) && (col < width)) {
		float value = 0;
		int limit = A.width;

		for (int k = 0; k < limit; ++k) {
			value += A.elements[row + limit + k] * B.elements[k * width + col];
		}
		C.elements[row * width + col] = value;
	}
}


// print error code
void printError(const string message, const cudaError_t error) {
	printF("%s: %s\n", message, cudaGetErrorString(error));
}



// create random matrix
void createRandomMatrix(matrix randomMatrix) {
	int height = randomMatrix.height;
	int width = randomMatrix.width;

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			randomMatrix.elements[i * width + j] = rand() / SCALE;
} 


// This function print out the different in time
long int printTime(const timeval start, const timeval end) {
    return ((end.tv_sec * 1000000L + end.tv_usec) - (start.tv_sec * 1000000L + start.tv_usec ));
}


// print result
void printResult(const timeval start, const timeval end, const blocksize testSize) {
	printf("Result (x y micro-second), %s, %s, %ld:", testSize.x, testSize.y);
	printTime(start, end);
}



// main function
int main (int argc, char* argv[]) {
	matrix A, B, C;
	blocksize currentSize;
	int i = 0;
	struct timeval start, end;

	// initialize random seed
	srand(time(NULL));
	
	// setup the matrices
	A.height = atoi(argv[1]);
	A.width = atoi(argv[2]);
	A.elements = (float*) malloc(A.width * A.height *sizeof(float));

	B.height = width_A;
	B.width = atoi(argv[3]);
	B.elements = (float*) malloc(B.width * B.height *sizeof(float));

	C.height = A.height;
	C.width = B.width;
	C.elements = (float*) malloc(C.widht * C.height *sizeof(float));

	// create random matrix for calculation
	createRandomMatrix(A);
	createRandomMatrix(B);

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
		printResult(C, currentSize);

		++i;
	}

/*
	for (int x = 0; x < MAX; ++x) {
		for (int y = 0; y < MAX; ++y) {
			currentSize.x = TESTSIZE[x];
			currentSize.y = TESTSIZE[y];

		// call host code
		gettimeofday(&start, NULL);
		matrixMultiplyHost(A, B, C, currentSize);
		gettimeofday(&end, NULL);
		printResult(C, currentSize);
		}
	}
*/

	return 0;
}



