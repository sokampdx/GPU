/*
Kam Pui So (Anthony)
CS510 GPU
Project Group A

Appliction:
Matrix Addition base on CUDA TOOLKIT Documentation

*/


#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//global
const int TESTSIZE[] = {1, 5, 7, 11, 13, 16, 23, 29, 32, 47, 64};
const int MAX_TEST = 11;
const float MAX_FLOAT = 100.0f;
const int REPEAT = 10;

// row major matrix struct
typedef struct {
	int width;
	int height;
	float* elements;
} matrix;

typedef struct{
	int x;
	int y;
} blocksize;


// print divider
void printDivider() {
	printf("-------------------------------\n");
}

// create randomize matrix
void createRandomMatrix(matrix sourceMatrix) {
	int height = sourceMatrix.height;
	int width = sourceMatrix.width;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
//		(*sourceMatrix).elements[(y * width) + x] = ((float) x+y) * 0.1;
			sourceMatrix.elements[(y * width) + x] = (float) rand() / (float) (RAND_MAX/MAX_FLOAT);
		}
	}
}


// print matrix
void printMatrix(const matrix valueMatrix) {
	int height = valueMatrix.height;
	int width = valueMatrix.width;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			printf("%.2f ", valueMatrix.elements[(y * width) + x]);
		}
		printf("\n");
	}
	printDivider();
}


// sequential matrix addition
void addMatrix(const matrix A, const matrix B, matrix result) {
	int height = result.height;
	int width = result.width;
	int index = 0;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			index = y * width + x;
			result.elements[index] = A.elements[index] + B.elements[index];
		}
	}
}


// print error code
void printError(char *message, cudaError_t error) {
	char errorString[255];
	strcpy(errorString, cudaGetErrorString(error));
	if (strcmp(errorString, "no error") == 1)
		printf("%s: %s\n", message, cudaGetErrorString(error));
}


// Kernel code - matrix addition
// A + B = C
__global__ void matrixAddKernel(const matrix A, const matrix B, matrix C) {
	int height = C.height;
	int width = C.width;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// check if row & col are within matrix size
	if ((row > height) || (col > width)) return;

	int index = row * width + col;

	C.elements[index] = A.elements[index] + B.elements[index]; 
}


// Host code - matrix addition
// A + B = C
// block size is determine at runtime
void matrixAddHost(const matrix A, const matrix B, matrix C, const blocksize dimension) {
	// variable declaration
	matrix A_device, B_device, C_device;
	cudaError_t err;
	int height = C.height;
	int width = C.width;
	size_t size = height * width * sizeof(float);

	A_device.width = B_device.width = C_device.width = width;
	A_device.height = B_device.height = C_device.height = height;

	// load A and B to device memory
	err = cudaMalloc(&A_device.elements, size);
	printError("CUDA malloc A", err);
	err = cudaMemcpy(A_device.elements, A.elements, size, cudaMemcpyHostToDevice);
	printError("Copy A to device", err);

	err = cudaMalloc(&B_device.elements, size);
	printError("CUDA malloc B", err);
	err = cudaMemcpy(B_device.elements, B.elements, size, cudaMemcpyHostToDevice);
	printError("Copy B to device", err);

	// allocate C in device memory
	err = cudaMalloc(&C_device.elements, size);
	printError("CUDA malloc C", err);
	
	// invoke kernel
	dim3 dimBlock(dimension.x, dimension.y);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
	matrixAddKernel<<<dimGrid, dimBlock>>>(A_device, B_device, C_device);
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





// main function
// usage ./a.out dimensionX dimensionY
int main (int argc, char*argv[]) {
	matrix A, B, C;
	blocksize currentSize;
	int x, y;
	int dimX = atoi(argv[1]);
	int dimY = atoi(argv[2]);

	// initialize random seed
	srand(time(NULL));



	// setup initial matrix
	A.height = dimX;
	A.width = dimY;
	A.elements = (float*) malloc(dimX * dimY * sizeof(float));

	B.height = dimX;
	B.width = dimY;
	B.elements = (float*) malloc(dimX * dimY * sizeof(float));

	C.height = dimX;
	C.width = dimY;
	C.elements = (float*) malloc(dimX * dimY * sizeof(float));

	// create random matrix
	createRandomMatrix(A);
	createRandomMatrix(B);

	// print initial matrix
	printMatrix(A);
	printMatrix(B);


/*
	// tranditional addition
	addMatrix(A, B, C);
*/

	// CUDA addition
	x = rand() % MAX_TEST;
	y = rand() % MAX_TEST;
	currentSize.x = TESTSIZE[x];
	currentSize.y = TESTSIZE[y];

	printf("x=%d, y=%d\n", x, y);

	matrixAddHost(A, B, C, currentSize);


	// print result
	printMatrix(C);


	// free matrix
	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}














