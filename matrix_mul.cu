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
const float MAX_FLOAT = 3.14f;
const int REPEAT = 9900;

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


// sequential matrix multiplication
void mulMatrix(const matrix A, const matrix B, matrix result) {
	int height = result.height;
	int intrim = A.width;
	int width = result.width;
	float value = 0.0f;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			value = 0.0f;
			for (int i = 0; i < intrim; ++i) {
				value += A.elements[y * intrim + i] * B.elements[i * width + x];
			}
			result.elements[y * width + x] = value;
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


// Kernel code - matrix multiplication
// A x B = C
__global__ void matrixMulKernel(const matrix A, const matrix B, matrix C) {
	int height = C.height;
	int intrim = A.width;
	int width = C.width;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float value = 0;

	// check if row & col are within matrix size
	if ((row > height) || (col > width)) return;

	for (int i = 0; i < intrim; ++i)
		value += A.elements[row * intrim + i] * B.elements[i * width + col];	

	C.elements[row * width + col] = value; 
}


// Host code - matrix multiplicattion
// A x B = C
// block size is determine at runtime
void matrixMulHost(const matrix A, const matrix B, matrix C, const blocksize dimension) {
	// variable declaration
	matrix A_device, B_device, C_device;
	cudaError_t err;
	int height = C.height;
	int intrim = A.width;
	int width = C.width;
	size_t size;

	A_device.width = B_device.height = intrim;
	B_device.width = C_device.width = width;
	A_device.height = C_device.height = height;

	// load A and B to device memory
	size = height * intrim * sizeof(float);
	err = cudaMalloc(&A_device.elements, size);
	printError("CUDA malloc A", err);
	err = cudaMemcpy(A_device.elements, A.elements, size, cudaMemcpyHostToDevice);
	printError("Copy A to device", err);

	size = intrim * width * sizeof(float);
	err = cudaMalloc(&B_device.elements, size);
	printError("CUDA malloc B", err);
	err = cudaMemcpy(B_device.elements, B.elements, size, cudaMemcpyHostToDevice);
	printError("Copy B to device", err);

	// allocate C in device memory
	size = height * width * sizeof(float);
	err = cudaMalloc(&C_device.elements, size);
	printError("CUDA malloc C", err);
	
	// invoke kernel
	dim3 dimBlock(dimension.x, dimension.y);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
	matrixMulKernel<<<dimGrid, dimBlock>>>(A_device, B_device, C_device);
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

// print result
void printResult(const timeval start, const timeval end, const blocksize testSize) {
printf("Result (x y micro-second), %d, %d, %ld\n", testSize.x, testSize.y, ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec )));
}


// run sizing test on kernel
void runSizeTest(const matrix A, const matrix B, matrix C) {
	blocksize currentSize;
	int i = 0;
	int x, y;
	struct timeval start, end;

	// set up test loop

	while ( i < REPEAT) {
		x = rand() % MAX_TEST;
		y = rand() % MAX_TEST;
		currentSize.x = TESTSIZE[x];
		currentSize.y = TESTSIZE[y];

//		currentSize.x = currentSize.y = 16;
		
		gettimeofday(&start, NULL);
		matrixMulHost(A, B, C, currentSize);
		gettimeofday(&end, NULL);
		printResult(start, end, currentSize);
//		printMatrix(C);

		++i;
	}
}


// main function
// usage ./a.out A.height A.width B.width
int main (int argc, char*argv[]) {
	matrix A, B, C;
	int dimX = atoi(argv[1]);
	int dimM = atoi(argv[2]);
	int dimY = atoi(argv[3]);

	// initialize random seed
	srand(time(NULL));

	// setup initial matrix
	A.height = dimX;
	A.width = dimM;
	A.elements = (float*) malloc(dimX * dimM * sizeof(float));

	B.height = dimM;
	B.width = dimY;
	B.elements = (float*) malloc(dimM * dimY * sizeof(float));

	C.height = dimX;
	C.width = dimY;
	C.elements = (float*) malloc(dimX * dimY * sizeof(float));

	// create random matrix
	createRandomMatrix(A);
	createRandomMatrix(B);

	// print initial matrix
//	printMatrix(A);
//	printMatrix(B);

	// tranditional addition
//	mulMatrix(A, B, C);

	// CUDA addition
	runSizeTest(A, B, C);

//	printMatrix(C);

	// free matrix
	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}















