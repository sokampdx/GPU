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
const int SIZE = 11;
const float MAX_FLOAT = 100.0f;

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

/*
// setup matrix
void setupMatrix(matrix *sourceMatrix, int dimX, int dimY) {
	(*sourceMatrix).height = dimX;
	(*sourceMatrix).width = dimY;
	(*sourceMatrix).elements = (float*) malloc(dimX * dimY * sizeof(float));

	createRandomMatrix(sourceMatrix);
}
*/


// print matrix
void printMatrix(matrix valueMatrix) {
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
void addMatrix(matrix A, matrix B, matrix result) {
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

// main function
// usage ./a.out dimensionX dimensionY
int main (int argc, char*argv[]) {
	matrix A, B, C;
	blocksize currentSize;
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

	// tranditional addition
	addMatrix(A, B, C);
	printMatrix(C);

	// free matrix
	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}














