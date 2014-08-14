/*
Kam Pui So (Anthony)
CS510 GPU
Homework 5 Part 2
Problem 14.1
*/

#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX 5
#define DIMX 10
#define DIMM 12
#define DIMY 8

// create randomize matrix
void createRandomMatrix(unsigned int *sourceMatrix, const int height, const int width) {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
		//		(*sourceMatrix).elements[(y * width) + x] = ((float) x+y) * 0.1;
		sourcematrix[(y * width) + x] = (unsigned int) rand() % MAX;
		}
	}
}

// print divider
void printDivider() {
	printf("-------------------------------\n");
}

// print matrix
void printMatrix(const unsigned int *valueMatrix, const int height, const int width) {
	for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
					printf("%d ", valueMatrix[(y * width) + x]);
			}
			printf("\n");
	}
	printDivider();
}

// main
int main (int argc, char** argv) {
	int height = DIMY;
	int width = DIMX;
	int intrim = DIMM;

	unsigned int A[height*intrim];
	unsigned int B[intrim*width];
	unsigned int C[height*width];

	// create matrix A & B
	createRandomMatrix(A, height, intrim);
	createRandomMatrix(B, intrim, width);

	// print matrix
	printMatrix(A, height, intrim);
	printMatrix(B, intrim, width);

	return 0;
}
