/*
Kam Pui So (Anthony)
CS510 GPU
Homework #5 Part 1
kernel text
*/

// MatrixMul Kernel
__kernel void matrixMul(__global unsigned int *C, __global unsigned int *A, __global unsigned int *B, 
												int height, int width, int intrim) {
	int col = get_global_id(0);
	int row = get_global_id(1);

	unsigned int current = 0;
	int k;

	for (k = 0; k < intrim; ++k) {
		current += A[row * intrim + k] * B[k * width + col];
	}

	C[row * width + col] = current;
}
