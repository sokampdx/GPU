/*
Kam Pui So (Anthony)
CS510 GPU
Homework 5 Part 2
Problem 14.1

References:
Kirk & Hwu. (2013). Programming Massively Parallel Processors
Zalius (2009). OpenCL Examples. http://gpgpu-computing4.blogspot.com/2009/09/opencl-program-structure.html
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
	int x, y;

	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
		//		(*sourceMatrix).elements[(y * width) + x] = ((float) x+y) * 0.1;
		sourceMatrix[(y * width) + x] = (unsigned int) rand() % MAX;
		}
	}
}

// print divider
void printDivider() {
	printf("-------------------------------\n");
}

// print matrix
void printMatrix(const unsigned int *valueMatrix, const int height, const int width) {
	int x, y;

	for (y = 0; y < height; ++y) {
			for (x = 0; x < width; ++x) {
					printf("%d ", valueMatrix[(y * width) + x]);
			}
			printf("\n");
	}
	printDivider();
}


// opencl host code
void opencl_host(const unsigned int *A, const unsigned int *B, unsigned int *C, 
								 const int height, const int width, const int intrim) {

	// Create OpenCL Context
	cl_int cl_err = CL_SUCCESS;
	cl_context cl_ctx = clCreateContextFromType(0, CL_DEVICE_TYPE_ALL, NULL, NULL, &cl_err);

	size_t parms_z;
	cl_err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, NULL, &parms_z);

	cl_device_id * cl_devs = (cl_device_id*) malloc(parms_z);
	cl_err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, parms_z, cl_devs, NULL);

	cl_command_queue cl_cmdQ = clCreateCommandQueue(cl_ctx, cl_devs[0], 0, &cl_err);

	
	// Setup device memory
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	d_A = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, sizeof(unsigned int) * (height*intrim), NULL, NULL);
	d_B = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, sizeof(unsigned int) * (intrim*width), NULL, NULL);
	d_C = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * (height*width), NULL, NULL);


	// Building OpenCL Kernel
	size_t kernelLen;
	char* cl_MatrixMul = oclLoadProgSource("matrixMulKernel.cl", " ", &kernelLen);
	cl_program cl_pgm = clCreateProgramWithSource(cl_ctx, 1, (const char**) &cl_MatrixMul, &kernelLen, &cl_err);
	cl_err = clBuildProgram(cl_pgm, 0, NULL, NULL, NULL, NULL);
	cl_kernel cl_kern = clCreateKernel(cl_pgm, "matrixMul", &cl_err);

	// Kernel Launch
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(cl_mem), (void *)&d_C);
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(cl_mem), (void *)&d_A);
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(cl_mem), (void *)&d_B);
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(int), (void *)&height);
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(int), (void *)&width);
	cl_err = clSetKernelArg(cl_kern, 0, sizeof(int), (void *)&intrim);

	size_t global_size[2], local_size[2];
	global_size[0] = 1024;
	global_size[1] = 1024;
	local_size[0] = 16;
	local_size[1] = 16;

	cl_event event;
	cl_err=clEnqueueNDRangeKernel(cl_cmdQ, cl_kern, 2, NULL, global_size, local_size, 0, NULL, NULL);
	cl_err=clWaitForEvents(1, &event);
	cl_err=clReleaseEvenet(event);
	cl_EnqueueReadBuffer(cl_cmdQ, d_C, CL_TRUE, 0, sizeof(unsigned int) * (height*width), C, 0, NULL, NULL);


	// free memory
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	free(cl_devs);
	free(cl_MatrixMul);
	clReleaseContext(cl_ctx);
	clReleaseKernel(cl_kern);
	clReleaseProgram(cl_pgm);
	clReleaseCommandQueue(cl_cmdQ);
}



// main
int main (int argc, char** argv) {
	int height = DIMY;
	int width = DIMX;
	int intrim = DIMM;

	unsigned int *A = (unsigned int*) malloc(sizeof(unsigned int) * (height*intrim));
	unsigned int *B = (unsigned int*) malloc(sizeof(unsigned int) * (intrim*width));
	unsigned int *C = (unsigned int*) malloc(sizeof(unsigned int) * (height*width));

	// set randome seed
	srand(time(NULL));

	// create matrix A & B
	createRandomMatrix(A, height, intrim);
	createRandomMatrix(B, intrim, width);

	// OpenCL host
	opencl_host(A, B, C, height, width, intrim);

	// print matrix
	printMatrix(A, height, intrim);
	printMatrix(B, intrim, width);
//	printMatrix(C, height, width);

	// free memory
	free(A);
	free(B);
	free(C);

	return 0;
}
