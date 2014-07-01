/*
KAM PUI SO (ANTHONY)
CS 510 GPU
Homework 1

The Cross-Over Point

CUDA really shines when given problems involving lots of data, but for small problems, using CUDA can be slower than a pure CPU solution. Since it can be difficult to get a feel for how large a problem needs to be before using the GPU becomes useful, this lab encourages you to find the "crossover point" for vector addition. Specifically: how large do the vectors need to be for the speed of GPU vector addition to eclipse the speed of CPU vector addition?

Modify the vector_addition.cu example to time how long it takes the CPU and GPU vector addition functions to operate on vectors of different magnitudes. Find (roughly) what magnitude constitutes the cross-over point for this problem on your system.

*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>


/* The old-fashioned CPU-only way to add two vectors */
void add_vectors_host(int *result, int *a, int *b, int n) 
{
    for (int i=0; i<n; i++)
        result[i] = a[i] + b[i];
}

/* The kernel that will execute on the GPU */
__global__ void add_vectors_kernel(int *result, int *a, int *b, int n) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // If we have more threads than the magnitude of our vector, we need to
    // make sure that the excess threads don't try to save results into
    // unallocated memory.
    if (idx < n)
        result[idx] = a[idx] + b[idx];
}

/* This function encapsulates the process of creating and tearing down the
 * environment used to execute our vector addition kernel. The steps of the
 * process are:
 *   1. Allocate memory on the device to hold our vectors
 *   2. Copy the vectors to device memory
 *   3. Execute the kernel
 *   4. Retrieve the result vector from the device by copying it to the host
 *   5. Free memory on the device
 */
void add_vectors_dev(int *result, int *a, int *b, int n) 
{
    // Step 1: Allocate memory
    int *a_dev, *b_dev, *result_dev;

    // Since cudaMalloc does not return a pointer like C's traditional malloc
    // (it returns a success status instead), we provide as it's first argument
    // the address of our device pointer variable so that it can change the
    // value of our pointer to the correct device address.
    cudaMalloc((void **) &a_dev, sizeof(int) * n);
    cudaMalloc((void **) &b_dev, sizeof(int) * n);
    cudaMalloc((void **) &result_dev, sizeof(int) * n);

    // Step 2: Copy the input vectors to the device
    cudaMemcpy(a_dev, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 3: Invoke the kernel
    // We allocate enough blocks (each 512 threads long) in the grid to
    // accomodate all `n` elements in the vectors. The 512 long block size
    // is somewhat arbitrary, but with the constraint that we know the
    // hardware will support blocks of that size.
    dim3 dimGrid((n + 512 - 1) / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    add_vectors_kernel<<<dimGrid, dimBlock>>>(result_dev, a_dev, b_dev, n);

    // Step 4: Retrieve the results
    cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(result_dev);
}

void print_vector(int *array, int n) 
{
    int i;
    for (i=0; i<n; i++)
        printf("%d ", array[i]);
    printf("\n");
}

void print_time(timeval start, timeval end)
{
    printf("Time = ");
    printf("%ld", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
    printf("\n");
}

int main(void) 
{
    int n = 5; // Length of the arrays
    int a[] = {0, 1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    int host_result[5];
    int device_result[5];

    struct timeval start, end;

    int deviceCount;
    int device;
    
    // show cuda capability
    cudaGetDeviceCount(&deviceCount);
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
               device, deviceProp.major, deviceProp.minor);
    }

    // print answers:

    printf("The CPU's answer: ");
    gettimeofday(&start, NULL);
    add_vectors_host(host_result, a, b, n);
    gettimeofday(&end, NULL);
    print_vector(host_result, n);
    print_time(start, end);

    printf("The GPU's answer: ");
    gettimeofday(&start, NULL);
    add_vectors_dev(device_result, a, b, n);
    gettimeofday(&end, NULL);
    print_vector(device_result, n);
    print_time(start, end); 

    return 0;
}

