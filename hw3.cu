/*
KAM PUI SO (ANTHONY)
CS 510 GPU
Homework 3
*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 40
#define HEIGHT 40
#define MASK_WIDTH 3
#define MASK_HEIGHT 3
#define MAX 2000
#define LIMIT 512
#define RANGE 10
#define ROW 0
#define COL 1

// global const



/*

/* The kernel that will execute on the GPU */
__global__ void basic_2d_kernel(int *board, int *result, int width, int height) {
}

/* This function encapsulates the process of creating and tearing down the
 * environment used to execute our game of life iteration kernel. The steps of the
 * process are:
 *   1. Allocate memory on the device to hold our board vectors
 *   2. Copy the board vectors to device memory
 *   3. Execute the kernel
 *   4. Retrieve the result board vector from the device by copying it to the host
 *   5. Free memory on the device
 */
void basic_2d_dev(int *board, int *result, int width, int height) {
    // Step 1: Allocate memory
    int *board_dev, *result_dev;
    int n = width * height;

    // Since cudaMalloc does not return a pointer like C's traditional malloc
    // (it returns a success status instead), we provide as it's first argument
    // the address of our device pointer variable so that it can change the
    // value of our pointer to the correct device address.
    cudaMalloc((void **) &board_dev, sizeof(int) * n);
    cudaMalloc((void **) &result_dev, sizeof(int) * n);

    // Step 2: Copy the input vectors to the device
    cudaMemcpy(board_dev, board, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 3: Invoke the kernel
    // We allocate enough blocks (each 512 threads long) in the grid to
    // accomodate all `n` elements in the vectors. The 512 long block size
    // is somewhat arbitrary, but with the constraint that we know the
    // hardware will support blocks of that size.
    dim3 dimGrid((n + LIMIT - 1) / LIMIT, 1, 1);
    dim3 dimBlock(LIMIT, 1, 1);
    step_kernel<<<dimGrid, dimBlock>>>(board_dev, result_dev, width, height);

    // Step 4: Retrieve the results
    cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(board_dev);
    cudaFree(result_dev);
}



/* The old-fashioned CPU-only way */
void basic_2d_host(int *current, int *next, int width, int height) {

}

*/



// fill the mask with random values
void fill_image(int *image, int width, int height) {
    int i;
    for (i = 0; i < (width * height); i++)
        image[i] = rand() % RANGE;
}


// print mask image
void print_image(int *image, int width, int height) {
    int x, y;
    for (y = 0; y<height; y++) {
        for (x = 0; x<width; x++) {
            printf("%c", image[y * width + x]);
        }
        printf("\n");
    }
    printf("-----\n");
}


// main function
int main(void) {
    // variable
    int n = WIDTH * HEIGHT;
    int m = MASK_WIDTH * MASK_HEIGHT;
    int *start = (int *) malloc(n* sizeof(int));
    int *mask = (int *) malloc(n * sizof(int));
    int *result = (int *) malloc(n * sizeof(int));

    // initialize the global "current"
    fill_image(mask, MASK_WIDTH, MASK_HEIGHT);
    print_image(mask, MASK_WIDTH, MASK_HEIGHT);
    fill_image(start, WIDTH, HEIGHT);
    print_image(start, WIDTH, HEIGHT);

    // free memory
    free(start);
    free(mask);
    free(result);

    return 0;
}





