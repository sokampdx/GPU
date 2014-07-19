/*
KAM PUI SO (ANTHONY)
CS 510 GPU
Homework 3

1. Complete problems 8.8 and 8.10 in the text (p. 196)
2. Compare the performance of the two different 2D versions of code 
   (one uses constant memory and one does not) on 6 different 
   array sizes.
Submit your code and your performance results by email to: 
   karavan@pdx.edu with subject: 
   GPU HW3 You can submit files of type .cu, .txt, .pdf, .doc 
   You can include your performance results directly in the email 
   if that is easier. 
*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 10
#define HEIGHT 10
#define MASK_WIDTH 3
#define MASK_HEIGHT 3
#define MAX 2000
#define LIMIT 100
#define RANGE 10
#define ROW 0
#define COL 1

// global const

const int DIAGMASK[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const int VERTMASK[9] = {1, 0, 0, 1, 0, 0, 1, 0, 0};
/*

// The kernel that will execute on the GPU
__global__ void basic_2d_kernel(int *board, int *result, int width, int height) {
}

// This function encapsulates the process of creating and tearing down the
// environment used to execute our game of life iteration kernel. The steps of the
// process are:
//   1. Allocate memory on the device to hold our board vectors
//   2. Copy the board vectors to device memory
//   3. Execute the kernel
//   4. Retrieve the result board vector from the device by copying it to the host
//   5. Free memory on the device
//
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

*/

// The old-fashioned CPU-only way
void basic_2d_host(int *start, int *mask, int *result, int width, int height, int mask_width, int mask_height) {

    int x;
    int y;
    int m_x;
    int m_y;
    int n_x;
    int n_y;
    int offset_x = mask_width / 2;
    int offset_y = mask_height / 2;
    int pvalue = 0;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            pvalue = 0;
            for (m_y = 0; m_y < mask_height; m_y++) {
               for (m_x = 0; m_x < mask_width; m_x++) {
                  n_x = (x + m_x - (offset_x) + width) % width;
                  n_y = (y + m_y - (offset_y) + height) % height;
                  pvalue += (start[n_y * width + n_x] * mask[m_y * mask_width + m_x]);
               }
            }
            result[y * width + x] = pvalue;
        }
    } 
}



// fill the mask with random values
void fill_image(int *image, int width, int height, int scale) {
    int i;
    for (i = 0; i < (width * height); i++)
        image[i] = rand() % scale;
}


// fill the mask with pattern values
void fill_pattern(int *image, int width, int height, int scale) {
    int i;
    for (i = 0; i < (width * height); i++) {
        if (i % (width / 2))
            image[i] = 0;
        else
            image[i] = rand() % scale;
    }
}


// print mask image
void print_image(int *image, int width, int height) {
    int x, y;
    for (y = 0; y<height; y++) {
        for (x = 0; x<width; x++) {
            printf("%d ", image[y * width + x]);
        }
        printf("\n");
    }
    printf("-----\n");
}


// normalize mask image
void normalize_image(int *image, int width, int height, int scale) {
    int i;
    int max = image[0];

    // find max and min
    for (i = 0; i < (width * height); i++) {
        if (image[i] > max)
            max = image[i];
    }

    for (i = 0; i < (width * height); i++) {
        image[i] = (int) ((float) image[i] / (float) max * (float) (scale -1)) ;
    }
}


// main function
int main(void) {
    // variable
    int n = WIDTH * HEIGHT;
    int m = MASK_WIDTH * MASK_HEIGHT;
    int *start = (int *) malloc(n* sizeof(int));
//    int *mask = (int *) malloc(m * sizeof(int));
    int *mask = (int *) VERTMASK;
    int *result = (int *) malloc(n * sizeof(int));

    int i = 0;
    int *temp;

    // initialize rand seed
    srand(time(NULL));

    // initialize the global images
//    fill_image(mask, MASK_WIDTH, MASK_HEIGHT, RANGE);
    print_image(mask, MASK_WIDTH, MASK_HEIGHT);
//    fill_image(start, WIDTH, HEIGHT, RANGE);
    fill_pattern(start, WIDTH, HEIGHT, RANGE);
    print_image(start, WIDTH, HEIGHT);
    
    while (i < LIMIT) {
        basic_2d_host(start, mask, result, WIDTH, HEIGHT, MASK_WIDTH, MASK_HEIGHT);
        normalize_image(result, WIDTH, HEIGHT, RANGE);
        print_image(result, WIDTH, HEIGHT);
        temp = result;
        result = start;
        start = temp;
        ++i;
    }

    // free memory
    free(start);
//    free(mask);
    free(result);

    return 0;
}





