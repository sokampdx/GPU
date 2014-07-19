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

NOTE where // ->>> represent the equivalent of const memory version when
I also coment locations where non-const memory version of code are located.

*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1024 
#define HEIGHT 1024
#define MASK_WIDTH 127 
#define MASK_HEIGHT 127
#define LOOPMAX 20
#define TILESIZE 32
#define RANGE 10
#define ROW 0
#define COL 1

// global const

const int DIAGMASK[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const int VERTMASK[9] = {1, 0, 0, 1, 0, 0, 1, 0, 0};

__constant__ int MASK[MASK_WIDTH * MASK_HEIGHT];

// The kernel that will execute on the GPU
__global__ void basic_2d_kernel(int *start, int *result, int width, int height, int mask_width, int mask_height) {
// __global__ void basic_2d_kernel(int *start, int *mask, int *result, int width, int height, int mask_width, int mask_height) {
    // declare kernel variable
    int center_x = blockDim.x * blockIdx.x + threadIdx.x;
    int center_y = blockDim.y * blockIdx.y + threadIdx.y;
    int current_x, current_y;
    int n_x_start_point = center_x - (mask_width / 2);
    int n_y_start_point = center_y - (mask_height / 2);
    int pvalue = 0;    

    // loop thru the mask area for one location
    for (int y = 0; y < mask_height; y++) {
        current_y = (n_y_start_point + y + height) % height;
        if ((current_y >= 0) && (current_y < height)) {
            for (int x = 0; x < mask_width; x++) {
                current_x = (n_x_start_point + x + width) % width;
                if ((current_x >= 0) && (current_x < width)) {
//                    pvalue += start[(current_y * width) + current_x] * mask[(y * mask_width) + x];    // only for non-const memory version
                    pvalue += start[(current_y * width) + current_x] * MASK[(y * mask_width) + x];
                }
            }
        }
    }
    result[(center_y * width) + center_x] = pvalue;
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
void basic_2d_dev(int *start, int *mask, int *result, int width, int height, int mask_width, int mask_height) {
    // Step 1: Allocate memory
//    int *mask_dev;    // only for non-const memory version
    int *start_dev,  *result_dev;
    int n = width * height;
    int m = mask_width * mask_height;

    // Since cudaMalloc does not return a pointer like C's traditional malloc
    // (it returns a success status instead), we provide as it's first argument
    // the address of our device pointer variable so that it can change the
    // value of our pointer to the correct device address.
    cudaMalloc((void **) &start_dev, sizeof(int) * n);
    cudaMalloc((void **) &result_dev, sizeof(int) * n);
//    cudaMalloc((void **) &mask_dev, sizeof(int) * m);    // only for non-const memory version

    // Step 2: Copy the input vectors to the device
    cudaMemcpy(start_dev, start, sizeof(int) * n, cudaMemcpyHostToDevice);
//    cudaMemcpy(mask_dev, mask, sizeof(int) * m, cudaMemcpyHostToDevice);    // only for non-const memory version
    cudaMemcpyToSymbol(MASK, mask, sizeof(int) * m);    // only for const memory version


    // Step 3: Invoke the kernel
    dim3 dimGrid(TILESIZE, TILESIZE, 1);
    dim3 dimBlock(ceil(width/ (float) TILESIZE), ceil(height/ (float) TILESIZE), 1);
//    basic_2d_kernel<<<dimGrid, dimBlock>>>(start_dev, mask_dev, result_dev, width, height, mask_width, mask_height); // only for non-const memory version
    basic_2d_kernel<<<dimGrid, dimBlock>>>(start_dev, result_dev, width, height, mask_width, mask_height);   // only for const memory version

    // Step 4: Retrieve the results
    cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(start_dev);
    cudaFree(result_dev);
//    cudaFree(mask_dev);    // only for non-const memory version
}



// The old-fashioned CPU-only way (for validation)
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



// fill the image with random values within the range
void fill_image(int *image, int width, int height, int range) {
    int i;
    for (i = 0; i < (width * height); i++)
        image[i] = rand() % range;
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


// print divider
void print_divider() {
    printf("---------------------------------------\n");
}



// print image as a width x height matrix
void print_image(int *image, int width, int height) {
    int x, y;
    for (y = 0; y<height; y++) {
        for (x = 0; x<width; x++) {
            printf("%d ", image[y * width + x]);
        }
        printf("\n");
    }
    print_divider();
}


// normalize mask image (for easy printing with large array)
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


// show device capability
void device_check() {
    int deviceCount;
    int device;

    cudaGetDeviceCount(&deviceCount);
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
/*
        print_divider();
        printf("Device %d has compute capability %d.%d\n", 
               device, deviceProp.major, deviceProp.minor);
	printf("Max Threads per Block: %d \n", deviceProp.maxThreadsPerBlock);
	printf("Max Threads for x direction per Block: %d \n", deviceProp.maxThreadsDim[0]);
	printf("Max Threads for y direction per Block: %d \n", deviceProp.maxThreadsDim[1]);
	printf("Max Threads for z direction per Block: %d \n", deviceProp.maxThreadsDim[2]);
	printf("Max Blocks for x direction per Grid: %d \n", deviceProp.maxGridSize[0]);
	printf("Max Blocks for y direction per Grid: %d \n", deviceProp.maxGridSize[1]);
	printf("Max Blocks for z direction per Grid: %d \n", deviceProp.maxGridSize[2]);
        printf("Max Warp Size: %d \n", deviceProp.warpSize);
        printf("Number of SM: %d \n", deviceProp.multiProcessorCount);
        printf("Max Threads per SM: %d \n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Number of Registers in each SM: %d \n", deviceProp.regsPerBlock);
        printf("Amount of Shared Memory Available: %zd \n", deviceProp.sharedMemPerBlock);
        printf("Amount of Constant Memory Available: %zd \n", deviceProp.totalConstMem);
        printf("Amount of Global Memory Available: %zd \n", deviceProp.totalGlobalMem);
        printf("Clock Rate: %d \n", deviceProp.clockRate);
        print_divider();
*/
    }
}


// print different of two times in nano seconds
void print_time(timeval begin, timeval end) {
    printf("Time = %ld us\n", ((end.tv_sec * 1000000 + end.tv_usec) - (begin.tv_sec * 1000000 + begin.tv_usec )));
}


// main convolution setup and tear down logics
void convolution(int width, int height, int mask_width, int mask_height) {
    // image variable
    int n = width * height;
    int *start = (int *) malloc(n* sizeof(int));
    int *result = (int *) malloc(n * sizeof(int));

    // mask variable 
    int m = mask_width * mask_height;
    int *mask = (int *) malloc(m * sizeof(int));
//    int *mask = (int *) VERTMASK;    // static mask with vertical 1's       
//    int *mask = (int *) DIAGMASK;    // static mask with diagonal 1's

    // time variable
    struct timeval begin, end;

    // initialize the mask image and global image
//    print_divider();
    fill_image(mask, mask_width, mask_height, RANGE);
//    print_image(mask, mask_width, mask_height);
    fill_image(start, width, height, RANGE);
//    fill_pattern(start, width, height, RANGE);
//    print_image(start, width, height);

    // run 2d convulotion with timer and print result
    gettimeofday(&begin, NULL);
//    basic_2d_host(start, mask, result, width, height, mask_width, mask_height);    // host only code for validation
    basic_2d_dev(start, mask, result, width, height, mask_width, mask_height);
    gettimeofday(&end, NULL);
//    print_image(result, width, height);    
    print_time(begin, end);
//    print_divider();

    // free memory
    free(start);
    free(result);
    free(mask);
}


// main function
int main(void) {
    // variables

    // initialize rand seed
    srand(time(NULL));

    // check device property (warm up device...)
    device_check();

    // test convolution
    convolution(WIDTH, HEIGHT, MASK_WIDTH, MASK_HEIGHT);

    return 0;
}
    
