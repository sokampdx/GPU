/*
KAM PUI SO (ANTHONY)
CS 510 GPU
Homework 2

The Game of Life
Rules:

Any live cell with fewer than two live neighbours dies, as if caused by under-population.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overcrowding.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 32 
#define HEIGHT 32
#define MAX 50
#define LIMIT 512
#define TILE 32
#define NEIGHBORS 8
#define ROW 0
#define COL 1

// global const
const int offsets[NEIGHBORS][2] = {{-1, 1},{0, 1},{1, 1},
                                   {-1, 0},       {1, 0},
                                   {-1,-1},{0,-1},{1,-1}};

__constant__ int offsets_dev[NEIGHBORS][2] = {{-1, 1},{0, 1},{1, 1},
                                              {-1, 0},       {1, 0},
                                              {-1,-1},{0,-1},{1,-1}};

/* The kernel that will execute on the GPU */
__global__ void step_kernel(int *board, int *result, int width, int height) {

    __shared__ int num_col[TILE];

    int n = width * height;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int num_neighbors = 0;
    int nx = 0;
    int ny = 0;
    int x = idx % width;
    int y = idx / width;
    int center = board[idx];
    int i = 0;

    num_col[x] = 0;

    __syncthreads();

    for (i = -1; i <= 1; ++i) {
//        nx = (x + offsets_dev[i][ROW] + width) % width;
        ny = (y + i + height) % height;
        num_col[x] += board[ny * width + x];
    }
   
    __syncthreads();          

    for (i = 1; i >= -1; --i) {
        nx = (x + i + TILE) % TILE;
        num_neighbors += num_col[nx];
    }

    __syncthreads();

    num_neighbors -= center;    

    // apply the Game of Life rules to this cell
    if (idx < n && ((center && num_neighbors==2) || num_neighbors==3))
        result[idx] = 1;
    else
        result[idx] = 0;

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
void step_dev(int *board, int *result, int width, int height) {
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



/* The old-fashioned CPU-only way to step thru game of life*/
void step(int *current, int *next, int width, int height) {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i = 0; i < NEIGHBORS; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][ROW] + width) % width;
                ny = (y + offsets[i][COL] + height) % height;
                if (current[ny * width + nx]) {
                    num_neighbors++;
                }
            }

            // apply the Game of Life rules to this cell
            next[y * width + x] = 0;
            if ((current[y * width + x] && num_neighbors==2) || num_neighbors==3) {
                next[y * width + x] = 1;
            }
        }
    }
}




// fill the board with random cells
void fill_board(int *board, int width, int height) {
    int i;
    for (i = 0; i < (width * height); i++)
        board[i] = rand() % 2;
}


// print board image
void print_board(int *board, int width, int height) {
    int x, y;
    for (y = 0; y<height; y++) {
        for (x = 0; x<width; x++) {
            char c = board[y * width + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}



// animate each cell
void animate(int *current, int *next, int width, int height) {
    struct timespec delay = {0, 0}; // 0.005 seconds
//    struct timespec delay = {0, 125000000}; // 0.125 seconds
//    struct timespec delay = {0, 250000000}; // 0.25 seconds
    struct timespec remaining;


//    while (1) {
    for (int i = 0; i < MAX; ++i) {
	printf("%d\n", i);
        print_board(current, width, height);
        step_dev(current, next, width, height);
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, sizeof(int) * width * height);
        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        nanosleep(&delay, &remaining);
    }
}



// main function
int main(void) {
    // variable
    int width = WIDTH;
    int height = HEIGHT;
    int n = width * height;
    int *current = (int *) malloc(n* sizeof(int));
    int *next = (int *) malloc(n * sizeof(int));

    // initialize the global "current"
    fill_board(current, width, height);
    animate(current, next, width, height);

    // free memory
    free(current);
    free(next);

    return 0;
}





