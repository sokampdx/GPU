/*
KAM PUI SO
CS510 GPU
Homework 4
*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMATOM 32 
#define DIMX 64
#define DIMY 64
#define DIMZ 64
#define MAXW 10
#define SPACE 2
#define TILESIZE 16

// data structure
typedef struct {
	float w;	// charge
  float x;
	float y;
	float z;
} AtomInfo;


// global variable
AtomInfo atominfo[NUMATOM];


// __constant__ AtomInfo atominfo[NUMATOM];



// kernal
__global__ void cenergy_kernel(float *energygrid, dim3 grid, float gridspacing, float z, AtomInfo *atominfo, int numatoms) {

	int xindex = blockIdx.x * blockDim.x + threadIdx.x;
	int yindex = blockIdx.y * blockDim.y + threadIdx.y;
	int k = z / gridspacing;
	int outaddr = grid.x * grid.y * k + grid.x * yindex + xindex;

	float curenergy = energygrid[outaddr];
	float coorx = gridspacing * (float) xindex;
	float coory = gridspacing * (float) yindex;
	int atomid;
	float energyval = 0.0f;

	for (atomid = 0; atomid < numatoms; ++atomid) {
		float dx = coorx - atominfo[atomid].x;
		float dy = coory - atominfo[atomid].y;
//		float dz = z - atominfo[atomid].z;
		energyval += atominfo[atomid].w / sqrtf(dx*dx + dy*dy + atominfo[atomid].z);
	}

	energygrid[outaddr] = curenergy + energyval;
}


// host
void cenergy_dev(float *energygrid, dim3 grid, float gridspacing, float z, AtomInfo *atominfo, int numatoms) {

	// Step 1: allocate memory
	float * dev_energygrid;
	AtomInfo * dev_atominfo;
	int gridSize = grid.x * grid.y * grid.z;

	cudaMalloc((void **) &dev_energygrid, sizeof(float) * gridSize);
	cudaMalloc((void **) &dev_atominfo, sizeof(AtomInfo) * numatoms);

	// Step 2: copy the input vector to the device
	cudaMemcpy(dev_atominfo, atominfo, sizeof(AtomInfo) * numatoms, cudaMemcpyHostToDevice);
	
	// Step 3: Invoke the kernel
	dim3 dimGrid(TILESIZE, TILESIZE, 1);
	dim3 dimBlock(ceil(grid.x / (float) TILESIZE), ceil(grid.y / (float) TILESIZE), 1);
	cenergy_kernel<<<dimGrid, dimBlock>>>(dev_energygrid, grid, gridspacing, z, dev_atominfo, numatoms);

	// Step 4: Retrieve the results
	cudaMemcpy(energygrid, dev_energygrid, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);

	// Step 5: Free device memory
	cudaFree(dev_energygrid);
	cudaFree(dev_atominfo);
}


// initialize atominfo
void initialize() {
	for (int i = 0; i < NUMATOM; ++i) {
		atominfo[i].w = rand() % MAXW;
		atominfo[i].x = rand() % DIMX;
		atominfo[i].y = rand() % DIMY;
		atominfo[i].z = rand() % DIMZ;
	}
}

/*
// uniform charge
void uniformCharge() {
	for (int i = 0; i < NUMATOM; ++i) {
		atominfo[i].w = 1.0f;
		atominfo[i].x = rand() % DIMX;
		atominfo[i].y = rand() % DIMY;
		atominfo[i].z = rand() % DIMZ;
	}
}
*/

// print atoms
void printAtoms() {
	for (int i = 0; i < NUMATOM; ++i) {
		printf("index=%d, charge=%.2f, x=%.2f, y=%.2f, z=%.2f\n", i, atominfo[i].w, atominfo[i].x, atominfo[i].y, atominfo[i].z);
	}
}


// serial energy calculation
void serial(float *energygrid, dim3 grid, float gridspacing, float z, int numatom) {
	int i, j, n;
	int k = z / gridspacing;
	float x, y, energy;

	for (j = 0; j < grid.y; ++j) {
		y = gridspacing * (float) j;
		for (i = 0; i < grid.x; ++i) {
			x = gridspacing * (float) i;
			energy = 0.0f;
			for (n = 0; n < numatom; ++n) {
				float	dx = x - atominfo[n].x;
				float dy = y - atominfo[n].y;
				float dz = z - atominfo[n].z;
				energy += atominfo[n].w / sqrtf(dx*dx + dy*dy + dz*dz);
			}
			energygrid[grid.x * grid.y * k + grid.x * j + i] = energy;
		}
	}
}


// print energy grid
void printEnergy(float *energygrid, dim3 grid, float gridspacing) {
	for (int z = 0; z < grid.z; ++z) {
		for (int y = 0; y < grid.y; ++y) {
			for (int x = 0; x < grid.x; ++x) {
				printf("x=%d, y=%d, z=%d, potential=%.2f\n", x, y, z, energygrid[grid.x *grid.y * z + grid.x * y + x]);
			}
		}
	}
}



// energy main
void energy() {
	dim3 grid(DIMX/SPACE, DIMY/SPACE, DIMZ/SPACE);
	float gridspacing = (float) SPACE;
	float energygrid[DIMX/SPACE * DIMY/SPACE * DIMZ/SPACE];
	float	z = 0.0f;	

	for (int i = 0; i < grid.z; ++i) {
		z = gridspacing * (float) i;

		cenergy_dev(energygrid, grid, gridspacing, z, atominfo, NUMATOM);
//		serial(energygrid, grid, gridspacing, z, NUMATOM);
	}
	printEnergy(energygrid, grid, gridspacing);
}

// main
int main(void) {
	
	// initialize
	srand(time(NULL));
	initialize();		

	printAtoms();
  energy();

	return 0;
}






