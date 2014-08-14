/*
KAM PUI SO
CS510 GPU
Homework 4

Problem 12.3 with const memory, memory coalescing
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

__constant__ AtomInfo CONST_ATOMINFO[NUMATOM];


// kernal
__global__ void cenergy_kernel(float *energygrid, dim3 grid, float gridspacing, float z, int numatoms) {

	int xindex = blockIdx.x * blockDim.x + threadIdx.x;
	int yindex = blockIdx.y * blockDim.y + threadIdx.y;
	int k = z / gridspacing;
	int outaddr = grid.x * grid.y * k + grid.x * yindex + xindex;

	float curenergy = energygrid[outaddr];
	float coorx = gridspacing * (float) xindex;
	float coory = gridspacing * (float) yindex;
	float gridspacing_coalesce = gridspacing * TILESIZE;
	int atomid;
	float energyvalx1, energyvalx2, energyvalx3, energyvalx4, energyvalx5, energyvalx6, energyvalx7, energyvalx8;
	energyvalx1 = energyvalx2 = energyvalx3 = energyvalx4 = 0.0f;
	energyvalx5 = energyvalx6 = energyvalx7 = energyvalx8 = 0.0f;

	for (atomid = 0; atomid < numatoms; ++atomid) {
		float dy = coory - CONST_ATOMINFO[atomid].y;
		float dyz2 = (dy*dy) + CONST_ATOMINFO[atomid].z;

		float dx1 = coorx - CONST_ATOMINFO[atomid].x;
		float dx2 = dx1 + gridspacing_coalesce;
		float dx3 = dx2 + gridspacing_coalesce;
		float dx4 = dx3 + gridspacing_coalesce;
		float dx5 = dx4 + gridspacing_coalesce;
		float dx6 = dx5 + gridspacing_coalesce;
		float dx7 = dx6 + gridspacing_coalesce;
		float dx8 = dx7 + gridspacing_coalesce;

		energyvalx1 += CONST_ATOMINFO[atomid].w / sqrtf(dx1*dx1 + dyz2);
		energyvalx2 += CONST_ATOMINFO[atomid].w / sqrtf(dx2*dx2 + dyz2);
		energyvalx3 += CONST_ATOMINFO[atomid].w / sqrtf(dx3*dx3 + dyz2);
		energyvalx4 += CONST_ATOMINFO[atomid].w / sqrtf(dx4*dx4 + dyz2);
		energyvalx5 += CONST_ATOMINFO[atomid].w / sqrtf(dx5*dx5 + dyz2);
		energyvalx6 += CONST_ATOMINFO[atomid].w / sqrtf(dx6*dx6 + dyz2);
		energyvalx7 += CONST_ATOMINFO[atomid].w / sqrtf(dx7*dx7 + dyz2);
		energyvalx8 += CONST_ATOMINFO[atomid].w / sqrtf(dx8*dx8 + dyz2);

	}

	energygrid[outaddr] += energyvalx1;
	energygrid[outaddr + TILESIZE] += energyvalx1;
	energygrid[outaddr + 2 * TILESIZE] += energyvalx2;
	energygrid[outaddr + 3 * TILESIZE] += energyvalx3;
	energygrid[outaddr + 4 * TILESIZE] += energyvalx4;
	energygrid[outaddr + 5 * TILESIZE] += energyvalx5;
	energygrid[outaddr + 6 * TILESIZE] += energyvalx6;
	energygrid[outaddr + 7 * TILESIZE] += energyvalx7;
}


// host
void cenergy_dev(float *energygrid, dim3 grid, float gridspacing, float z, AtomInfo *atominfo, int numatoms) {

	// Step 1: allocate memory
	float * dev_energygrid;
//	AtomInfo * dev_atominfo;
	int gridSize = grid.x * grid.y * grid.z;

	cudaMalloc((void **) &dev_energygrid, sizeof(float) * gridSize);
//	cudaMalloc((void **) &dev_atominfo, sizeof(AtomInfo) * numatoms);
	

	// Step 2: copy the input vector to the device
//	cudaMemcpy(dev_atominfo, atominfo, sizeof(AtomInfo) * numatoms, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CONST_ATOMINFO, atominfo, sizeof(AtomInfo) * numatoms);

	// Step 3: Invoke the kernel
	dim3 dimGrid(TILESIZE, TILESIZE, 1);
	dim3 dimBlock(ceil(grid.x / (float) TILESIZE), ceil(grid.y / (float) TILESIZE), 1);
	cenergy_kernel<<<dimGrid, dimBlock>>>(dev_energygrid, grid, gridspacing, z, numatoms);

	// Step 4: Retrieve the results
	cudaMemcpy(energygrid, dev_energygrid, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);

	// Step 5: Free device memory
	cudaFree(dev_energygrid);
//	cudaFree(dev_atominfo);
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






