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

#define NUMATOM 4 
#define DIMX 4
#define DIMY 4
#define DIMZ 4
#define MAXW 10
#define SPACE 2

// data structure
typedef struct {
	float w;	// charge
  float x;
	float y;
	float z;
} AtomInfo;


// global variable
AtomInfo atominfo[NUMATOM];
dim3 grid;


//__constant__ AtomInfo atominfo[NUMATOM];


/*
// kernal
__global__ void cenergy(float *energygrid, dim3 grid, float gridspacing, float z, float *atominfo, int numatoms) {

	int xindex = blockIdx.x * blockDim.x + threadIdx.x;
	int yindex = blockIdx.y * blockDim.y + threadIdx.y;
	int k = z / gridspacing;
	int outaddr = grid.x * grid.y * k + grid.x * yindex + xindex;

	float curenergy = enerygrid[outaddr];
	float coorx = gridspacing * (float) xindex;
	float coory = gridspacing * (float) yindex;
	int atomid;
	float energyval = 0.0f;

	for (atomid = 0; atomid < numatoms; ++atomid) {
		float dx = coorx - atominfo[atomid].x;
		float dy = coory - atominfo[atomid].y;
		energyval += atominfo[atomid].w * rsqrtf(dx*dx + dy*dy + atominfo[atomid].z);
	}

	energygrid[outaddr] = curenergy + energyval;
}
*/



// initialize atominfo
void initialize() {
	for (int i = 0; i < NUMATOM; ++i) {
		atominfo[i].w = rand() % MAXW;
		atominfo[i].x = rand() % DIMX;
		atominfo[i].y = rand() % DIMY;
		atominfo[i].z = rand() % DIMZ;
	}

	grid.x = DIMX;
	grid.y = DIMY;
	grid.z = DIMZ;
}


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

	for (j = 0; j < grid.y; ++j) {
		float y = gridspacing * (float) j;
		for (i = 0; i < grid.x; ++i) {
			float x = gridspacing * (float) i;
			float energy = 0.0f;
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
	float gridspacing = SPACE;
	float energygrid[DIMX/SPACE * DIMY/SPACE * DIMZ/SPACE];
	float	z = 0.0f;	

	serial(energygrid, grid, gridspacing, z, NUMATOM);
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






