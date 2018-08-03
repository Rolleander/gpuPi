#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
typedef unsigned long long Counter;
const Counter WARP_SIZE = 32;
const Counter BLOCKS = 2048; 
const Counter ITERATIONS = 5000; 

__global__ void kernel(Counter* pointTotals)
{
	//shared counter for all the threads 
	__shared__ Counter threadTotals[WARP_SIZE];
	int uniqId = threadIdx.x + blockIdx.x * blockDim.x;
	int threadId = threadIdx.x;
	//init random
	curandState_t curandom;
	curand_init(clock64(), uniqId, 0, &curandom);
	threadTotals[threadId] = 0;
	//loop for point generation and circle check
	for (Counter i = 0; i < ITERATIONS; i++)
	{
		double x = curand_uniform(&curandom) ;
		double y = curand_uniform(&curandom) ;
		double distanceToCenter = sqrt(x*x + y*y);
		bool inCircle = distanceToCenter <= 1;
		if (inCircle) {
			threadTotals[threadId]+=1;
		}
	}
	__syncthreads();
	//thread#0 
	if (threadId == 0) {
		//aggregate the results of all threads 
		int blockId = blockIdx.x;
		pointTotals[blockId] = 0;
		for (Counter i = 0; i < WARP_SIZE; i++) {
			pointTotals[blockId] += threadTotals[i];
		}
	}
}

int main(int argc, char *argv[]) {
	if(argc!=1){
	//	printf("Wrong number of argmuments, needed: WarpSize, Blocks, Iterations\n");
	//	printf("Using default settings... \n");
	}
	else{
	//	WARP_SIZE = strtoull( argv[0], NULL, 10 );
	//	BLOCKS = strtoull( argv[1], NULL, 10 );
	//	ITERATIONS = strtoull( argv[2], NULL, 10 );
	}
	printf("WARP SIZE: %llu \n",WARP_SIZE);
	printf("BLOCKS: %llu \n",BLOCKS);
	printf("ITERATIONS: %llu \n",ITERATIONS);	
	printf("Init\n");
	int numDev;
	CUDA_CALL(cudaGetDeviceCount(&numDev));
	if (numDev < 1) {
		printf( "No CUDA device found! \n");
		return 1;
	}
	Counter* cpu_pointsInCircle = (Counter*)malloc(BLOCKS * sizeof(Counter));
	Counter* gpu_pointsInCircle;
	Counter circleHits = 0;
	Counter throwCount = BLOCKS * WARP_SIZE * ITERATIONS;
	CUDA_CALL(cudaMalloc((void**)&gpu_pointsInCircle, BLOCKS * sizeof(Counter)));
	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	printf("Call Kernel\n");
	CUDA_CALL(cudaEventRecord(start));
	kernel << <BLOCKS, WARP_SIZE >> >(gpu_pointsInCircle);
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaEventRecord(stop));
	CUDA_CALL(cudaMemcpy(cpu_pointsInCircle, gpu_pointsInCircle, BLOCKS * sizeof(Counter), cudaMemcpyDeviceToHost));
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Aggregate Results\n");
	for (Counter i = 0; i < BLOCKS; i++) {
		circleHits += cpu_pointsInCircle[i];
	}
	double pi = ((double)circleHits / (double)(throwCount)) * 4;
	printf("Finished calculating in %f seconds! \n", milliseconds / 1000.0);
	printf("=>  %llu in Circle of \n", circleHits);
	printf("=>  %llu Points \n", throwCount);
	printf("=>  PI = %.21g \n", pi);
	printf("[PI is = 3.14159265358979323846]\n");
	free(cpu_pointsInCircle);
	CUDA_CALL(cudaFree(gpu_pointsInCircle));
	//keep the console open
	std::cin.get();
	return 0;
}
