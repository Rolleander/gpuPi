#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

typedef unsigned long long Counter;
const Counter WARP_SIZE = 32; 
const Counter BLOCKS = 2048; 
const Counter ITERATIONS = 1000; 

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
		float x = curand_uniform(&curandom) ;
		float y = curand_uniform(&curandom) ;
		float distanceToCenter = sqrt(x*x + y*y);
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

int main() {
	printf("Init\n");
	Counter* cpu_pointsInCircle = (Counter*)malloc(BLOCKS * sizeof(Counter));
	Counter* gpu_pointsInCircle;
	Counter circleHits = 0;
	Counter throwCount = BLOCKS * WARP_SIZE * ITERATIONS;
	cudaMalloc((void**)&gpu_pointsInCircle, BLOCKS * sizeof(Counter));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Call Kernel\n");
	cudaEventRecord(start);
	kernel << <BLOCKS, WARP_SIZE >> >(gpu_pointsInCircle);
	cudaEventRecord(stop);
	cudaMemcpy(cpu_pointsInCircle, gpu_pointsInCircle, BLOCKS * sizeof(Counter), cudaMemcpyDeviceToHost);
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
	cudaFree(gpu_pointsInCircle);
	//keep the console open
	std::cin.get();
	return 0;
}
