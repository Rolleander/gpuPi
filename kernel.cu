#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#define BLOCKS 256
#define THREADS 512
#define LOOPS 1000000
#define N THREADS * BLOCKS
#define KERNEL_CALLS 1

__global__ void init(unsigned int seed, curandState_t* states) {
  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int index = threadId * BLOCKS + blockId;
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
    index, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
}

__global__ void kernel(curandState_t* states, unsigned int* pointsInCircle)
{
  unsigned int circleHits = 0;
  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int index = threadId * BLOCKS + blockId;
  for(unsigned int i=0; i<LOOPS; i++)
  {
    float x = curand_uniform (&states[index])*2-1;
    float y = curand_uniform (&states[index])*2-1;
    float distanceToCenter = sqrt(x*x+y*y);
    bool inCircle = distanceToCenter <=1;
    if(inCircle){
      circleHits++;
    }
  }
 // printf("kernel thread %d block %d hits %u \n",threadId, blockId, circleHits);
  pointsInCircle[index]= circleHits;
}

int main() {
     printf("Start\n");
     curandState_t* states;
     cudaMalloc((void**) &states, N * sizeof(curandState_t));

     printf("Init\n");
     init<<<BLOCKS, THREADS>>>(time(0), states);
     unsigned int cpu_pointsInCircle[N];
     unsigned int* gpu_pointsInCircle;
     unsigned long long int circleHits = 0;
     unsigned long long int throwCount = (long long) N *  (long long) LOOPS * KERNEL_CALLS;    
     cudaMalloc((void**) &gpu_pointsInCircle, N * sizeof(unsigned int));
     float milliseconds = 0;
    for(int j=0; j<KERNEL_CALLS; j++)
    {
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop); 
     printf("Call Kernel\n");
     cudaEventRecord(start);
     kernel<<<BLOCKS, THREADS>>>(states, gpu_pointsInCircle);
     cudaEventRecord(stop);
     cudaMemcpy(cpu_pointsInCircle, gpu_pointsInCircle, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
     float ms;
     cudaEventElapsedTime(&ms, start, stop);
     printf("Aggregate Results\n");
     milliseconds += ms;      
     for (int i = 0; i < N; i++) {
       circleHits = circleHits + cpu_pointsInCircle[i];       
    //   printf("circle hits %llu \n", circleHits);
     }
    }
     double pi = ((double) circleHits / (double) (throwCount) )* 4;
     printf("Finished calculating in %f seconds! \n", milliseconds/1000.0);
     printf("=>  %llu in Circle of \n",  circleHits);
     printf("=>  %llu Points \n",  throwCount);
     printf("=>  PI = %.21g \n", pi);
     printf("[PI is = 3.14159265358979323846]\n");
     cudaFree(states);
     cudaFree(gpu_pointsInCircle);
  
  //keep the console open
  std::cin.get();
  return 0;

}
