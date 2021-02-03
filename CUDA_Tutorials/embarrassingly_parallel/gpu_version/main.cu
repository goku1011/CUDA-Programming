#include <iostream>
#include <cmath>
#include <stdlib.h> // Library for rand()
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h> // Stops underlining of __global__
#include <device_launch_parameters.h> // Stops underlining of threadIdx, etc

using namespace std;

struct float3{
  float x;
  float y;
  float z;
};

__global__ void FindClosestGPU(float3* points, int* indices, int count){

  if(count <= 1){
    return;
  }
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < count){

    float3 thispoint = points[idx];
    float distToCloset = 3.40282e38f;

    for(int curPoint=0; curPoint<count; curPoint++){

      if(curPoint == idx)continue;
      float dist = (thispoint.x-points[curPoint].x)*(thispoint.x-points[curPoint].x);
        dist += (thispoint.y-points[curPoint].y)*(thispoint.y-points[curPoint].y);
        dist += (thispoint.z-points[curPoint].z)*(thispoint.z-points[curPoint].z);

      if(dist < distToCloset){
        distToCloset = dist;
        indices[idx] = curPoint;
      }
    }
  }
}

int main(){
  srand(time(NULL));

  const int count = 10000;

  int  *h_indices = new int[count];
  float3 *h_points = new float3[count];
  float3 *d_points;
  int *d_indices;

  if(cudaMalloc(&d_points, sizeof(float3)*count) != cudaSuccess){
    cout << "Failed to allocate memory in GPU"<< endl;
    delete[] h_points;
    delete[] h_indices;
    return -1;
  }
  if(cudaMalloc(&d_indices, sizeof(int)*count) != cudaSuccess){
    cout << "Failed to allocate memory in GPU"<< endl;
    cudaFree(d_points);
    delete[] h_points;
    delete[] h_indices;
    return -1;
  }

  // Create a list of random points
  for(int i=0; i<count; i++){
    h_points[i].x = (float)((rand()%10000) - 5000);
    h_points[i].y = (float)((rand()%10000) - 5000);
    h_points[i].z = (float)((rand()%10000) - 5000);
  }

  cudaMemcpy(d_points, h_points, sizeof(float3)*count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, sizeof(int)*count, cudaMemcpyHostToDevice);

  // This variable is used to keep track of the fastest time so far
  long fastest = 2^31 - 1;

  // Run the algo 32 times, with number of threads as multiples of 32
  for(int q=32; q<=1024; q+32){
    long startTime = clock();
    FindClosestGPU <<< (count/q + 1), q >>> (h_points, h_indices count);
    long finishTime = clock();

    long runtime = finishTime-startTime;
    cout<<"Run " <<q<< " took "<< runtime <<" millis "<<endl;

    if(runtime<fastest){
      fastest=runtime;
    }
  }

  // Move results from device to host
  cudaMemcpy(h_points, d_points, sizeof(float3)*count, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indices, d_indices, sizeof(int)*count, cudaMemcpyDeviceToHost);

  cout<<"Fastest Time: "<< fastest<<endl;

  cudaFree(d_points);
  cudaFree(d_indices);
  delete[] h_points;
  delete[] h_indices;
  return 0;
}
