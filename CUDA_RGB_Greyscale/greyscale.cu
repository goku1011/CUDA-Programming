#include <math.h>
#include "utils.h"
#include <stdio.h>
#include <algorithm>

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols){

  int index_x = blockIdx.x * blockDim.x + threadIdx.x; // index along x
  int index_y = blockIdx.y * blockDim.y + threadIdx.y; // index along y

  int gridwidth = blockDim.x * gridDim.x; // total number of threads along x
  int index = index_y * gridwidth + index_x;

  // greyscale image computation
  //I = .299f * R + .587f * G + .114f * B
  greyImage[index] =  .299f * rgbaImage[index].x + .587f * rgbaImage[index].y + .114f * rgbaImage[index].z;
}

void your_rgba_to_greyscale(uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols){

  const int thread = 16;
  const dim3 blockSize ( thread, thread, 1 );
  const dim3 gridSize ( ceil(numRows/(float)thread), ceil(numCols/(float)thread), 1);

  // kernel call, launch configuration
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  // CPU code execution continues, even though the GPU execution is running. The following is
  // added to halt CPU execution until all the GPU CUDA threads have completed execution
  cudaDeviceSynchronize();
  checkCudaErrors( cudaGetLastError());
}
