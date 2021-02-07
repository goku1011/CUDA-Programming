#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
using namespace std;

size_t numRows();
size_t numCols();

void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const string& filename);

void postProcess(const string& output_file);

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols);


#include "HW.cpp"

int main(int argc, char **argv){

  uchar4  *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  string input_file;
  string output_file;

  if(argc==3){
    input_file = string(argv[1]);
    output_file = string(argv[2]);
  }
  else{
    cerr << "/* error message */" << endl;
    exit(1);
  }

  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  GpuTimer timer;
  timer.Start();

  // call the grayscale code
  your_rgba_to_greyscale(d_rgbaImage, d_greyImage, numRows(), numCols());
  timer.Stop();

  // CPU code execution continues, even though the GPU execution is running. The following is
  // added to halt CPU execution until all the GPU CUDA threads have completed execution
  cudaDeviceSynchronize();
  checkCudaErrors( cudaGetLastError());

  int err = timer.Elapsed();

  if(err < 0){
    std::cerr << "Could not print timing information" << '\n';
    exit(1);
  }else{
    cout << "msecs : "<< err << endl;
  }

  postProcess(output_file);
  return 0;
}
