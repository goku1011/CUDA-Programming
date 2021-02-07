#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows(){
  return imageRGBA.rows;
}
size_t numCols(){
  return imageRGBA.cols;
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename){

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);

  if(image.empty()){
    std::cerr << "Cannot open file: " << filename << '\n';
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  // Allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  // Allocate memory on the device for both input and output
  checkCudaErrors( cudaMalloc( *d_rgbaImage, numPixels * sizeof(uchar4)));
  checkCudaErrors( cudaMalloc( *d_greyImage, numPixels * sizeof(unsigned char)));

  checkCudaErrors( cudaMemset( *d_greyImage, 0, numPixels * sizeof(unsigned char)));
  // copy input rgba image to the GPU
  checkCudaErrors( cudaMemcpy( *d_rgbaImage, *inputImage, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}


void postProcess(const std::string& output_file){
  const int numPixels = numCols() * numRows();

  // copy the output back to the Host
  checkCudaErrors( cudaMemcpy( imageGrey.ptr<unsigned char>(0), d_greyImage__, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  // output the image
  cv::imwrite(output_file.c_str(), imageGrey);

  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}
