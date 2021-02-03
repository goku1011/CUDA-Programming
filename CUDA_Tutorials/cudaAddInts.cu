#include<iostream>
#include<cuda.h>

using namespace std;

__global__ void AddIntsCUDA(int *a, int *b)
{
	a[0] += b[0];
}

int main()
{

int h_a=5, h_b=9;
int *d_a, *d_b;

if(cudaMalloc(&d_a, sizeof(int)) != cudaSuccess)
{
	cout<<"Error allocating memory!"<<endl;
	return 0;
}
if(cudaMalloc(&d_b, sizeof(int)) != cudaSuccess)
{
	cout<<"Error allocating memory!"<<endl;
	return 0;
}

if(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
{
	cout<<"Error copying memory!"<<endl;
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}
if(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice) != cudaSu	ccess)
{
	cout<<"Error copying memory!"<<endl;
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}

AddIntsCUDA<<<1,1>>>(d_a, d_b);

if(cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
{
	cout<<"Error copying memory!"<<endl;
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}

cout<< "The answer is : "<<a<<endl;

cudaFree(d_a);
cudaFree(d_b);

return 0;
}
