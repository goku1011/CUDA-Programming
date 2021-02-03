#include <iostream>
#include <cmath>
#include <stdlib.h> // Library for rand()
#include <ctime>

using namespace std;

struct float3{
  float x;
  float y;
  float z;
};

void FindClosestCPU(float3* points, int* indices, int count){

  if(count <= 1){
    return;
  }

  for(int curPoint=0; curPoint<count; curPoint++){

    float distToCloset = 3.40282e38f;
    for(int j=0; j<count; j++){

      if(j==curPoint)continue;
      float dist = (points[i].x-points[curPoint].x)*(points[i].x-points[curPoint].x);
      dist += (points[i].y-points[curPoint].y)*(points[i].y-points[curPoint].y);
      dist += (points[i].z-points[curPoint].z)*(points[i].z-points[curPoint].z);

      if(dist < distToCloset){
        distToCloset = dist;
        indices[curPoint] = i;
      }
    }
  }
}

int main(){
  srand(time(NULL));

  const int count = 10000;

  int  *indexOfClosest = new int[count];
  float3 *points = new float3[count];

  // Create a list of random points
  for(int i=0; i<count; i++){
    points[i].x = (float)((rand()%10000) - 5000);
    points[i].y = (float)((rand()%10000) - 5000);
    points[i].z = (float)((rand()%10000) - 5000);
  }

  // This variable is used to keep track of the fastest time so far
  long fastest = 1000000;

  // Run the algo 32 times
  for(int q=0; q<=32; q++){
    long startTime = clock();
    FindClosestCPU(points, indexOfClosest count);
    long finishTime = clock();

    long runtime = finishTime-startTime;
    cout<<"Run " <<q<< " took "<< runtime <<" millis "<<endl;

    if(runtime<fastest){
      fastest=runtime;
    }
  }
  cout<<"Fastest Time: "<< fastest<<endl;

  delete[] points;
  delete[] indexOfClosest;
  return 0;
}
