#include <cstdio>
#include <cstdlib>
#include <vector>


__global__ void setBucketToZero(int bucket[]) {
  int i = threadIdx.x; // retrieve i from thread index
  bucket[i] = 0;
}

__global__ void incrementBucket(int bucket[], int key[]) {
  int i = threadIdx.x; // retrieve i from thread index
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void bucketSortCuda(int bucket[], int key[], int range, int offset[]) {
  int i = threadIdx.x; // retrieve i from thread index

  for (int j = 1; j<range; j<<=1) {
    offset[i] = bucket[i];
    if(i>=j) {
      bucket[i] += offset[i-j];
    }
     
  }
  
  for (int j=0; bucket[i]>0; bucket[i]--) {
    key[j++] = i;
  }
}


int main() {
  int n = 50;
  int range = 5;

  int *key;
  cudaMallocManaged(&key, n*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  int *offset;
  cudaMallocManaged(&offset, range*sizeof(int)); 

  setBucketToZero<<<1,range>>>(bucket);
  cudaDeviceSynchronize();

  incrementBucket<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();

  bucketSortCuda<<<1,n>>>(bucket, key, range, offset);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
