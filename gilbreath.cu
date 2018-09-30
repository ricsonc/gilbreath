#include <cuda.h>
#include <cuda_runtime.h>           // cudaFreeHost()
#include "gilbreath.h"
#include <stdio.h>
#include <cassert>
#include <vector>
#include <algorithm>

#define SUBSIZE 2048
#define MARGIN 1024
#define CHUNKSIZE SUBSIZE+MARGIN
#define DEBUG 1
#define STRIDE 32

__device__ __inline__ void adjdiff(int *src, int *dest, int fromend){
    for(int i = threadIdx.x; i < CHUNKSIZE-1-fromend; i+= blockDim.x){
        //int diff = src[i]-src[i+1];
        //dest[i] = std::abs(diff);
        dest[i] = __sad(src[i], src[i+1], 0);
    }
}

__device__ __inline__ void check02(int *arr, bool *is02, int fromend){
    for(int i = threadIdx.x; i < CHUNKSIZE-fromend; i+= blockDim.x){
        is02[i] = arr[i] <= 2.5; //no rounding
    }
}

__device__ __inline__ void reduce(bool *arr, int size){
    for(int i = size+threadIdx.x; i < CHUNKSIZE; i+= blockDim.x){
        arr[i] = true;
    }
    __syncthreads();
    int limit = CHUNKSIZE;
    while(limit > 1){
        limit >>= 1;
        for(int i = threadIdx.x; i < limit; i+= blockDim.x){
            arr[i] = arr[2*i] && arr[2*i+1];
        }
    }
}

__global__ void test_sub_chunk(uint64_t *primes, size_t len, int *maxr){
    __shared__ int temp1[CHUNKSIZE];
    __shared__ int temp2[CHUNKSIZE];
    __shared__ bool is02[CHUNKSIZE];

    int start = SUBSIZE*blockIdx.x;
    for(int i = threadIdx.x; i < CHUNKSIZE; i+= blockDim.x){ 
        int64_t p0 = (int64_t) primes[i+start];
        int64_t p1 = (int64_t) primes[i+start+1];
        temp1[i] = __sad(p0, p1, 0);
    }
    __syncthreads();

    int r = 0;
    for(; r < MARGIN; r +=2){

        adjdiff(temp1, temp2, r);
        __syncthreads();

        adjdiff(temp2, temp1, r+1);
        __syncthreads();

        if(r % STRIDE == 0){
            check02(temp1, is02, r+2);
            __syncthreads();

            reduce(is02, CHUNKSIZE-r-2);
            __syncthreads();
            
            if(is02[0]){
                break;
            }
        }
    }

    __syncthreads();
    if(threadIdx.x == 0){
        maxr[blockIdx.x] = r+3;
    }
}

result test_chunk(uint64_t *primes, long stop, long size, std::vector<int> &maxr){
    uint64_t lastprime;
    cudaMemcpy(&lastprime, primes+size-1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(lastprime >= stop);

    int num_blocks = size/SUBSIZE-2; //this is important
    int num_threads = 256;
    int *devmaxr;
    cudaMalloc(&devmaxr, sizeof(int)*num_blocks);
    test_sub_chunk<<<num_blocks,num_threads>>>(primes, size, devmaxr);

    maxr.resize(num_blocks);
    cudaMemcpy(&maxr[0], devmaxr, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
    cudaFree(devmaxr);

    int max_ = *std::max_element(maxr.begin(), maxr.end());

    uint64_t minprime;
    cudaMemcpy(&minprime, primes, sizeof(uint64_t), cudaMemcpyDeviceToHost);    
    return {max_ < MARGIN, max_, minprime, num_blocks};
}
