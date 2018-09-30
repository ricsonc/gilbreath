#include <primesieve.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>           // cudaFreeHost()
#include "CUDASieve/cudasieve.hpp"  // CudaSieve::getHostPrimes()

#include "gilbreath.h"

#define SEGMENTSIZE 10000000000
#define MARGIN 10000000
#define DEBUG 0

//from gilbreath.cu:
#define SUBSIZE 2048
#define CHUNKSIZE 3072
#define STRIDE 32

/*
bool verify_block(uint64_t *primes, int start, int lb, int ub){
    int reps = 0;
    std::vector<uint64_t> hostprimes;
    hostprimes.resize(CHUNKSIZE);
    cudaMemcpy(&hostprimes[0], primes+start, sizeof(uint64_t)*CHUNKSIZE, cudaMemcpyDeviceToHost);
    //cast to signed
    std::vector<int64_t> diff(hostprimes.begin(), hostprimes.end());
    while(1){
        reps++;
        std::adjacent_difference(diff.begin(), diff.end(), diff.begin());
        std::transform(diff.begin(), diff.end(), diff.begin(),
                       [](int64_t n){return std::abs(n);});
        
        int64_t maxdiff = *std::max_element(diff.begin()+reps, diff.end());
        if(maxdiff <= 2){
            break;
        }
    }
    return (lb <= reps && reps <= ub);
}

void report_chunk(uint64_t *primes, std::vector<int> &maxr){
    for(auto it = maxr.begin(); it != maxr.end(); it++){
        int blockidx = it-maxr.begin();
        int startidx = blockidx*SUBSIZE;
        int endidx = startidx+CHUNKSIZE-1;
        uint64_t startprime;
        uint64_t endprime;
        cudaMemcpy(&startprime, primes+startidx, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&endprime, primes+endidx, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        int num_reps = *it;
        //printf("%llu %llu %d %d\n",
        //       startprime, endprime, num_reps-STRIDE+1, num_reps);
        if(DEBUG){
            verify_block(primes, startidx, num_reps-STRIDE+1, num_reps);
            printf("block %d was ok\n", blockidx);
        }
    }
}
*/

void report_results(result R){
    std::ofstream log;
    log.open("log", std::ios_base::app);
    log << R.chunknum << " ";
    log << R.max_reps << " ";
    log << R.minprime << " ";
    log << R.numblocks << std::endl;
}

bool testrange(long start){

    uint64_t * primes;
    std::vector<int> max_repeats;
    
    long bottom = start*SEGMENTSIZE;
    long chunknum = start;
    while(true){
        auto t0 = std::chrono::high_resolution_clock::now();

        long top = bottom + SEGMENTSIZE;
        size_t count;
        primes = CudaSieve::getDevicePrimes(bottom, top+MARGIN, count);

        auto t1 = std::chrono::high_resolution_clock::now();

        //double check, incase of hardware error
        result R = test_chunk(primes, top, count, max_repeats);
        R.chunknum = chunknum;
        if(!R.success){
            return false;
        }
        report_results(R);
        cudaFree(primes);

        //due to memory leak in sieve?
        if(!(chunknum % 100)){
            printf("resetting device\n");
            cudaDeviceReset();
        }
        bottom = top;
        chunknum++;
        
        auto t2 = std::chrono::high_resolution_clock::now();
        long d0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        long d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

        printf("chunk number %ld in time %ld %ld\n", chunknum, d0, d1);
        fflush(stdout);
    }
    return true;
}

int main(int argc, char **argv)
{
    long start;
    if(argc > 1){
        start = (long)atoi(argv[1]);
    } else {
        start = 1;
    }
    printf("starting at %ld\n", start);

    testrange(start);

    printf("test failed, possible counterexample\n");
}

/*
 * the expected margin we need is 5.134 d^2 - 22.40 d + 37.6 
 * for 22 this is 2030
 * for 21 this is 1831
 * for 20 this is 1643
 * for 19 this is 1465
 * for 18 this is 1298
 * for 17 this is 1141
 * for 16 this is 994
 * for 15 this is 817
 * for 14 this is 730
 * key : hand long segments off to cpu
 */
