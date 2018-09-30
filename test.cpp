#include <iostream>
#include <stdint.h>
#include <math.h>                   // pow()
#include <cuda_runtime.h>           // cudaFreeHost()
#include "CUDASieve/cudasieve.hpp"  // CudaSieve::getHostPrimes()
#include <chrono>

int main()
{
    size_t len;

    uint64_t b0 = 1000000000000000; //10^15
    uint64_t range = 1000000000; //10^9

    CudaSieve::getDevicePrimes(0, 200, len);
    
    uint64_t * devprimes;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    devprimes = CudaSieve::getDevicePrimes(b0, b0+range, len);
    cudaFree(devprimes);
    auto t1 = std::chrono::high_resolution_clock::now();

    devprimes = CudaSieve::getDevicePrimes(b0+range, b0+range*2, len);
    cudaFree(devprimes);
    auto t2 = std::chrono::high_resolution_clock::now();

    devprimes = CudaSieve::getDevicePrimes(b0, b0+range*2, len);
    cudaFree(devprimes);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    long dta = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    long dtb = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    long dtc = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();    
    printf("%ld %ld\n", dta+dtb, dtc);

    return 0;
    
    uint64_t * hostprimes = (uint64_t *)malloc(sizeof(uint64_t)*len);
    
    cudaMemcpy(hostprimes, devprimes, sizeof(uint64_t)*len, cudaMemcpyDeviceToHost);
    
    for(uint32_t i = 0; i < len; i++)
        std::cout << hostprimes[i] << std::endl;

    //cudaFreeHost(primes);            // must be freed with this call b/c page-locked memory is used.
    return 0;
}
