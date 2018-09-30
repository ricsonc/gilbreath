#include <vector>

typedef struct Result {
    bool success;
    int max_reps;
    uint64_t minprime;
    int numblocks;
    int chunknum;
} result;

result test_chunk(uint64_t *primes, long stop, long size, std::vector<int> &maxr);
