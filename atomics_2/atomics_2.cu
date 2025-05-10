#include <iostream>

#include "../common/include/Tools.hpp"
#include "../common/include/GPUTimer.hpp"
#include "../common/include/CPUTimer.hpp"

#define N 3'000'000

unsigned long long arithmSum(unsigned long long n, unsigned long long a1, unsigned long long an)
{
    return (n * (a1 + an)) / 2;
}

void reduceCPU()
{
    unsigned long long *elements, result;

    elements = new unsigned long long[N];

    for (int i = 0; i < N; i++)
    {
        elements[i] = i;
    }

    result = 0;

    CPUTimer timer;
    timer.start_timer();

    for (int i = 0; i < N; i++)
    {
        result += elements[i];
    }

    timer.stop_timer();

    delete[] elements;

    double elapsed = timer.get_elapsed_time_ms();

    unsigned long long expectedResult = arithmSum(N, 0, N - 1);

    std::cout << "\nResult: " << result << std::endl;
    std::cout << "Expected: " << expectedResult << std::endl;
    std::cout << "Difference (should be 0): " << result - expectedResult << std::endl;

    std::cout << "\nElapsed time: " << elapsed << " ms" << std::endl;
}

__global__ void reduceKernelV2(unsigned long long *input, unsigned long long *output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    const int tid = threadIdx.x;
    const int offset = blockDim.x * 2 * gridDim.x;

    if (idx >= N)
    {
        return;
    }

    __shared__ unsigned long long localAccumulator;

    if (tid == 0)
    {
        localAccumulator = 0;
    }

    __syncthreads();

    unsigned long long sum = 0;

    while (idx < N)
    {
        sum += input[idx];
        if (idx + blockDim.x < N)
        {
            sum += input[idx + blockDim.x];
        }

        idx += offset;
    }

    atomicAdd(&localAccumulator, sum);

    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(output, localAccumulator);
    }
}

void reducePinned(const cudaDeviceProp prop)
{
    unsigned long long *hostElements, *result, *devElements, *devResult;

    CHECK_ERROR(cudaHostAlloc((void **)&hostElements, N * sizeof(unsigned long long), cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CHECK_ERROR(cudaHostAlloc((void **)&result, sizeof(unsigned long long), cudaHostAllocMapped));

    for (int i = 0; i < N; i++)
    {
        hostElements[i] = i;
    }

    CHECK_ERROR(cudaHostGetDevicePointer((void **)&devElements, (void *)hostElements, 0));
    CHECK_ERROR(cudaHostGetDevicePointer((void **)&devResult, (void *)result, 0));

    const int blocks = prop.multiProcessorCount;
    const int threads = prop.maxThreadsPerBlock;

    GPUTimer timer;
    timer.start_timer();

    reduceKernelV2<<<blocks, threads>>>(devElements, devResult);

    CHECK_ERROR(cudaDeviceSynchronize());

    float elapsed = timer.stop_timer();

    unsigned long long expectedResult = arithmSum(N, 0, N - 1);

    std::cout << "Number of blocks: " << blocks << std::endl;
    std::cout << "Number of threads: " << threads << std::endl;
    std::cout << "Number of elements: " << N << std::endl;

    std::cout << "\nResult: " << (*result) << std::endl;
    std::cout << "Expected: " << expectedResult << std::endl;
    std::cout << "Difference (should be 0): " << (*result) - expectedResult << std::endl;

    std::cout << "\nElapsed time: " << elapsed << " ms" << std::endl;

    CHECK_ERROR(cudaFreeHost(hostElements));
    CHECK_ERROR(cudaFreeHost(result));
}

cudaDeviceProp getAndPrintGPUInfo()
{
    cudaDeviceProp prop;

    CHECK_ERROR(cudaGetDeviceProperties(&prop, 0)); // Getting GPU info from device 0 (default GPU)

    std::cout << "- DEVICE PROPERTIES (device 0) -\n"
              << std::endl;

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multi-processor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per dimension: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "Integrated: " << (prop.integrated ? "Yes" : "No") << std::endl;
    std::cout << "Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
    std::cout << std::endl;

    return prop;
}

int main()
{
    CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::cout << "================== Reduce CPU ==================" << std::endl;

    reduceCPU();

    std::cout << "\n================== Reduce GPU (with pinned memory and V2 kernel)==================\n"
              << std::endl;

    cudaDeviceProp prop = getAndPrintGPUInfo();

    reducePinned(prop);

    return 0;
}
