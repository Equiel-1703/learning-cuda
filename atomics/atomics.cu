#include <iostream>

#include "../common/include/Tools.hpp"
#include "../common/include/GPUTimer.hpp"
#include "../common/include/CPUTimer.hpp"

#define N 3'000'000

unsigned long long arithmSum(unsigned long long n, unsigned long long a1, unsigned long long an)
{
    return (n * (a1 + an)) / 2;
}

__global__ void reduceKernel(unsigned long long *input, unsigned long long *output)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    const int tid = threadIdx.x;

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

    unsigned long long sum = input[idx];
    if (idx + blockDim.x < N)
    {
        sum += input[idx + blockDim.x];
    }

    atomicAdd(&localAccumulator, sum);

    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(output, localAccumulator);
    }
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

void reducePinned(const int threads)
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

    const int blocks = (N + (threads * 2) - 1) / (threads * 2);

    GPUTimer timer;
    timer.start_timer();

    reduceKernel<<<blocks, threads>>>(devElements, devResult);

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

void reduce(int const threads)
{
    unsigned long long *hostElements, *deviceElements, *deviceResult, hostResult;

    hostElements = new unsigned long long[N];

    CHECK_ERROR(cudaMalloc((void **)&deviceElements, N * sizeof(unsigned long long)));
    CHECK_ERROR(cudaMalloc((void **)&deviceResult, sizeof(unsigned long long)));
    CHECK_ERROR(cudaMemset(deviceResult, 0, sizeof(unsigned long long)));

    for (int i = 0; i < N; i++)
    {
        hostElements[i] = i;
    }

    GPUTimer timer;
    timer.start_timer();

    CHECK_ERROR(
        cudaMemcpy(
            (void *)deviceElements,
            (void *)hostElements,
            N * sizeof(unsigned long long),
            cudaMemcpyKind::cudaMemcpyHostToDevice));

    const int blocks = (N + (threads * 2) - 1) / (threads * 2);

    reduceKernel<<<blocks, threads>>>(deviceElements, deviceResult);

    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(
        cudaMemcpy(
            (void *)&hostResult,
            (void *)deviceResult,
            sizeof(unsigned long long),
            cudaMemcpyKind::cudaMemcpyDeviceToHost));

    float elapsed = timer.stop_timer();

    CHECK_ERROR(cudaFree(deviceElements));
    CHECK_ERROR(cudaFree(deviceResult));

    delete[] hostElements;

    unsigned long long expectedResult = arithmSum(N, 0, N - 1);

    std::cout << "Number of blocks: " << blocks << std::endl;
    std::cout << "Number of threads: " << threads << std::endl;
    std::cout << "Number of elements: " << N << std::endl;

    std::cout << "\nResult: " << hostResult << std::endl;
    std::cout << "Expected: " << expectedResult << std::endl;
    std::cout << "Difference (should be 0): " << hostResult - expectedResult << std::endl;

    std::cout << "\nElapsed time: " << elapsed << " ms" << std::endl;
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

    std::cout << "\n================== Reduce GPU ==================\n"
              << std::endl;

    cudaDeviceProp prop = getAndPrintGPUInfo();
    reduce(prop.maxThreadsPerBlock);

    std::cout << "\n================== Reduce GPU (pinned memory) ==================\n"
              << std::endl;

    reducePinned(prop.maxThreadsPerBlock);

    return 0;
}
