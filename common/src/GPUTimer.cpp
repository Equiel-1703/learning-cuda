#include "../include/GPUTimer.hpp"

GPUTimer::GPUTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GPUTimer::~GPUTimer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GPUTimer::start_timer()
{
    cudaEventRecord(start, 0);
}

float GPUTimer::stop_timer()
{
    float time_ms;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    return time_ms;
}
