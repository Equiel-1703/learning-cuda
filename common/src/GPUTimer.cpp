#include "../include/GPUTimer.hpp"

GPUTimer::GPUTimer()
{
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
}

GPUTimer::~GPUTimer()
{
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));
}

void GPUTimer::start_timer()
{
    CHECK_ERROR(cudaEventRecord(start, 0));
}

float GPUTimer::stop_timer()
{
    float time_ms;

    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));
    CHECK_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    return time_ms;
}
