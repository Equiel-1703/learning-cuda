#pragma once

#include "Tools.hpp"

#include <cuda_runtime.h>

class GPUTimer
{
private:
    cudaEvent_t start, stop;

public:
    GPUTimer();
    ~GPUTimer();

    void start_timer();
    float stop_timer();
};