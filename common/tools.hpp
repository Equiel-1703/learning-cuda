#pragma once

#include <iostream>
#include <cuda_runtime.h>

void check_error(cudaError_t err_code, const char *file, int line)
{
    if (err_code != cudaSuccess)
    {
        std::cerr << "ERROR " << cudaGetErrorString(err_code) << "\n\tAt file " << file << " line " << line << std::endl;
        std::cerr << "CUDA error code: " << err_code << std::endl;
        std::cerr << "Exiting..." << std::endl;
        exit(-1);
    }
}

#define CHECK_ERROR(err_call) check_error(err_call, __FILE__, __LINE__);
