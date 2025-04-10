#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_ERROR(err_call) Tools::check_error(err_call, __FILE__, __LINE__);

class Tools
{
public:
    static void check_error(cudaError_t err_code, const char *file, int line);
};
