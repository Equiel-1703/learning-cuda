#include "../include/Tools.hpp"

void Tools::check_error(cudaError_t err_code, const char *file, int line)
{
    if (err_code != cudaSuccess)
    {
        std::cerr << "ERROR " << cudaGetErrorString(err_code) << "\n\tAt file " << file << " line " << line << std::endl;
        std::cerr << "CUDA error code: " << err_code << std::endl;
        std::cerr << "Exiting..." << std::endl;
        exit(-1);
    }
}