#pragma once

#include <cuda_runtime.h>

struct Color
{
    double r, g, b;

    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(double r, double g, double b) : r(r), g(g), b(b) {}

    __host__ __device__ Color operator*(const double scalar) const
    {
        return Color(r * scalar, g * scalar, b * scalar);
    }
};
