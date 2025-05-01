#pragma once

#include <iostream>
#include <cuda_runtime.h>

class Vector
{
private:
    double x, y, z;

public:
    __host__ __device__ Vector(double x, double y, double z) : x(x), y(y), z(z)
    {
    }

    __host__ __device__ Vector() : Vector(0, 0, 0)
    {
    }

    __host__ __device__ ~Vector()
    {
    }

    __device__ double getX() const
    {
        return x;
    }

    __device__ double getY() const
    {
        return y;
    }

    __device__ double getZ() const
    {
        return z;
    }

    void setX(double x)
    {
        this->x = x;
    }

    void setY(double y)
    {
        this->y = y;
    }

    void setZ(double z)
    {
        this->z = z;
    }

    __device__ double length() const
    {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ double distanceTo(const Vector &other) const
    {
        Vector diff = *this - other;
        return diff.length();
    }

    __device__ double dot(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __device__ Vector scale(double scalar) const
    {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    __device__ Vector normalize() const
    {
        double len = length();
        if (len == 0)
            return Vector(0, 0, 0);
        return Vector(x / len, y / len, z / len);
    }

    __host__ __device__ Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Vector &operator=(const Vector &other)
    {
        if (this != &other)
        {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
        }

        return *this;
    }
};
