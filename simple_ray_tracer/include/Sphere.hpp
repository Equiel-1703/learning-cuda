#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "Vector.hpp"
#include "Color.hpp"

class Sphere
{
private:
    Vector center;
    double radius;
    Color color;

public:
    __host__ __device__ Sphere(Vector center, double radius, Color color)
        : center(center), radius(radius), color(color) {}

    __host__ __device__ Sphere(Vector center, double radius)
        : center(center), radius(radius), color(Color(1.0, 1.0, 1.0)) {} // Default color is white

    __host__ __device__ ~Sphere()
    {
        // Destructor logic if needed
    }

    __device__ Vector getCenter() const
    {
        return center;
    }

    __device__ double getRadius() const
    {
        return radius;
    }

    __device__ Color getColor() const
    {
        return color;
    }

    __device__ double hitDistance(const Vector &rayDirection) const
    {
        // Here we are assuming that the ray starts at the origin (0, 0, 0)
        Vector L = center.scale(1.0);

        // Calculate the coefficients of the quadratic equation
        // We omit a since it is always 1 for normalized rayDirection
        double b = 2.0 * rayDirection.dot(L);
        double c = L.dot(L) - radius * radius;

        double discriminant = b * b - 4.0 * c;

        if (discriminant < 0)
        {
            return -1.0; // No intersection
        }

        // Calculate the intersection points
        double discriminantSqrt = sqrt(discriminant);

        double t1 = (-b - discriminantSqrt) / 2.0;
        double t2 = (-b + discriminantSqrt) / 2.0;

        if (t1 > 0 && t2 > 0)
        {
            return (t1 < t2) ? t1 : t2; // Return the closest positive intersection
        }
        else if (t1 > 0)
        {
            return t1; // Return t1 if t2 is negative
        }
        else if (t2 > 0)
        {
            return t2; // Return t2 if t1 is negative
        }

        return -1.0; // No positive intersection (both are negative)
    }
};
