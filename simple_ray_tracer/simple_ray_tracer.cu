#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "../common/include/Tools.hpp"
#include "../common/include/GPUTimer.hpp"
#include "../common/include/BMP.hpp"

#include "include/Vector.hpp"
#include "include/Sphere.hpp"
#include "include/Color.hpp"

#define WIDTH 800
#define HEIGHT 600

#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

#define THREADS 16

#define NUM_SPHERES 10

double randDouble(double min, double max)
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}

int randInt(int min, int max)
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max - 1);
    return distribution(generator);
}

__global__ void renderKernel(Color *bitmap, Sphere *spheres)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int linearIdx = x + y * WIDTH;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    const double aspectRatio = (double)WIDTH / (double)HEIGHT;
    const double fov = 60.0;
    const double fovTan = tan(fov * 0.5 * M_PI / 180.0);

    double ndcX = ((x + 0.5) / WIDTH) * 2.0 - 1.0;
    double ndcY = ((y + 0.5) / HEIGHT) * 2.0 - 1.0;

    Vector rayDir(ndcX * aspectRatio * fovTan, ndcY * fovTan, -1.0);
    rayDir = rayDir.normalize();

    double dist = -1.0;
    int idx = -1;
    for (int i = 0; i < NUM_SPHERES; i++)
    {
        double new_dist = spheres[i].hitDistance(rayDir);
        if (new_dist > 0.0 && (dist < 0.0 || new_dist < dist))
        {
            dist = new_dist;
            idx = i;
        }
    }

    if (dist < FAR_PLANE && dist > NEAR_PLANE)
    {
        bitmap[linearIdx] = spheres[idx].getColor() * std::clamp(1.0 - (dist / FAR_PLANE) * 20.0, 0.0, 1.0);
    }
    else
    {
        // Normalize Y between 0.0 and 1.0
        double t = 0.5 * (rayDir.getY() + 1.0);

        Color white(143, 183, 247);
        Color blue(41, 104, 240); // Light sky blue

        // Linear interpolation between white and blue
        Color sky = white * (1.0 - t) + blue * t;

        bitmap[linearIdx] = sky;
    }
}

int main(int argc, char const *argv[])
{
    Sphere *spheres, *spheresDevice;

    spheres = new Sphere[NUM_SPHERES];
    for (int i = 0; i < NUM_SPHERES; i++)
    {
        Vector center(randDouble(-5.0, 5.0), randDouble(-5.0, 5.0), -randDouble(1.0, 5.0));
        double radius = randDouble(0.5, 1.0);
        Color color(randInt(100, 255), randInt(0, 255), randInt(0, 255));

        spheres[i] = Sphere(center, radius, color);
    }

    CHECK_ERROR(cudaMalloc(&spheresDevice, sizeof(Sphere) * NUM_SPHERES));
    CHECK_ERROR(cudaMemcpy(spheresDevice, spheres, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyKind::cudaMemcpyHostToDevice));

    const int numPixels = WIDTH * HEIGHT;

    Color *bitmap, *bitmapDevice;
    bitmap = new Color[numPixels];
    CHECK_ERROR(cudaMalloc(&bitmapDevice, sizeof(Color) * numPixels));

    const int xBlocks = (WIDTH + THREADS - 1) / THREADS;
    const int yBlocks = (HEIGHT + THREADS - 1) / THREADS;
    dim3 grid(xBlocks, yBlocks, 1);
    dim3 threads(THREADS, THREADS, 1);

    renderKernel<<<grid, threads>>>(bitmapDevice, spheresDevice);

    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(bitmap, bitmapDevice, sizeof(Color) * numPixels, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    // Create BMP
    BMP image("ray_tracer.bmp", WIDTH, HEIGHT);
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            Color c = bitmap[x + y * WIDTH];
            image.writePixel(x, y, c.r, c.g, c.b);
        }
    }
    image.save();

    CHECK_ERROR(cudaFree(bitmapDevice));
    CHECK_ERROR(cudaFree(spheresDevice));

    delete[] bitmap;

    return 0;
}