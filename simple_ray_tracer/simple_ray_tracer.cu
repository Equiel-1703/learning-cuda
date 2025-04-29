#include <iostream>

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

__global__ void renderKernel(Color *bitmap, Sphere *s)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int linearIdx = x + y * WIDTH;
    const double aspectRatio = (double)WIDTH / (double)HEIGHT;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    double ndcX = ((x + 0.5) / WIDTH) * 2.0 - 1.0;
    double ndcY = ((y + 0.5) / HEIGHT) * 2.0 - 1.0;

    Vector ray(ndcX * aspectRatio, ndcY, NEAR_PLANE);

    double dist = s->hitDistance(ray.normalize());

    if (dist < FAR_PLANE && dist > NEAR_PLANE)
    {
        bitmap[linearIdx] = s->getColor() * (1.0 - (dist / FAR_PLANE) * 20.0);
    }
    else
    {
        bitmap[linearIdx] = Color(0, 0, 0);
    }
}

int main(int argc, char const *argv[])
{
    Sphere sphere(Vector(0, 0, -20), 19.8, Color(128, 126, 0)), *s;

    CHECK_ERROR(cudaMalloc(&s, sizeof(Sphere)));
    CHECK_ERROR(cudaMemcpy(s, &sphere, sizeof(Sphere), cudaMemcpyKind::cudaMemcpyHostToDevice));

    const int numPixels = WIDTH * HEIGHT;

    Color *bitmap, *bitmapDevice;
    bitmap = new Color[numPixels];
    CHECK_ERROR(cudaMalloc(&bitmapDevice, sizeof(Color) * numPixels));

    const int xBlocks = (WIDTH + THREADS - 1) / THREADS;
    const int yBlocks = (HEIGHT + THREADS - 1) / THREADS;
    dim3 grid(xBlocks, yBlocks, 1);
    dim3 threads(THREADS, THREADS, 1);

    renderKernel<<<grid, threads>>>(bitmapDevice, s);

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
    CHECK_ERROR(cudaFree(s));

    delete[] bitmap;

    return 0;
}