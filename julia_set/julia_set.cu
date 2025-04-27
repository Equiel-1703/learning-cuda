#include <iostream>
#include <stdint.h>
#include <complex>

#include "../common/include/Tools.hpp"
#include "../common/include/BMP.hpp"

#define MAX_ITERATIONS 100
#define THRESHOLD 2.0

#define THREADS_PER_AXIS 32
#define MAX_BLOCKS 6

using std::complex;
using std::cout, std::cerr, std::endl;

__device__ unsigned int run_iterations(double real, double img)
{
    complex<double> z(real, img);
    const complex<double> c(-0.8, 0.156);

    unsigned int it = 1;

    while (it <= MAX_ITERATIONS)
    {
        z = z * z + c;

        if (std::abs(z) > THRESHOLD)
            break;

        it += 1;
    }

    return it;
}

__global__ void compute_julia_set(uint8_t *img, int *img_size)
{
    const int IMG_SIZE = *img_size;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= IMG_SIZE || y >= IMG_SIZE)
        return;

    // printf("tid_x: %d, tid_y:%d, bid_x: %d, bid_y: %d, x: %d, y: %d, linear_idx: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y, linear_idx);

    const double x_min = -1.0, x_max = 1.0;
    const double y_min = -1.0, y_max = 1.0;

    const int inc_y = blockDim.y * gridDim.y;
    const int inc_x = blockDim.x * gridDim.x;

    for (int it_y = y; it_y < IMG_SIZE; it_y += inc_y)
    {
        for (int it_x = x; it_x < IMG_SIZE; it_x += inc_x)
        {
            double x_frac = x_min + (x_max - x_min) * (double(it_x) / double(IMG_SIZE - 1));
            double y_frac = y_min + (y_max - y_min) * (double(it_y) / double(IMG_SIZE - 1));

            unsigned int its = run_iterations(x_frac, y_frac);
            double color_intensity = double(its) / MAX_ITERATIONS;

            int linear_idx = it_x + it_y * IMG_SIZE;

            img[linear_idx * 3] = uint8_t(200 * color_intensity);
            img[linear_idx * 3 + 1] = uint8_t(100 * color_intensity);
            img[linear_idx * 3 + 2] = uint8_t(10 * color_intensity);
        }
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        cerr << "USAGE: " << argv[0] << " <image_size_px>" << endl;
        return EXIT_FAILURE;
    }

    const int IMG_SIZE = atoi(argv[1]);
    const int IMG_PIXELS = IMG_SIZE * IMG_SIZE;

    if (IMG_SIZE < 10)
    {
        cerr << "ERROR: Image size must be at least 10x10 pixels." << endl;
        return EXIT_FAILURE;
    }

    int *IMG_SIZE_DEV;
    uint8_t *img, *img_device;

    img = new uint8_t[IMG_PIXELS * 3];

    CHECK_ERROR(cudaMalloc(&img_device, sizeof(uint8_t) * IMG_PIXELS * 3));
    CHECK_ERROR(cudaMalloc(&IMG_SIZE_DEV, sizeof(int)));

    CHECK_ERROR(cudaMemcpy(IMG_SIZE_DEV, &IMG_SIZE, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));

    const int BLOCKS = std::min((IMG_SIZE + THREADS_PER_AXIS - 1) / THREADS_PER_AXIS, MAX_BLOCKS);
    const dim3 dim_blocks(BLOCKS, BLOCKS, 1);
    const dim3 dim_threads(THREADS_PER_AXIS, THREADS_PER_AXIS, 1);

    compute_julia_set<<<dim_blocks, dim_threads>>>(img_device, IMG_SIZE_DEV);
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(img, img_device, sizeof(uint8_t) * IMG_PIXELS * 3, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    // Creating bitmap
    BMP bmp_img("julia.bmp", IMG_SIZE, IMG_SIZE);

    for (size_t i = 0; i < IMG_SIZE; i++)
    {
        for (size_t j = 0; j < IMG_SIZE; j++)
        {
            int index = (i + j * IMG_SIZE) * 3;
            bmp_img.writePixel(i, j,
                               img[index],
                               img[index + 1],
                               img[index + 2]);
        }
    }

    bmp_img.save();

    cout << "BLOCKS: " << BLOCKS << "x" << BLOCKS << "(" << BLOCKS * BLOCKS << ")" << endl;
    cout << "THREADS: " << THREADS_PER_AXIS << "x" << THREADS_PER_AXIS << "(" << THREADS_PER_AXIS * THREADS_PER_AXIS << ")" << endl;
    cout << "IMG SIZE: " << IMG_SIZE << "x" << IMG_SIZE << endl;

    // Cleaning memory
    CHECK_ERROR(cudaFree(img_device));
    CHECK_ERROR(cudaFree(IMG_SIZE_DEV));

    delete[] img;

    return EXIT_SUCCESS;
}
