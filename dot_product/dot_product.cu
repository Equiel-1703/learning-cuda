#include <iostream>

#include "../common/include/Tools.hpp"

#define VEC_SIZE 150
#define THREADS_NUM 128
#define MAX_BLOCKS 32
#define NUM_ELEMENTS_PREVIEW 10

__global__ void dot(int *vect_1, int *vect_2, int *vect_3)
{
    __shared__ int cache[THREADS_NUM];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_id = threadIdx.x;
    int temp = 0;

    while (tid < VEC_SIZE)
    {
        temp += vect_1[tid] * vect_2[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cache_id] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cache_id < i)
            cache[cache_id] += cache[cache_id + i];

        i /= 2;
    }

    if (cache_id == 0)
        vect_3[blockIdx.x] = cache[0];
}

int main(int argc, char const *argv[])
{
    const int num_blocks = std::min((VEC_SIZE + THREADS_NUM - 1) / THREADS_NUM, MAX_BLOCKS);

    int *vect_1, *vect_2, *vect_3;

    vect_1 = new int[VEC_SIZE];
    vect_2 = new int[VEC_SIZE];
    vect_3 = new int[num_blocks];

    // Filling arrays with predictable values
    for (int i = 0; i < VEC_SIZE; i++)
    {
        vect_1[i] = i;
        vect_2[i] = 3;
    }

    int *v1_dev, *v2_dev, *v3_dev;

    // Creating vectors in GPU
    CHECK_ERROR(cudaMalloc(&v1_dev, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc(&v2_dev, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc(&v3_dev, sizeof(int) * num_blocks));

    // Copying data to GPU
    CHECK_ERROR(cudaMemcpy(v1_dev, vect_1, sizeof(int) * VEC_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(v2_dev, vect_2, sizeof(int) * VEC_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice));

    // Calling kernel
    dot<<<num_blocks, THREADS_NUM>>>(v1_dev, v2_dev, v3_dev);
    CHECK_ERROR(cudaDeviceSynchronize());

    // Copiando o resultado para a CPU
    CHECK_ERROR(cudaMemcpy(vect_3, v3_dev, sizeof(int) * num_blocks, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    std::cout << "Vect 1: ";
    for (int i = 0; i < NUM_ELEMENTS_PREVIEW; i++)
    {
        std::cout << vect_1[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "Vect 2: ";
    for (int i = 0; i < NUM_ELEMENTS_PREVIEW; i++)
    {
        std::cout << vect_2[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "Result (after GPU processing): ";
    for (int i = 0; i < num_blocks; i++)
    {
        std::cout << vect_3[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Expected result: " << (3 * (VEC_SIZE - 1)) * VEC_SIZE / 2 << std::endl;

    std::cout << "\n";
    std::cout << "* Número de elementos: " << VEC_SIZE << "\n"
              << "* Número de blocos usados: " << num_blocks << "\n"
              << "* Número de threads por bloco: " << THREADS_NUM << std::endl;

    // Liberando memória
    CHECK_ERROR(cudaFree(v1_dev));
    CHECK_ERROR(cudaFree(v2_dev));
    CHECK_ERROR(cudaFree(v3_dev));

    delete[] vect_1;
    delete[] vect_2;
    delete[] vect_3;

    return 0;
}
