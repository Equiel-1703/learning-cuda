#include <iostream>

#include "../common/include/Tools.hpp"
#include "../common/include/GPUTimer.hpp"

#define VEC_SIZE 60'000
#define NUM_ELEMENTS_PREVIEW 10

__global__ void vec_sum(int *vec_1, int *vec_2, int *vec_3)
{
    int vec_index = blockIdx.x;

    if (vec_index < VEC_SIZE)
    {
        vec_3[vec_index] = vec_1[vec_index] + vec_2[vec_index];
    }
}

int main()
{
    // Criando vetores na CPU e medidor de tempo da GPU
    int a[VEC_SIZE], b[VEC_SIZE], c[VEC_SIZE];
    GPUTimer timer;

    // Preenchendo vetores
    for (int i = 0; i < VEC_SIZE; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Alocando vetores na GPU
    int *vec_1, *vec_2, *vec_3;

    CHECK_ERROR(cudaMalloc((void **)&vec_1, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc((void **)&vec_2, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc((void **)&vec_3, sizeof(int) * VEC_SIZE));

    // Copiando vetores a e b para GPU
    CHECK_ERROR(cudaMemcpy(vec_1, a, sizeof(a), cudaMemcpyKind::cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(vec_2, b, sizeof(b), cudaMemcpyKind::cudaMemcpyHostToDevice));

    // Chamando kernel para realizar a computação. Estou lançando um bloco para cada elemento do vetor.
    timer.start_timer();
    vec_sum<<<VEC_SIZE, 1>>>(vec_1, vec_2, vec_3);

    CHECK_ERROR(cudaDeviceSynchronize());

    float time_taken = timer.stop_timer();

    // Copiando o resultado pra CPU
    CHECK_ERROR(cudaMemcpy(c, vec_3, sizeof(c), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    std::cout << "A: ";
    for (int i = 0; i < NUM_ELEMENTS_PREVIEW; i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "B: ";
    for (int i = 0; i < NUM_ELEMENTS_PREVIEW; i++)
    {
        std::cout << b[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "C: ";
    for (int i = 0; i < NUM_ELEMENTS_PREVIEW; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "Número de elementos: " << VEC_SIZE << std::endl;
    std::cout << "Tempo de execução: " << time_taken << " ms" << std::endl;

    // Liberando memória
    CHECK_ERROR(cudaFree(vec_1));
    CHECK_ERROR(cudaFree(vec_2));
    CHECK_ERROR(cudaFree(vec_3));

    return 0;
}
