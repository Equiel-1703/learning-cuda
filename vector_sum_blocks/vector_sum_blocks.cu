#include <iostream>

#include "../common/tools.hpp"

#define VEC_SIZE 10'000

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
    // Criando vetores na CPU
    int a[VEC_SIZE], b[VEC_SIZE], c[VEC_SIZE];

    // Preenchendo vetores
    for (int i = 0; i < VEC_SIZE; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // Alocando vetores na GPU
    int *vec_1, *vec_2, *vec_3;

    CHECK_ERROR(cudaMalloc((void **)&vec_1, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc((void **)&vec_2, sizeof(int) * VEC_SIZE));
    CHECK_ERROR(cudaMalloc((void **)&vec_3, sizeof(int) * VEC_SIZE));

    // Copiando vetores a e b para GPU
    CHECK_ERROR(cudaMemcpy(vec_1, a, sizeof(a), cudaMemcpyKind::cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(vec_2, a, sizeof(b), cudaMemcpyKind::cudaMemcpyHostToDevice));

    // Chamando kernel para realizar a computação. Estou lançando um bloco para cada elemento do vetor.
    vec_sum<<<VEC_SIZE, 1>>>(vec_1, vec_2, vec_3);

    CHECK_ERROR(cudaDeviceSynchronize());

    // Copiando o resultado pra CPU
    CHECK_ERROR(cudaMemcpy(c, vec_3, sizeof(c), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    std::cout << "SOMA: ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Liberando memória
    CHECK_ERROR(cudaFree(vec_1));
    CHECK_ERROR(cudaFree(vec_2));
    CHECK_ERROR(cudaFree(vec_3));

    return 0;
}
