#include <iostream>

#include "../common/include/CPUTimer.hpp"

#define VEC_SIZE 60'000
#define NUM_ELEMENTS_PREVIEW 10

int main(int argc, char const *argv[])
{
    // Creating vectors
    int a[VEC_SIZE], b[VEC_SIZE], c[VEC_SIZE];

    for (int i = 0; i < VEC_SIZE; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    CPUTimer timer;

    timer.start_timer();

    for (int i = 0; i < VEC_SIZE; i++)
    {
        c[i] = a[i] + b[i];
    }

    timer.stop_timer();

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
    std::cout << "Tempo de execução: "
              << timer.get_elapsed_time_ns() << " ns / "
              << timer.get_elapsed_time_ms() << " ms"
              << std::endl;

    return 0;
}
