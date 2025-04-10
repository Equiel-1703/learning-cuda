#include "../include/CPUTimer.hpp"

CPUTimer::CPUTimer()
{
    elapsed_time_ns = ch::nanoseconds(0);
}

void CPUTimer::start_timer()
{
    start_time = ch::high_resolution_clock::now();
}

void CPUTimer::stop_timer()
{
    end_time = ch::high_resolution_clock::now();
    elapsed_time_ns = ch::duration_cast<ch::nanoseconds>(end_time - start_time);
}

long CPUTimer::get_elapsed_time_ns()
{
    return elapsed_time_ns.count();
}

double CPUTimer::get_elapsed_time_ms()
{
    return static_cast<double>(elapsed_time_ns.count()) / 1e6;
}
