#include <chrono>

namespace ch = std::chrono;

class CPUTimer
{
private:
    ch::high_resolution_clock::time_point start_time, end_time;
    ch::nanoseconds elapsed_time_ns;

public:
    CPUTimer();
    void start_timer();
    void stop_timer();

    long get_elapsed_time_ns();
    double get_elapsed_time_ms();
};