#include "Settings.h"

namespace vio_slam
{

ProcessTime::ProcessTime(const char* what /*"whatever"*/) : wFunc(what), start(clock()){}

void ProcessTime::totalTime()
{
    total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    std::cout << "-------------------------\n";
    std::cout << wFunc <<" Total Processing Time  : " << total  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}

void ProcessTime::averageTime(const int times)
{
    total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    std::cout << "-------------------------\n";
    std::cout << wFunc <<" Average Processing Time  : " << (float)total/times  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}
} // namespace vio_slam