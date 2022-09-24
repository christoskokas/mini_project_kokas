#include "Settings.h"

namespace vio_slam
{

ProcessTime::ProcessTime(const char* what /*"whatever"*/) : wFunc(what), start(clock()){}

void ProcessTime::totalTime()
{
    total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    std::cout << "-------------------------\n";
    std::cout << wFunc <<" Processing Time  : " << total  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}
} // namespace vio_slam