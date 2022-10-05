#include "Settings.h"

namespace vio_slam
{

ProcessTime::ProcessTime(const char* what /*"whatever"*/) : wFunc(what), start(clock()){}

void ProcessTime::totalTime()
{
    total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    totalOverTimes += total;
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

void ProcessTime::averageTimeOverTimes(const int times)
{
    std::cout << "-------------------------\n";
    std::cout << wFunc <<" Average Processing Time  : " << (float)totalOverTimes/times  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}

ConfigFile::ConfigFile(const char* config /*"config.yaml"*/) : configPath(config)
{
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
    configNode = YAML::LoadFile(dir_path + "/../config/" + configPath);
}

// template<typename T> 
// T ConfigFile::getValue(const std::string& first, const std::string& second)
// {
//     if (!second.empty())
//         return configNode[first][second].as<T>();
//     else
//         return configNode[first].as<T>();
// }

} // namespace vio_slam