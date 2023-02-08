#pragma once

#ifndef SETTINGS_H
#define SETTINGS_H

#include <ros/ros.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <chrono>

#define KITTI_DATASET true
#define KITTI_SEQ "00"
#define ZED_DATASET false
#define ZED_DEMO true
#define V1_02 false
#define SIMULATION true
#define DRAWMATCHES false

#define SAVEODOMETRYDATA false

namespace vio_slam
{

/**
 * @brief Logging Function
 * @param com Comment on the value to be printed
 * @param toPrint Value to be printed
 * @param level Level of Importance.
 * DEBUG = 0,
 * INFO = 1,
 * WARNING = 2,
 * ERROR = 3
 */
class Logging
{
    enum Level
    {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        NOCOUT
    };

    Level curLevel {WARNING};

    public:
        template <typename T>
        Logging(const char* com, const T& toPrint, const int level, const char* measurements = "")
        {
            if (curLevel != Level::NOCOUT)
                if ((curLevel <= level) && (level < 4))
                {
                    switch (level)
                    {
                        case Level::DEBUG :
                            std::cout << "[DEBUG] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::INFO :
                            std::cout << "[INFO] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::WARNING :
                            std::cout << "[WARNING] : " << com << " " << toPrint << measurements  << '\n';
                            break;
                        case Level::ERROR :
                            std::cout << "[ERROR] : " << com << " " << toPrint << measurements << '\n';
                            break;
                        default : 
                            break;
                    }
                }
        }

};

class Timer
{
    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::duration<float> duration;
    const char* message;
    public:
        Timer(const char* _message = "Timer took");
        ~Timer();
};

class ProcessTime
{
    private:
        const char* wFunc;
    public:
        clock_t total;
        clock_t totalOverTimes {0};
        clock_t start;
        ProcessTime(const char* what = "whatever");
        void totalTime();
        void averageTime(const int times);
        void averageTimeOverTimes(const int times);

};

class ConfigFile
{
    private:
        const char* configPath;

    public:
        YAML::Node configNode;
        ConfigFile(const char* config = "config.yaml");
        template<typename T> 
        T getValue(const std::string& first = "", const std::string& second = "", const std::string& third = "")
        {
            if (!third.empty())
                return configNode[first][second][third].as<T>();
            else if (!second.empty())
                return configNode[first][second].as<T>();
            else
                return configNode[first].as<T>();
        }
};

template <typename T, typename U>
void reduceVectorInliersTemp(std::vector<T>& vec,std::vector<U>& check)
{
    int j {0};
    for (int i = 0; i < int(check.size()); i++)
        vec[j++] = vec[check[i]];
    vec.resize(j);
}

template <typename T, typename U>
void reduceVectorTemp(std::vector<T>& vec,std::vector<U>& check)
{
    int j {0};
    for (int i = 0; i < int(vec.size()); i++)
        if (check[i])
            vec[j++] = vec[i];
    vec.resize(j);
}

template <typename T, typename U>
void reduceVectorWithValue(std::vector<T>& vec,std::vector<U>& check, const float value)
{
    int j {0};
    for (int i = 0; i < int(vec.size()); i++)
        if (check[i] < value)
            vec[j++] = vec[i];
    vec.resize(j);
}

template <typename T>
void checkVectorTemp(std::vector<bool>& vec,std::vector<T>& check)
{
    for (int i = 0; i < int(vec.size()); i++)
        if (!(check[i] && vec[i]))
            vec[i] = false;

}

} // namespace vio_slam


#endif // SETTINGS_H