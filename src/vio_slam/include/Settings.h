#pragma once

#ifndef SETTINGS_H
#define SETTINGS_H

#include <ros/ros.h>
#include <iostream>
#include <yaml-cpp/yaml.h>


namespace vio_slam
{
    
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

} // namespace vio_slam


#endif // SETTINGS_H