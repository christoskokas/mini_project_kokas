#pragma once

#ifndef SETTINGS_H
#define SETTINGS_H

#include <ros/ros.h>
#include <iostream>

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

} // namespace vio_slam


#endif // SETTINGS_H