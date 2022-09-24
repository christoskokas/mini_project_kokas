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
        clock_t start;
        clock_t total;
        const char* wFunc;
    public:
        ProcessTime(const char* what = "whatever");
        void totalTime();

};

} // namespace vio_slam


#endif // SETTINGS_H