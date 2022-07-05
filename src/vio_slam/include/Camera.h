#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include <ros/ros.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"

namespace vio_slam
{

/**
 * @brief Camera class that contains intrinsic values
 * 
 */

class Camera
{
    private:
        int counter;
        float camera_time;
        float imu_time;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
        ros::Subscriber imu_subscriber;
    public:
        double fx {},fy {},cx {}, cy {};
        double k1 {}, k2 {}, p1 {}, p2 {}, k3{};
        std::string path {};
        cv::Mat cameraMatrix {};
        cv::Mat distCoeffs {};
        Camera(ros::NodeHandle *nh);
        Camera() = default;
        ~Camera();
        float GetFx();
        void setIntrinsicValues(ros::NodeHandle* nh, const std::string& cameraPath);
};

/**
 * @brief Zed Camera class that contains 2 cameras and IMU
 * 
 */

class Zed_Camera
{
    private:
    
    public:
        bool rectified {};
        float mBaseline, mFps;
        int mWidth, mHeight;
        Camera cameraLeft;
        Camera cameraRight;
        cv::Mat sensorsTranslate {};
        cv::Mat sensorsRotate {};
        Zed_Camera(ros::NodeHandle *nh, bool rectified);
        ~Zed_Camera();
        void GetResolution();
        void setCameraMatrices(ros::NodeHandle* nh);
        void setCameraValues(ros::NodeHandle* nh);

};


}

#endif // CAMERA_H