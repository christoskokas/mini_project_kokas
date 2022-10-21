#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "Settings.h"
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
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <chrono>


namespace vio_slam
{

class CameraPose
{
    private:

    public:
        Eigen::Matrix4d pose;
        Eigen::Matrix4d poseInverse;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
        //  = std::chrono::high_resolution_clock::now();
        CameraPose(Eigen::Matrix4d _pose = Eigen::Matrix4d::Identity(), std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp = std::chrono::high_resolution_clock::now());
        void setPose(Eigen::Matrix4d poseT);
        void setInvPose(Eigen::Matrix4d poseT);
};

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
        void setIntrinsicValues(const std::string& cameraPath, ConfigFile* confFile);
};

/**
 * @brief Zed Camera class that contains 2 cameras and IMU
 * 
 */

class Zed_Camera
{
    private:
    
    public:
        bool addKeyFrame {false};
        bool rectified {};
        float mBaseline, mFps;
        int mWidth, mHeight;

        Camera cameraLeft;
        Camera cameraRight;

        CameraPose cameraPose;

        ConfigFile* confFile;
        
        cv::Mat sensorsTranslate {};
        cv::Mat sensorsRotate {};
        Zed_Camera(ConfigFile& yamlFile);
        ~Zed_Camera();
        void setCameraMatrices();
        void setCameraValues();

};


}

#endif // CAMERA_H