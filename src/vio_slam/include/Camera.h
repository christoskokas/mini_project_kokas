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
#include <Eigen/LU>
#include <chrono>


namespace vio_slam
{

class CameraPose
{
    private:
        double vx {}, vy {}, vz {};
    public:
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d rTwc = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d rTcw = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d refPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d Rv;
        Eigen::Vector3d tv;
        Eigen::Matrix4d poseInverse = Eigen::Matrix4d::Identity();
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

        //  = std::chrono::high_resolution_clock::now();
        CameraPose(Eigen::Matrix4d _pose = Eigen::Matrix4d::Identity(), std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp = std::chrono::high_resolution_clock::now());
        void setVel(const double _vx, const double _vy, const double _vz);
        void setPose(const Eigen::Matrix4d& poseT);
        void setPose(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose);
        Eigen::Matrix4d getPose() const;
        Eigen::Matrix4d getInvPose() const;
        void changePose(Eigen::Matrix4d& _keyPose);
        void setInvPose(const Eigen::Matrix4d poseT);
        void separatePose();
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
        // ros::Publisher pub;
        // ros::Subscriber camera_subscriber;
        // ros::Subscriber imu_subscriber;
    public:
        double fx {},fy {},cx {}, cy {};
        double k1 {}, k2 {}, p1 {}, p2 {}, k3{};
        std::string path {};
        cv::Mat cameraMatrix {};
        cv::Mat D = cv::Mat::zeros(1,5,CV_64F);
        cv::Mat K = cv::Mat::eye(3,3,CV_64F);
        cv::Mat R = cv::Mat::eye(3,3,CV_64F);
        cv::Mat P = cv::Mat::eye(3,4,CV_64F);
        Eigen::Matrix<double,3,3> intrisics = Eigen::Matrix<double,3,3>::Identity();
        cv::Mat distCoeffs {};
        Camera(ros::NodeHandle *nh);
        Camera() = default;
        ~Camera();
        float GetFx();
        void setIntrinsicValuesUnR(const std::string& cameraPath, ConfigFile* confFile);
        void setIntrinsicValuesR(const std::string& cameraPath, ConfigFile* confFile);
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
        int numOfFrames {};
        std::string seq {};

        Camera cameraLeft;
        Camera cameraRight;

        CameraPose cameraPose;

        ConfigFile* confFile;
        Eigen::Matrix<double,4,4> extrinsics = Eigen::Matrix<double,4,4>::Identity();
        cv::Mat sensorsTranslate {};
        cv::Mat sensorsRotate {};
        Zed_Camera(ConfigFile* yamlFile);
        ~Zed_Camera();
        void setCameraMatrices();
        void setCameraValues();
        float setBaseline();

};


}

#endif // CAMERA_H