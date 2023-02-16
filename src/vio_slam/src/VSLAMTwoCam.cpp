#include "Settings.h"
#include "System.h"
#include "Camera.h"
// #include "FeatureDrawer.h"
#include "trial.h"
#include "Frame.h"
#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <iostream>
#include <sstream>
#include <string>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <signal.h>

volatile sig_atomic_t flag = 0;

// TODO MUTEX THREAD FOR FEATURE MATCHING AND LOOP 
// CLOSING SO THAT THE PROGRAM CONTINUES WHILE SEARCHING FOR LOOP CLOSING

// TODO NOW Change camera cpp, add transform from imu to camera

void signal_callback_handler(int signum) {
    flag = 1;
}

int main (int argc, char **argv)
{
#if KITTI_DATASET
    std::string file = std::string("config_kitti_") + KITTI_SEQ + std::string(".yaml");
    // vio_slam::ConfigFile yamlFile(file.c_str());
#elif ZED_DATASET
    std::string file = "config_exp.yaml";
    // vio_slam::ConfigFile yamlFile("config_exp.yaml");
#elif ZED_DEMO
    std::string file = "config_demo_zed.yaml";
#elif V1_02
    std::string file = "config_V1_02.yaml";
#elif SIMULATION
    std::string file = "config_simulation.yaml";
#else
    std::string file = "config.yaml";
    // vio_slam::ConfigFile yamlFile("config.yaml");
#endif

    signal(SIGINT, signal_callback_handler);

    vio_slam::ConfigFile* confFile = new vio_slam::ConfigFile(file.c_str());


    bool multi {false};
    try{
        multi = confFile->getValue<bool>("multi");
    }
    catch(std::exception& e)
    {
        std::cerr << "multi variable not set in config file!" << std::endl;
        return 1;
    }

    vio_slam::System* voSLAM;
    
    if ( multi )
        voSLAM = new vio_slam::System(confFile, multi);
    else
        voSLAM = new vio_slam::System(confFile);

    const vio_slam::Zed_Camera* mZedCamera = voSLAM->mZedCamera;
    const vio_slam::Zed_Camera* mZedCameraB = voSLAM->mZedCameraB;

        const int nFrames {mZedCamera->numOfFrames};
    std::vector<std::string>leftImagesStr, rightImagesStr, leftImagesStrB, rightImagesStrB;
    leftImagesStr.reserve(nFrames);
    rightImagesStr.reserve(nFrames);
    leftImagesStrB.reserve(nFrames);
    rightImagesStrB.reserve(nFrames);

    const std::string imagesPath = confFile->getValue<std::string>("imagesPath");

    const std::string leftPath = imagesPath + "left/";
    const std::string rightPath = imagesPath + "right/";
    const std::string leftPathB = imagesPath + "leftBack/";
    const std::string rightPathB = imagesPath + "rightBack/";
    const std::string fileExt = confFile->getValue<std::string>("fileExtension");

    const size_t imageNumbLength = 6;

    for ( size_t i {0}; i < nFrames; i++)
    {
        std::string frameNumb = std::to_string(i);
        std::string frameStr = std::string(imageNumbLength - std::min(imageNumbLength, frameNumb.length()), '0') + frameNumb;
        leftImagesStr.emplace_back(leftPath + frameStr + fileExt);
        rightImagesStr.emplace_back(rightPath + frameStr + fileExt);
        leftImagesStrB.emplace_back(leftPathB + frameStr + fileExt);
        rightImagesStrB.emplace_back(rightPathB + frameStr + fileExt);
    }

    cv::Mat rectMap[2][2], rectMapB[2][2];
    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[1][0], rectMap[1][1]);
    }

    if ( !mZedCameraB->rectified)
    {
        cv::initUndistortRectifyMap(mZedCameraB->cameraLeft.K, mZedCameraB->cameraLeft.D, mZedCameraB->cameraLeft.R, mZedCameraB->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMapB[0][0], rectMapB[0][1]);
        cv::initUndistortRectifyMap(mZedCameraB->cameraRight.K, mZedCameraB->cameraRight.D, mZedCameraB->cameraRight.R, mZedCameraB->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMapB[1][0], rectMapB[1][1]);
    }
    
    double timeBetFrames = 1.0/mZedCamera->mFps;

    for ( size_t frameNumb{0}; frameNumb < nFrames; frameNumb++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat imageLeft = cv::imread(leftImagesStr[frameNumb],cv::IMREAD_COLOR);
        cv::Mat imageRight = cv::imread(rightImagesStr[frameNumb],cv::IMREAD_COLOR);
        cv::Mat imageLeftB = cv::imread(leftImagesStrB[frameNumb],cv::IMREAD_COLOR);
        cv::Mat imageRightB = cv::imread(rightImagesStrB[frameNumb],cv::IMREAD_COLOR);

        cv::Mat imLRect, imRRect;
        cv::Mat imLRectB, imRRectB;

        if ( !mZedCamera->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
            cv::remap(imageRight, imRRect, rectMap[1][0], rectMap[1][1], cv::INTER_LINEAR);
        }
        else
        {
            imLRect = imageLeft.clone();
            imRRect = imageRight.clone();
        }

        if ( !mZedCameraB->rectified )
        {
            cv::remap(imageLeftB, imLRectB, rectMapB[0][0], rectMapB[0][1], cv::INTER_LINEAR);
            cv::remap(imageRightB, imRRectB, rectMapB[1][0], rectMapB[1][1], cv::INTER_LINEAR);
        }
        else
        {
            imLRectB = imageLeftB.clone();
            imRRectB = imageRightB.clone();
        }

        voSLAM->trackNewImageMutli(imLRect, imRRect, imLRectB, imRRectB, frameNumb);


        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
        // Logging("DURATION", duration,3);

        if ( duration < timeBetFrames )
            usleep((timeBetFrames-duration)*1e6);

        if ( flag == 1 )
            break;

    }

    
    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("camTrajectory.txt", "camPosition.txt");
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);


}