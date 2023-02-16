#include "Settings.h"
#include "System.h"
#include "Camera.h"
// #include "FeatureDrawer.h"
#include "trial.h"
#include "Frame.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
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

// TODO MUTEX THREAD FOR FEATURE MATCHING AND LOOP 
// CLOSING SO THAT THE PROGRAM CONTINUES WHILE SEARCHING FOR LOOP CLOSING

// TODO NOW Change camera cpp, add transform from imu to camera

class GetImagesROS
{
    public:
        GetImagesROS(vio_slam::System* _voSLAM) : voSLAM(_voSLAM){};

        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB);

        vio_slam::System* voSLAM;

        cv::Mat rectMap[2][2];
        cv::Mat rectMapB[2][2];
        int frameNumb {0};
};

void signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum << std::endl;
    ros::shutdown();
    // Terminate program
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

    ros::init(argc, argv, "Double Stereo");
    ros::start();
    signal(SIGINT, signal_callback_handler);

    vio_slam::ConfigFile* confFile = new vio_slam::ConfigFile(file.c_str());


    bool multi {false};
    try{
        multi = confFile->getValue<bool>("multi");
    }
    catch(std::exception& e)
    {
        multi = false;
    }

    vio_slam::System* voSLAM;
    


    
    if ( multi )
        voSLAM = new vio_slam::System(confFile, multi);
    else
        voSLAM = new vio_slam::System(confFile);

    GetImagesROS imgROS(voSLAM);

    vio_slam::Zed_Camera* mZedCamera = voSLAM->mZedCamera;
    vio_slam::Zed_Camera* mZedCameraB = voSLAM->mZedCameraB;

    mZedCamera->numOfFrames = INT_MAX;
    mZedCameraB->numOfFrames = INT_MAX;


    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[0][0], imgROS.rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[1][0], imgROS.rectMap[1][1]);

    }

    if ( !mZedCameraB->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCameraB->cameraLeft.K, mZedCameraB->cameraLeft.D, mZedCameraB->cameraLeft.R, mZedCameraB->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMapB[0][0], imgROS.rectMapB[0][1]);
        cv::initUndistortRectifyMap(mZedCameraB->cameraRight.K, mZedCameraB->cameraRight.D, mZedCameraB->cameraRight.R, mZedCameraB->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMapB[1][0], imgROS.rectMapB[1][1]);

    }

    cv::Mat im(mZedCamera->mHeight, mZedCamera->mWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::namedWindow("Tracked KeyPoints");
    cv::imshow("Tracked KeyPoints", im);
    cv::waitKey(1);

    cv::namedWindow("Tracked KeyPointsB");
    cv::imshow("Tracked KeyPointsB", im);
    cv::waitKey(1000);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/camera_1/left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/camera_1/right/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> left_subB(nh, "/camera_4/left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> right_subB(nh, "/camera_4/right/image_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, left_subB,right_subB);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3,_4));

    ros::spin();

    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("camTrajectory.txt", "camPosition.txt");
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);

    return 0;
}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrLeftB;
    try
    {
        cv_ptrLeftB = cv_bridge::toCvShare(msgLeftB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRightB;
    try
    {
        cv_ptrRightB = cv_bridge::toCvShare(msgRightB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imLeft, imRight;
    cv::Mat imLeftB, imRightB;
    if( !voSLAM->mZedCamera->rectified )
    {
        cv::remap(cv_ptrLeft->image,imLeft, rectMap[0][0],rectMap[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image,imRight,rectMap[1][0],rectMap[1][1],cv::INTER_LINEAR);
    }
    if( !voSLAM->mZedCameraB->rectified )
    {
        cv::remap(cv_ptrLeftB->image,imLeftB, rectMapB[0][0],rectMapB[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRightB->image,imRightB,rectMapB[1][0],rectMapB[1][1],cv::INTER_LINEAR);
    }
    if ( cv_ptrLeft->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeft->image, imLeft, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRight->image, imRight, CV_GRAY2BGR);
    }
    else
    {
        imLeft = cv_ptrLeft->image;
        imRight = cv_ptrRight->image;
    }
    if ( cv_ptrLeftB->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeftB->image, imLeftB, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRightB->image, imRightB, CV_GRAY2BGR);
    }
    else
    {
        imLeftB = cv_ptrLeftB->image;
        imRightB = cv_ptrRightB->image;
    }
    voSLAM->trackNewImageMutli(imLeft, imRight, imLeftB, imRightB, frameNumb);
    frameNumb ++;
}