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
#include <nav_msgs/Odometry.h>
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

class GetImagesROS
{
    public:
        GetImagesROS(vio_slam::System* _voSLAM) : voSLAM(_voSLAM){};

        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight,const nav_msgs::OdometryConstPtr& msgGT);

        void saveTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition);

        // void saveGT(const nav_msgs::Odometry::ConstPtr& msgGT);

        vio_slam::System* voSLAM;

        std::vector<Eigen::Vector3d> gtPositions;
        std::vector<Eigen::Quaterniond> gtQuaternions;

        Eigen::Matrix4d T_w_c1;

        cv::Mat rectMap[2][2];
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

    ros::init(argc, argv, "Stereo_VSLAM");
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

    mZedCamera->numOfFrames = INT_MAX;

    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[0][0], imgROS.rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[1][0], imgROS.rectMap[1][1]);

    }

    cv::Mat im(mZedCamera->mHeight, mZedCamera->mWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::namedWindow("Tracked KeyPoints");
    cv::imshow("Tracked KeyPoints", im);
    cv::waitKey(1000);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/camera_1/left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/camera_1/right/image_rect", 1);
    message_filters::Subscriber<nav_msgs::Odometry> gt_sub(nh, "/ground_truth/left_camera_optical_frame_1", 1);
    // ros::Subscriber gt_sub;
    // gt_sub = nh.subscribe<nav_msgs::Odometry>("/ground_truth/left_camera_optical_frame_1", 1, &GetImagesROS::saveGT, &imgROS);
    

    // message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/kitti/camera_gray/left/image_rect", 1);
    // message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/kitti/camera_gray/right/image_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, gt_sub);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3));

    ros::spin();

    try
    {
        std::vector<double> _T_w_c1 = confFile->getValue<std::vector<double>>("T_w_c1", "data");
        imgROS.T_w_c1 << _T_w_c1[0],_T_w_c1[1],_T_w_c1[2],_T_w_c1[3],_T_w_c1[4],_T_w_c1[5],_T_w_c1[6],_T_w_c1[7],_T_w_c1[8],_T_w_c1[9],_T_w_c1[10],_T_w_c1[11],_T_w_c1[12],_T_w_c1[13],_T_w_c1[14],_T_w_c1[15];
    }
    catch(const std::exception& e)
    {
    }

    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("camTrajectory.txt", "camPosition.txt");
    imgROS.saveTrajectoryAndPositions("GTcamTrajectory.txt", "GTcamPosition.txt");
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);




    return 0;
}

// void GetImagesROS::saveGT(const nav_msgs::Odometry::ConstPtr& msgGT)
// {
//     std::cout << "gt : " << msgGT->pose.pose.position << std::endl;
// }

void GetImagesROS::saveTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition)
{
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    Eigen::Quaterniond& q = gtQuaternions[0];
    Eigen::Matrix4d startPose = Eigen::Matrix4d::Identity();
    startPose.block<3,3>(0,0) = q.toRotationMatrix();
    startPose.block<3,1>(0,3) = gtPositions[0];
    Eigen::Matrix4d startPoseInv = startPose.inverse();
    Eigen::Matrix4d T_c1_w = T_w_c1.inverse();
    std::cout << "T_C1_W" << T_c1_w << std::endl;
    for ( size_t i{0}, end{gtPositions.size()}; i < end; i ++)
    {
        Eigen::Quaterniond& q = gtQuaternions[i];
        Eigen::Matrix4d Pose = Eigen::Matrix4d::Identity();
        Pose.block<3,3>(0,0) = q.toRotationMatrix();
        Pose.block<3,1>(0,3) = gtPositions[i];
        std::cout << "startPoseInv * Pose" << (startPoseInv * Pose).transpose() << std::endl;
        std::cout << "startPoseInv * Pose * T_c1_w" << startPoseInv * Pose * T_c1_w << std::endl;
        Eigen::Matrix4d PoseT = (T_c1_w * startPoseInv * Pose).transpose();
        std::cout << "PoseT" << PoseT << std::endl;
        for (int32_t i{0}; i < 12; i ++)
        {
            if ( i == 0 )
                datafile << PoseT(i);
            else
                datafile << " " << PoseT(i);
            if ( i == 3 || i == 7 || i == 11 )
                datafilePos << PoseT(i) << " ";
        }
        datafile << '\n';
        datafilePos << '\n';
    }
}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight,const nav_msgs::OdometryConstPtr& msgGT)
{
    gtPositions.emplace_back(msgGT->pose.pose.position.x, msgGT->pose.pose.position.y, msgGT->pose.pose.position.z);
    gtQuaternions.emplace_back(msgGT->pose.pose.orientation.w, msgGT->pose.pose.orientation.x, msgGT->pose.pose.orientation.y, msgGT->pose.pose.orientation.z);

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

    cv::Mat imLeft, imRight;
    if( !voSLAM->mZedCamera->rectified )
    {
        cv::remap(cv_ptrLeft->image,imLeft, rectMap[0][0],rectMap[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image,imRight,rectMap[1][0],rectMap[1][1],cv::INTER_LINEAR);
        voSLAM->trackNewImage(imLeft, imRight, frameNumb);
    }
    else
    {
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
        voSLAM->trackNewImage(imLeft, imRight, frameNumb);
    }
    frameNumb ++;
}