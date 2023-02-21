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

#define GTPOSE true

class GetImagesROS
{
    public:
        GetImagesROS(vio_slam::System* _voSLAM) : voSLAM(_voSLAM){};

        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight,const nav_msgs::OdometryConstPtr& msgGT);
        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight);

        void saveGTTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition);

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


    vio_slam::System* voSLAM;

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

    bool gtPose {false};
    try
    {
        gtPose = confFile->getValue<bool>("gtPose");
    }
    catch(const std::exception& e)
    {
        gtPose = false;
    }
    const std::string gtPath = (gtPose) ? confFile->getValue<std::string>("gtPath") : "";

    const std::string leftPath = confFile->getValue<std::string>("leftPathCam");
    const std::string rightPath = confFile->getValue<std::string>("rightPathCam");
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, leftPath, 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, rightPath, 1);

#if GTPOSE
    message_filters::Subscriber<nav_msgs::Odometry> gt_sub(nh, gtPath, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, gt_sub);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3));

#else
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2));

#endif

    // message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/kitti/camera_gray/left/image_rect", 1);
    // message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/kitti/camera_gray/right/image_rect", 1);

    ros::spin();

    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("single_cam_traj.txt", "sigle_cam_pos.txt");
#if GTPOSE
    imgROS.saveGTTrajectoryAndPositions("ground_truth_traj.txt", "ground_truth_pos.txt");
#endif
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);




    return 0;
}

void GetImagesROS::saveGTTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition)
{
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    Eigen::Quaterniond& q = gtQuaternions[0];
    Eigen::Matrix4d startPose = Eigen::Matrix4d::Identity();
    // tf tf_echo base_footprint /left_camera_optical_frame -> 0.15 0.06 0.25
    // then fill in baseToCam with x->-y, y->z, z->x but z = 0 so -0.06 0.0 0.15
    Eigen::Vector3d baseToCam(-0.06,0.0,0.15);
    startPose.block<3,3>(0,0) = q.toRotationMatrix();
    startPose.block<3,1>(0,3) = startPose.block<3,3>(0,0) 
    * baseToCam + gtPositions[0];
    Eigen::Matrix4d startPoseInv = (startPose).inverse();
    for ( size_t i{0}, end{gtPositions.size()}; i < end; i ++)
    {
        Eigen::Quaterniond& q = gtQuaternions[i];
        Eigen::Matrix4d Pose = Eigen::Matrix4d::Identity();
        Pose.block<3,3>(0,0) = q.toRotationMatrix();
        Pose.block<3,1>(0,3) = q.toRotationMatrix() * baseToCam +  gtPositions[i];
        // Pose.block<3,1>(0,3) = startPoseInv.block<3,3>(0,0) * (gtPositions[i] - gtPositions[0]);
        // PoseTT.block<3,3>(0,0) = ( startPoseInv.block<3,3>(0,0) * Pose.block<3,3>(0,0) );
        Eigen::Matrix4d PoseTT =  startPoseInv * Pose;

        Eigen::Matrix4d PoseT = PoseTT.transpose();
        for (int32_t j{0}; j < 12; j ++)
        {
            if ( j == 0 )
                datafile << PoseT(j);
            else
                datafile << " " << PoseT(j);
            if ( j == 3 || j == 7 || j == 11 )
                datafilePos << PoseT(j) << " ";
        }
        datafile << '\n';
        datafilePos << '\n';
    }
}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight,const nav_msgs::OdometryConstPtr& msgGT)
{
    gtPositions.emplace_back(-msgGT->pose.pose.position.y, msgGT->pose.pose.position.z, msgGT->pose.pose.position.x);
    gtQuaternions.emplace_back(-msgGT->pose.pose.orientation.w, msgGT->pose.pose.orientation.y, msgGT->pose.pose.orientation.z, -msgGT->pose.pose.orientation.x);

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

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight)
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
