#include "Settings.h"
#include "System.h"
#include "Camera.h"
// #include "FeatureDrawer.h"
#include "trial.h"
#include "Frame.h"
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
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

#define GTPOSE false
#define PUBPOINTCLOUD true

class GetImagesROS
{
    public:
        GetImagesROS(vio_slam::System* _voSLAM) : voSLAM(_voSLAM){};

        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB,const nav_msgs::OdometryConstPtr& msgGT);
        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB);

        void saveGTTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition);
        void publishOdom(const std_msgs::Header& _header);

        ros::Publisher pc_pub;
        ros::Publisher odom_pub;
        tf::TransformBroadcaster odom_broadcaster;

        vio_slam::System* voSLAM;

        std::vector<Eigen::Vector3d> gtPositions;
        std::vector<Eigen::Quaterniond> gtQuaternions;

        Eigen::Matrix4d T_w_c1;

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

    ros::init(argc, argv, "Double_Stereo");
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
    const std::string leftPathB = confFile->getValue<std::string>("leftPathCamB");
    const std::string rightPathB = confFile->getValue<std::string>("rightPathCamB");

    ros::NodeHandle nh;
    
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, leftPath, 10);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, rightPath, 10);
    message_filters::Subscriber<sensor_msgs::Image> left_subB(nh, leftPathB, 10);
    message_filters::Subscriber<sensor_msgs::Image> right_subB(nh, rightPathB, 10);
    imgROS.odom_pub = nh.advertise<nav_msgs::Odometry>("/odom",1);
#if PUBPOINTCLOUD
    std::vector<double> Twc1 = confFile->getValue<std::vector<double>>("T_w_c1","data");
    imgROS.T_w_c1 << Twc1[0],Twc1[1],Twc1[2],Twc1[3],Twc1[4],Twc1[5],Twc1[6],Twc1[7],Twc1[8],Twc1[9],Twc1[10],Twc1[11],Twc1[12],Twc1[13],Twc1[14],Twc1[15];
    imgROS.pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/vio_slam/points2",1);
#endif
#if GTPOSE
    message_filters::Subscriber<nav_msgs::Odometry> gt_sub(nh, gtPath, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, left_subB,right_subB, gt_sub);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3,_4, _5));
#else
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, left_subB,right_subB);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3,_4));


#endif
    ros::spin();

    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("dual_cam_traj.txt", "dual_cam_pos.txt");
#if GTPOSE
    imgROS.saveGTTrajectoryAndPositions("GTDcamTrajectory.txt", "GTDcamPosition.txt");
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
    datafile.close();
    datafilePos.close();

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
#if PUBPOINTCLOUD
    pcl::PointCloud<pcl::PointXYZRGB> cloud_;
    const float offset {0.02f};
    const std::vector<vio_slam::MapPoint*>& actMPs = voSLAM->map->activeMapPoints;
    for (size_t i {0}, end{actMPs.size()}; i < end; i ++)
    {
        const vio_slam::MapPoint* mp = actMPs[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        // std::cout << "point : " << pt << std::endl;
        if ( pt(2) < (T_w_c1(2,3) + offset) || pt(2) > offset )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    const std::vector<vio_slam::MapPoint*>& actMPsB = voSLAM->map->activeMapPointsB;
    for (size_t i {0}, end{actMPsB.size()}; i < end; i ++)
    {
        const vio_slam::MapPoint* mp = actMPsB[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        // std::cout << "point : " << pt << std::endl;
        if ( pt(2) < (T_w_c1(2,3) + offset) || pt(2) > offset )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    sensor_msgs::PointCloud2 pc2_msg_;
    pcl::toROSMsg(cloud_, pc2_msg_);
    pc2_msg_.header.frame_id = "map";
    pc2_msg_.header.stamp = msgLeft->header.stamp;
    pc_pub.publish(pc2_msg_);
#endif
    publishOdom(msgLeft->header);
}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB,const nav_msgs::OdometryConstPtr& msgGT)
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
#if PUBPOINTCLOUD
    pcl::PointCloud<pcl::PointXYZRGB> cloud_;
    const float offset {0.02f};
    const std::vector<vio_slam::MapPoint*>& actMPs = voSLAM->map->activeMapPoints;
    for (size_t i {0}, end{actMPs.size()}; i < end; i ++)
    {
        const vio_slam::MapPoint* mp = actMPs[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        // std::cout << "point : " << pt << std::endl;
        if ( pt(2) < (T_w_c1(2,3) + offset) || pt(2) > offset )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    const std::vector<vio_slam::MapPoint*>& actMPsB = voSLAM->map->activeMapPointsB;
    for (size_t i {0}, end{actMPsB.size()}; i < end; i ++)
    {
        const vio_slam::MapPoint* mp = actMPsB[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        // std::cout << "point : " << pt << std::endl;
        if ( pt(2) < (T_w_c1(2,3) + offset) || pt(2) > offset )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    sensor_msgs::PointCloud2 pc2_msg_;
    pcl::toROSMsg(cloud_, pc2_msg_);
    pc2_msg_.header.frame_id = "map";
    pc2_msg_.header.stamp = msgLeft->header.stamp;
    pc_pub.publish(pc2_msg_);
#endif
    publishOdom(msgLeft->header);
}

void GetImagesROS::publishOdom(const std_msgs::Header& _header)
{
    Eigen::Matrix4d camPose = voSLAM->mZedCamera->cameraPose.pose;
    const Eigen::Quaterniond q(camPose.block<3,3>(0,0));
    Eigen::Vector3d tra = camPose.block<3,1>(0,3);
    tra = T_w_c1.block<3,3>(0,0) * tra;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = _header.stamp;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_footprint";

    odom_trans.transform.translation.x = tra(0);
    odom_trans.transform.translation.y = tra(1);
    odom_trans.transform.translation.z = tra(2);
    
    odom_trans.transform.rotation.w = -q.w();
    odom_trans.transform.rotation.x = -q.z();
    odom_trans.transform.rotation.y = q.x();
    odom_trans.transform.rotation.z = q.y();

    odom_broadcaster.sendTransform(odom_trans);
    
    nav_msgs::Odometry odom;
    odom.header.stamp = _header.stamp;
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = tra(0);
    odom.pose.pose.position.y = tra(1);
    odom.pose.pose.position.z = tra(2);

    odom.pose.pose.orientation.w = -q.w();
    odom.pose.pose.orientation.x = -q.z();
    odom.pose.pose.orientation.y = q.x();
    odom.pose.pose.orientation.z = q.y();

    //set the velocity
    odom.child_frame_id = "base_footprint";
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;

    odom_pub.publish(odom);
}
