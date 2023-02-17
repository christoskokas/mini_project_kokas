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

Eigen::Matrix4d convertSE3(const Eigen::Matrix4d& T_wc1,
                           const Eigen::Matrix4d& T_c1c2) {

    // Convert SE3 matrix T_wc1 to a 3D rotation matrix R_wc1 and a translation vector t_wc1
    Eigen::Matrix3d R_wc1 = T_wc1.block<3,3>(0,0);
    Eigen::Vector3d t_wc1 = T_wc1.block<3,1>(0,3);

    // Convert SE3 matrix T_c1c2 to a 3D rotation matrix R_c1c2 and a translation vector t_c1c2
    Eigen::Matrix3d R_c1c2 = T_c1c2.block<3,3>(0,0);
    Eigen::Vector3d t_c1c2 = T_c1c2.block<3,1>(0,3);

    // Compute the new translation vector t_wc2
    Eigen::Vector3d t_wc2 = R_wc1 * t_c1c2 + t_wc1;

    // Compute the new rotation matrix R_wc2
    Eigen::Matrix3d R_wc2 = R_wc1 * R_c1c2;

    // Construct the new SE3 matrix T_wc2
    Eigen::Matrix4d T_wc2 = Eigen::Matrix4d::Identity();
    T_wc2.block<3,3>(0,0) = R_wc2;
    T_wc2.block<3,1>(0,3) = t_wc2;

    return T_wc2;
}

// Converts an SO3 camera pose from one world reference coordinate system to another
// Inputs:
//   R_w1_c: the SO3 camera pose in the first world reference coordinate system
//   T_w1_w2: the 4x4 transformation matrix that relates the first world reference coordinate system to the second
// Outputs:
//   R_w2_c: the SO3 camera pose in the second world reference coordinate system
Eigen::Matrix3d convertSO3(const Eigen::Matrix3d& R_w1_c, const Eigen::Matrix4d& T_w1_w2) {
  // Extract the rotation matrix from the transformation matrix
  Eigen::Matrix3d R_w1_w2 = T_w1_w2.block<3, 3>(0, 0);

  // Compute the inverse of the transformation matrix
  Eigen::Matrix4d T_w2_w1 = T_w1_w2.inverse();

  // Extract the rotation matrix from the inverse transformation matrix
  Eigen::Matrix3d R_w2_w1 = T_w2_w1.block<3, 3>(0, 0);

  // Compute the camera pose in the second world reference coordinate system
  Eigen::Matrix3d R_w2_c = R_w1_w2 * R_w1_c * R_w2_w1;

  return R_w2_c;
}

// Function to convert a camera pose from the ROS Gazebo world coordinate system to the Pangolin cpp world coordinate system.
// Input: T_World_to_Cam_Gazebo - SE3 representation of the camera pose in the ROS Gazebo world coordinate system
// Output: T_World_to_Cam_Pangolin - SE3 representation of the camera pose in the Pangolin cpp world coordinate system
Eigen::Matrix4d convertGazeboToWorld(const Eigen::Matrix4d& T_World_to_Cam_Gazebo)
{
    // Step 1: Get the SE3 representation of the camera pose in the ROS Gazebo world coordinate system
    Eigen::Matrix4d T_Gazebo_to_Cam = T_World_to_Cam_Gazebo.inverse();

    // Step 2: Multiply the SE3 representation from step 1 by a transformation matrix that rotates the coordinate system from ROS Gazebo to Pangolin cpp
    // The transformation matrix is a 4x4 matrix with the following values:
    Eigen::Matrix4d T_Gazebo_to_Pangolin;
    T_Gazebo_to_Pangolin << 0, 0, 1, 0,
                            -1, 0, 0, 0,
                            0, -1, 0, 0,
                            0, 0, 0, 1;

    Eigen::Matrix4d T_Cam_to_World_Pangolin = T_Gazebo_to_Pangolin * T_Gazebo_to_Cam;

    // Step 3: Get the SE3 representation of the camera pose in the Pangolin cpp world coordinate system
    Eigen::Matrix4d T_World_to_Cam_Pangolin = T_Cam_to_World_Pangolin.inverse();

    return T_World_to_Cam_Pangolin;
}

Eigen::Matrix4d gazeboToOpenCV(const Eigen::Matrix4d& pose_in_gazebo) {
    Eigen::Isometry3d pose_gazebo = Eigen::Isometry3d::Identity();
    pose_gazebo.matrix() = pose_in_gazebo;

    // Apply translation from Gazebo world origin to OpenCV world origin
    Eigen::Matrix4d pose_opencv = Eigen::Matrix4d::Identity();
    pose_opencv.block<3,1>(0,3) << pose_gazebo.translation()[1],
                                  -pose_gazebo.translation()[2],
                                  pose_gazebo.translation()[0];

    // Apply rotation from Gazebo world orientation to OpenCV world orientation
    pose_opencv.block<3,3>(0,0) << pose_gazebo.rotation().transpose();

    return pose_opencv;
}

// Define the function to convert camera pose from world coordinate system to OpenCV camera coordinate system
Eigen::Matrix4d convertCameraPoseToOpenCV(const Eigen::Matrix4d& cameraPose) {
    // Define the conversion matrix from world to OpenCV camera coordinate system
    Eigen::Matrix4d convMat;
    convMat << 1, 0, 0, 0,
               0, -1, 0, 0,
               0, 0, -1, 0,
               0, 0, 0, 1;

    // Convert the camera pose to the OpenCV camera coordinate system
    Eigen::Matrix4d openCVPose = convMat * cameraPose * convMat;

    return openCVPose;
}

Eigen::Matrix4d convertCameraPose(const Eigen::Matrix4d& gazeboPose)
{
  // Gazebo coordinate system: X forward, Y left, Z up
  // OpenCV coordinate system: X right, Y down, Z forward
  // We need to rotate the camera pose around the X axis by -90 degrees,
  // and around the Y axis by 180 degrees, to convert from Gazebo to OpenCV

  Eigen::Matrix3d R_x;
  R_x << 1, 0, 0,
         0, 0, -1,
         0, 1, 0;

  Eigen::Matrix3d R_y;
  R_y << -1, 0, 0,
          0, 1, 0,
          0, 0, -1;

  Eigen::Matrix3d R = R_y * R_x;

  Eigen::Matrix4d opencvPose = Eigen::Matrix4d::Identity();
  opencvPose.block<3, 3>(0, 0) = R * gazeboPose.block<3, 3>(0, 0);
  opencvPose.block<3, 1>(0, 3) = R * gazeboPose.block<3, 1>(0, 3);
  return opencvPose;
}

Eigen::Matrix4d convertPoseKitti(const Eigen::Matrix4d& pose_gazebo)
{
    Eigen::Matrix4d T_kitti;

    // Define transformation matrix from Gazebo camera to KITTI camera
    Eigen::Matrix4d T_gazebo_to_kitti;
    T_gazebo_to_kitti << 0, 0, 1, 0,
                         -1, 0, 0, 0,
                         0, -1, 0, 0,
                         0, 0, 0, 1;

    // Define transformation matrix from Gazebo world to KITTI world
    Eigen::Matrix4d T_world_gazebo_to_kitti;
    T_world_gazebo_to_kitti << 1, 0, 0, 0,
                               0, -1, 0, 0,
                               0, 0, -1, 0,
                               0, 0, 0, 1;

    // Compute the KITTI pose by first transforming the Gazebo pose to the KITTI world frame, 
    // and then transforming it to the KITTI camera coordinate frame
    T_kitti = T_world_gazebo_to_kitti * pose_gazebo * T_gazebo_to_kitti;

    return T_kitti;
}

Eigen::Matrix4d convertGazeboToKitti(const Eigen::Matrix4d& pose) {
    // Define the transformation matrices for the coordinate system conversion
    // ROS Gazebo camera coordinate frame to KITTI dataset camera coordinate frame
    const Eigen::Matrix3d R_gazebo_to_kitti = (Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()) *
                                               Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitX())).toRotationMatrix();
    Eigen::Matrix4d T_gazebo_to_kitti = Eigen::Matrix4d::Identity();
    T_gazebo_to_kitti.block<3,3>(0,0) = R_gazebo_to_kitti;

    // Convert the pose from the ROS Gazebo camera coordinate frame to the KITTI dataset camera coordinate frame
    return T_gazebo_to_kitti * pose * T_gazebo_to_kitti.inverse();
}

Eigen::Matrix4d convertToKITTI(const Eigen::Matrix4d& pose_gazebo) {
  // Define the transformation matrices between the two coordinate systems
  // The rotation matrix is just a 180 degree rotation around the x-axis
  Eigen::Matrix3d R_gazebo_to_kitti;
  R_gazebo_to_kitti << 1,  0,  0,
                       0, -1,  0,
                       0,  0, -1;
  Eigen::Matrix4d T_gazebo_to_kitti = Eigen::Matrix4d::Identity();
  T_gazebo_to_kitti.block<3, 3>(0, 0) = R_gazebo_to_kitti;

  // Apply the transformation to the input pose
  Eigen::Matrix4d pose_kitti = T_gazebo_to_kitti * pose_gazebo;

  return pose_kitti;
}

void GetImagesROS::saveTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition)
{
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    Eigen::Quaterniond& q = gtQuaternions[0];
    Eigen::Matrix4d startPose = Eigen::Matrix4d::Identity();
    startPose.block<3,3>(0,0) = q.toRotationMatrix();
    startPose.block<3,1>(0,3) = gtPositions[0];
    Eigen::Matrix4d startPoseT = convertGazeboToWorld(startPose);
    Eigen::Matrix4d startPoseInvT = (startPoseT).inverse();
    Eigen::Matrix4d T_c1_w = T_w_c1.inverse();
    Eigen::Matrix4d startPoseInv = (startPose).inverse();
    Eigen::Matrix3d Rot;
    Rot << 0,1,0,-1,0,0,0,0,1;
    for ( size_t i{0}, end{gtPositions.size()}; i < end; i ++)
    {
        Eigen::Quaterniond& q = gtQuaternions[i];
        Eigen::Matrix4d Pose = Eigen::Matrix4d::Identity();
        Pose.block<3,3>(0,0) = q.toRotationMatrix();
        Pose.block<3,1>(0,3) = startPoseInv.block<3,3>(0,0) * (gtPositions[i] - gtPositions[0]);
        Eigen::Matrix4d PoseTT = Pose;
        PoseTT.block<3,3>(0,0) = ( startPoseInv.block<3,3>(0,0) * Pose.block<3,3>(0,0) );

        // Eigen::Matrix4d PoseTTT = convertGazeboToWorld(Pose);
        // Eigen::Matrix4d PoseT = (startPoseInvT * PoseTTT).transpose();
        // Eigen::Matrix4d PoseTTT = convertCameraPose(PoseTT);
        Eigen::Matrix4d PoseTTT = PoseTT;
        // PoseTTT.block<4,1>(0,3) = T_w_c1 * PoseTT.block<4,1>(0,3);
        // PoseTTT.block<3,3>(0,0) = Rot * PoseTT.block<3,3>(0,0) * Rot.transpose();
        Eigen::Matrix4d PoseT = PoseTTT.transpose();
        // Eigen::Matrix4d PoseTTT = gazeboToOpenCV(PoseTT);
        // Eigen::Matrix4d PoseT = PoseTTT.transpose();
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