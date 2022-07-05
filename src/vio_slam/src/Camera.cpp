#include "Camera.h"
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


#define CAMERA_PATH "/camera_1/right/image_raw"
#define CAMERA_PATH_2 "/camera_1/left/image_raw"
#define POINTCLOUD_PATH "/camera_1/points2"
#define IMU_PATH "/imu/data"

namespace vio_slam
{

Zed_Camera::Zed_Camera(ros::NodeHandle *nh)
{

    nh->getParam("/Camera/width",m_width);
    nh->getParam("/Camera/height",m_height);
    nh->getParam("/Camera/fps",m_fps);
    nh->getParam("/Camera/bl",m_baseline);
    nh->getParam("/Camera_l/fx",camera_left.fx);
    nh->getParam("/Camera_l/fy",camera_left.fy);
    nh->getParam("/Camera_l/cx",camera_left.cx);
    nh->getParam("/Camera_l/cy",camera_left.cy);
    nh->getParam("/Camera_l/k1",camera_left.k1);
    nh->getParam("/Camera_l/k2",camera_left.k2);
    nh->getParam("/Camera_l/p1",camera_left.p1);
    nh->getParam("/Camera_l/p2",camera_left.p2);
    nh->getParam("/Camera_l/k3",camera_left.k3);
    nh->getParam("/Camera_r/fx",camera_right.fx);
    nh->getParam("/Camera_r/fy",camera_right.fy);
    nh->getParam("/Camera_r/cx",camera_right.cx);
    nh->getParam("/Camera_r/cy",camera_right.cy);
    nh->getParam("/Camera_r/k1",camera_right.k1);
    nh->getParam("/Camera_r/k2",camera_right.k2);
    nh->getParam("/Camera_r/p1",camera_right.p1);
    nh->getParam("/Camera_r/p2",camera_right.p2);
    nh->getParam("/Camera_r/k3",camera_right.k3);
}

Zed_Camera::~Zed_Camera()
{

}

void Zed_Camera::GetResolution()
{
    ROS_INFO("Height : [%d], Width : [%d]", m_height, m_width);
}

Camera::Camera(ros::NodeHandle *nh)
{
    // counter = 0;
    // camera_time = 0;
    // imu_time = 0;
    // camera_subscriber = nh->subscribe(CAMERA_PATH_2, 1000, 
    //     &Camera::callback_number, this);
    // imu_subscriber = nh->subscribe(IMU_PATH, 1000, 
    //     &Camera::callback_number_2, this);
}

Camera::~Camera()
{
    
}

// void Camera::callback_number(const sensor_msgs::Image& msg) 
// {
//     camera_time = msg.header.stamp.sec + msg.header.stamp.nsec*1e-9;
// }

// void Camera::callback_number_2(const sensor_msgs::Imu& msg_2)
// {
//     imu_time = msg_2.header.stamp.sec + msg_2.header.stamp.nsec*1e-9;
//     // ROS_INFO("The time difference between camera-imu is : [%f] \n", imu_time - camera_time);
// }

float Camera::GetFx()
{
    return fx;
}

void Camera::GetIntrinsicValues()
{
    ROS_INFO("\n fx : [%f] \n fy : [%f] \n cx : [%f] \n cy : [%f] \n", fx, fy, cx, cy);
}

} //namespace vio_slam

