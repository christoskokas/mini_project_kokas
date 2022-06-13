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

using namespace std;

int main (int argc, char **argv)
{
    ros::init(argc, argv, "Camera");
    ros::NodeHandle nh;
    vio_slam::Zed_Camera zedcamera(&nh);
    zedcamera.camera_left;
    // std::cout << "xd" << zedcamera.camera_left->getFx() << std::endl;
    // Zed_Camera::Camera_2 camera_right = Zed_Camera::Camera_2(&nh);
    // Zed_Camera::Camera_2 camera_rightfx = Zed_Camera::Camera2::getFx();
    ros::spin();
}