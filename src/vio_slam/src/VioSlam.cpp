#include "Camera.h"
#include "Viewer.h"
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

// using namespace std;

int main (int argc, char **argv)
{
    ros::init(argc, argv, "VioSlam");
    ros::NodeHandle nh;
    vio_slam::Zed_Camera zedcamera(&nh);
    std::cout << "xd      " << zedcamera.camera_left->GetFx() << std::endl;
    zedcamera.GetResolution();
    std::cout << "Left Camera"  << std::endl;
    zedcamera.camera_left->GetIntrinsicValues();
    std::cout << "Right Camera" << std::endl;
    zedcamera.camera_right->GetIntrinsicValues();
    vio_slam::FeatureDrawer fv(&nh);
    // Zed_Camera::Camera_2 camera_right = Zed_Camera::Camera_2(&nh);
    // Zed_Camera::Camera_2 camera_rightfx = Zed_Camera::Camera2::getFx();
    ros::spin();
}