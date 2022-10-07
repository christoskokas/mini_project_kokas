#include "Camera.h"
#include "Settings.h"
#include "FeatureDrawer.h"
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

// TODO MUTEX THREAD FOR FEATURE MATCHING AND LOOP 
// CLOSING SO THAT THE PROGRAM CONTINUES WHILE SEARCHING FOR LOOP CLOSING

// TODO NOW Change camera cpp, add transform from imu to camera

int main (int argc, char **argv)
{
    vio_slam::ConfigFile yamlFile("config.yaml");
    ros::init(argc, argv, "VioSlam");
    ros::NodeHandle nh;
    bool rectified {};
    nh.getParam("rectified",rectified);
    vio_slam::FeatureStrategy featureMatchingStrat = vio_slam::FeatureStrategy::orb;
    vio_slam::Zed_Camera zedcamera(yamlFile);
    std::cout << "xd      " << zedcamera.cameraLeft.GetFx() << std::endl;
    const vio_slam::Zed_Camera* zedptr = &zedcamera;
    vio_slam::FeatureDrawer fv(&nh, zedptr);
    vio_slam::Frame frame;
    std::thread worker(&vio_slam::Frame::pangoQuit, frame,zedptr);
    // Zed_Camera::Camera_2 camera_right = Zed_Camera::Camera_2(&nh);
    // Zed_Camera::Camera_2 camera_rightfx = Zed_Camera::Camera2::getFx();
    ros::spin();
    worker.join();
}