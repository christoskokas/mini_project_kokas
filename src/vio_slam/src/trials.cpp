#include "Settings.h"
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

// TODO MUTEX THREAD FOR FEATURE MATCHING AND LOOP 
// CLOSING SO THAT THE PROGRAM CONTINUES WHILE SEARCHING FOR LOOP CLOSING

// TODO NOW Change camera cpp, add transform from imu to camera

int main (int argc, char **argv)
{
    vio_slam::ConfigFile yamlFile("config.yaml");
    std::cout << " YAML FILE \n " << yamlFile.configNode["Camera_l"]["fx"].as<float>() << '\n';

    std::cout << " YAML FILE TEMPLATE \n " << yamlFile.getValue<std::string>("Camera_l_path") << '\n';

    // ros::init(argc, argv, "trial");
    // ros::NodeHandle nh;
    
    // vio_slam::FeatureStrategy featureMatchingStrat = vio_slam::FeatureStrategy::orb;
    vio_slam::Zed_Camera zedcamera(yamlFile);
    vio_slam::Logging lel("now",2);
    vio_slam::Zed_Camera* zedptr = &zedcamera;
    // vio_slam::FeatureDrawer fv(&nh, zedptr);
    vio_slam::Frame frame;
    std::cout << "\nFeature Extraction Trials\n" << '\n';
    std::cout << "-------------------------\n";
    vio_slam::RobustMatcher2 rb(zedptr);
    std::thread worker(&vio_slam::Frame::pangoQuit, frame, zedptr);
    std::thread tester(&vio_slam::RobustMatcher2::beginTest, &rb);
    // std::thread worker(&vio_slam::Frame::pangoQuit, frame, &nh, &fv.leftImage.pointsPosition);
    // Zed_Camera::Camera_2 camera_right = Zed_Camera::Camera_2(&nh);
    // Zed_Camera::Camera_2 camera_rightfx = Zed_Camera::Camera2::getFx();
    // ros::spin();
    worker.join();
    tester.join();
}
