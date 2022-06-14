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

// class Zed_Camera {
//     private:
//         float m_baseline;
        
//     public:
//         class Camera {
//             private:
//                 int counter;
//                 float camera_time;
//                 float imu_time;
//                 float fx = 260.04;
//                 ros::Publisher pub;
//                 ros::Subscriber camera_subscriber;
//                 ros::Subscriber imu_subscriber;
//             public:
//                 Camera(ros::NodeHandle *nh) {
//                     counter = 0;
//                     camera_time = 0;
//                     imu_time = 0;
//                     camera_subscriber = nh->subscribe(CAMERA_PATH_2, 1000, 
//                         &Camera::callback_number, this);
//                     imu_subscriber = nh->subscribe(IMU_PATH, 1000, 
//                         &Camera::callback_number_2, this);
//                 }
//                 void callback_number(const sensor_msgs::Image& msg) {
//                     camera_time = msg.header.stamp.sec + msg.header.stamp.nsec*1e-9;
                    
//                 }
//                 void callback_number_2(const sensor_msgs::Imu& msg_2) {
//                     imu_time = msg_2.header.stamp.sec + msg_2.header.stamp.nsec*1e-9;
//                     ROS_INFO("The time difference between camera-imu is : [%f] \n", imu_time - camera_time);
//                 }
//                 float getFx(){
//                     return fx;
//                 }
//         };
//         Camera* camera_left;
//         Camera* camera_right;
        
//         Zed_Camera(ros::NodeHandle *nh) {
//             camera_left = new Camera(nh);
//             camera_right = new Camera(nh);
//         }
        
// };

Zed_Camera::Zed_Camera(ros::NodeHandle *nh)
{
    camera_left = new Camera(nh);
    camera_right = new Camera(nh);
    nh->getParam("/Camera/width",m_width);
    nh->getParam("/Camera/height",m_height);
    nh->getParam("/Camera/fps",m_fps);
    nh->getParam("/Camera/bl",m_baseline);
    nh->getParam("/Camera_l/fx",camera_left->fx);
    nh->getParam("/Camera_l/fy",camera_left->fy);
    nh->getParam("/Camera_l/cx",camera_left->cx);
    nh->getParam("/Camera_l/cy",camera_left->cy);
    nh->getParam("/Camera_r/fx",camera_right->fx);
    nh->getParam("/Camera_r/fy",camera_right->fy);
    nh->getParam("/Camera_r/cx",camera_right->cx);
    nh->getParam("/Camera_r/cy",camera_right->cy);
}

Zed_Camera::~Zed_Camera()
{

}

void Zed_Camera::GetResolution()
{
    ROS_INFO("Height : [%d], Width : [%d]", m_height, m_width);
}

Zed_Camera::Camera::Camera(ros::NodeHandle *nh)
{
    counter = 0;
    camera_time = 0;
    imu_time = 0;
    camera_subscriber = nh->subscribe(CAMERA_PATH_2, 1000, 
        &Camera::callback_number, this);
    imu_subscriber = nh->subscribe(IMU_PATH, 1000, 
        &Camera::callback_number_2, this);
}

void Zed_Camera::Camera::callback_number(const sensor_msgs::Image& msg) {
            camera_time = msg.header.stamp.sec + msg.header.stamp.nsec*1e-9;
}

void Zed_Camera::Camera::callback_number_2(const sensor_msgs::Imu& msg_2)
{
    imu_time = msg_2.header.stamp.sec + msg_2.header.stamp.nsec*1e-9;
    // ROS_INFO("The time difference between camera-imu is : [%f] \n", imu_time - camera_time);
}

float Zed_Camera::Camera::GetFx()
{
    return fx;
}

void Zed_Camera::Camera::GetIntrinsicValues()
{
    ROS_INFO("\n fx : [%f] \n fy : [%f] \n cx : [%f] \n cy : [%f] \n", fx, fy, cx, cy);
}

} //namespace vio_slam

// int main (int argc, char **argv)
// {
//     ros::init(argc, argv, "Camera");
//     ros::NodeHandle nh;
//     vio_slam::Zed_Camera zedcamera(&nh);
//     zedcamera.camera_left;
//     // std::cout << "xd" << zedcamera.camera_left->getFx() << std::endl;
//     // Zed_Camera::Camera_2 camera_right = Zed_Camera::Camera_2(&nh);
//     // Zed_Camera::Camera_2 camera_rightfx = Zed_Camera::Camera2::getFx();
//     ros::spin();
// }

