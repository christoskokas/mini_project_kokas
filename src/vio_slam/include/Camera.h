#ifndef CAMERA_H
#define CAMERA_H

#include <ros/ros.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

namespace vio_slam
{
    class Zed_Camera
    {
        private:
            float m_baseline;
        public:
            class Camera
            {
                private:
                    int counter;
                    float camera_time;
                    float imu_time;
                    float fx = 260.04;
                    ros::Publisher pub;
                    ros::Subscriber camera_subscriber;
                    ros::Subscriber imu_subscriber;
                public:
                    Camera(ros::NodeHandle *nh);
                    ~Camera();
                    void callback_number(const sensor_msgs::Image& msg);
                    void callback_number_2(const sensor_msgs::Imu& msg_2);
                    float GetFx();
            };
            Camera* camera_left;
            Camera* camera_right;
            Zed_Camera(ros::NodeHandle *nh);
            ~Zed_Camera();

    };
}

#endif // CAMERA_H