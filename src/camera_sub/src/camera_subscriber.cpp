#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <sstream>
#include <string>

#define CAMERA_PATH "/camera_1/right/image_raw"
#define CAMERA_PATH_2 "/camera_2/right/image_raw"

class Camera_1 {
    private:
        int counter,counter_2;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
    public:
        Camera_1(ros::NodeHandle *nh) {
            counter = 0;
            camera_subscriber = nh->subscribe(CAMERA_PATH, 1000, 
                &Camera_1::callback_number, this);
        }
        void callback_number(const sensor_msgs::Image& msg) {
            counter += 1;
            ROS_INFO("I heard from camera_1 : [%d] times \n", counter);
        }
};

class Camera_2 {
    private:
        int counter;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
    public:
        Camera_2(ros::NodeHandle *nh) {
            counter = 0;
            camera_subscriber = nh->subscribe(CAMERA_PATH_2, 1000, 
                &Camera_2::callback_number, this);
        }
        void callback_number(const sensor_msgs::Image& msg) {
            counter += 1;
            ROS_INFO("I heard from camera_2 : [%d] times \n", counter);
            // for (int i = 0; i < msg.height; i++)
            // {
            //     for (int j = 0; j < msg.width; i++)
            //     {
            //         ROS_INFO("[%f]         \n", &msg.data[i,j]);
            //     }
            //     ROS_INFO("\n");
            // }




            // TO DO ::: PUBLISH TO A TOPIC THE IMAGE 







            
            
        }
};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "camera_subscriber");
    ros::NodeHandle nh;
    Camera_1 camera1 = Camera_1(&nh);
    Camera_2 camera2 = Camera_2(&nh);
    ros::spin();
}