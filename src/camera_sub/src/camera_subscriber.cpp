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

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class Camera_1 {
    private:
        int counter,counter_2;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
    public:
        Camera_1(ros::NodeHandle *nh) {
            counter = 0;
            pub = nh->advertise<sensor_msgs::Image>("/right_image", 10); 
            camera_subscriber = nh->subscribe(CAMERA_PATH, 1000, 
                &Camera_1::callback_number, this);
        }
        void callback_number(const sensor_msgs::Image& msg) {
            counter += 1;
            if (counter % 50 == 0)
            {
                ROS_INFO("I heard from camera_1 : [%d] times \n", counter);
            }
            pub.publish(msg);
        }
};

class Camera_2 {
    private:
        int counter;
        float camera_time;
        float imu_time;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
        ros::Subscriber imu_subscriber;
    public:
        Camera_2(ros::NodeHandle *nh) {
            counter = 0;
            camera_time = 0;
            imu_time = 0;
            camera_subscriber = nh->subscribe(CAMERA_PATH_2, 1000, 
                &Camera_2::callback_number, this);
            imu_subscriber = nh->subscribe(IMU_PATH, 1000, 
                &Camera_2::callback_number_2, this);
        }
        void callback_number(const sensor_msgs::Image& msg) {
            camera_time = msg.header.stamp.sec + msg.header.stamp.nsec*1e-9;
            
        }
        void callback_number_2(const sensor_msgs::Imu& msg_2) {
            imu_time = msg_2.header.stamp.sec + msg_2.header.stamp.nsec*1e-9;
            ROS_INFO("The time difference between camera-imu is : [%f] \n", imu_time - camera_time);
        }
};

class PointCloud_1 {
        private:
        int counter;
        ros::Publisher pub;
        ros::Subscriber camera_subscriber;
    public:
        PointCloud_1(ros::NodeHandle *nh) {
            counter = 0;
            camera_subscriber = nh->subscribe(POINTCLOUD_PATH, 1, 
                &PointCloud_1::callback_number, this);
        }
        void callback_number(const PointCloud::ConstPtr& msg) {
            printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
            BOOST_FOREACH (const pcl::PointXYZ& pt, msg->points)
            if (!isnan(pt.x))
            {
                printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
            }
            
            
        }
};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "camera_subscriber");
    ros::NodeHandle nh;
    Camera_2 camera2 = Camera_2(&nh);
    ros::spin();
}