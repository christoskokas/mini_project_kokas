#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>

#define CAMERA_PATH "/camera_3/right/image_raw"

class NumberCounter {
    private:
        int counter;
        ros::Publisher pub;
        ros::Subscriber number_subscriber;
        ros::ServiceServer reset_service;
    public:
        NumberCounter(ros::NodeHandle *nh) {
            counter = 0;
            number_subscriber = nh->subscribe(CAMERA_PATH, 1000, 
                &NumberCounter::callback_number, this);
        }
        void callback_number(const std_msgs::Int64& msg) {
            counter += 1;
            ROS_INFO("I heard from camera : [%d] times", msg);
        }
};
int main (int argc, char **argv)
{
    ros::init(argc, argv, "camera_sub");
    ros::NodeHandle nh;
    NumberCounter nc = NumberCounter(&nh);
    ros::spin();
}