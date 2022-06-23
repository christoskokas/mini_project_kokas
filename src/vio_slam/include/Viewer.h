#pragma once

#ifndef VIEWER_H
#define VIEWER_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace vio_slam
{

class FeatureDrawer
{
    private:
        image_transport::ImageTransport m_it;
        image_transport::Subscriber m_image_sub;
        image_transport::Publisher m_image_pub;
        std::string m_camera_path;
    public:
        FeatureDrawer(ros::NodeHandle *nh);
        ~FeatureDrawer();
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);
        void addFeatures();

};


}


#endif // VIEWER_H