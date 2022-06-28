#pragma once

#ifndef VIEWER_H
#define VIEWER_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"

namespace vio_slam
{



class FeatureDrawer
{
    private:
        image_transport::ImageTransport m_it;
        image_transport::Subscriber mLeftImageSub;
        image_transport::Subscriber mRightImageSub;
        image_transport::Publisher mLeftImagePub;
        image_transport::Publisher mRightImagePub;
        std::vector<uchar> leftImage;
        std::vector<uchar> rightImage;
        std::string mLeftCameraPath;
        std::string mRightCameraPath;
        std::string mFeatureMatchStrat;
    public:
        FeatureDrawer(ros::NodeHandle *nh, std::string* featureMatchStrat);
        ~FeatureDrawer();
        void leftImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void rightImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void addFeatures();


};


}


#endif // VIEWER_H