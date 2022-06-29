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
#include "opencv2/core.hpp"

namespace vio_slam
{

/**
 * @brief Feature Detection Strategy
 * Choises : orb, fast, brisk.
 * 
 * brisk is not recommended because it is too slow for real time applications
 * 
 */
enum class FeatureStrategy
{
    orb,
    fast,
    brisk,
};

/**
 * @brief 
 * 
 */

class FeatureDrawer
{
    private:
        image_transport::ImageTransport m_it;
        image_transport::Subscriber mLeftImageSub;
        image_transport::Subscriber mRightImageSub;
        image_transport::Publisher mLeftImagePub;
        image_transport::Publisher mRightImagePub;
        cv::Mat leftImage;
        cv::Mat rightImage;
        cv::Mat leftDescript;
        cv::Mat rightDescript;
        std::vector<cv::KeyPoint> leftKeypoints;
        std::vector<cv::KeyPoint> rightKeypoints;
        std::string mLeftCameraPath;
        std::string mRightCameraPath;
        FeatureStrategy mFeatureMatchStrat;
    public:
        FeatureDrawer(ros::NodeHandle *nh, FeatureStrategy& featureMatchStrat);
        ~FeatureDrawer();
        void leftImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void rightImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void addFeatures();
        void FeatureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);


};


}


#endif // VIEWER_H