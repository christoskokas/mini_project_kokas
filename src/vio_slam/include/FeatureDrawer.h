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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

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
        image_transport::Publisher mImageMatches;
        cv::Mat leftImage;
        cv::Mat rightImage;
        cv::Mat leftDescript;
        cv::Mat rightDescript;
        cv:: Mat R1, R2, P1, P2, Q;
        std::vector< float > leftCameraMatrix = {};
        std::vector< float > rightCameraMatrix = {};
        std::vector< float > distLeft = {};
        std::vector< float > distRight = {};
        std::vector< float > sensorsRotate = {};
        std::vector< float > sensorsTranslate = {};
        std::vector<cv::KeyPoint> leftKeypoints;
        std::vector<cv::KeyPoint> rightKeypoints;
        std::string mLeftCameraPath;
        std::string mRightCameraPath;
        FeatureStrategy mFeatureMatchStrat;

        int width {}, height {};
    public:
        message_filters::Subscriber<sensor_msgs::Image> leftIm;
        message_filters::Subscriber<sensor_msgs::Image> rightIm;
        message_filters::Synchronizer<MySyncPolicy> img_sync;
        FeatureDrawer(ros::NodeHandle *nh, FeatureStrategy& featureMatchStrat);
        ~FeatureDrawer();
        void leftImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void rightImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void addFeatures();
        void FeatureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);
        void featureMatch();
        void findFeatures(const sensor_msgs::ImageConstPtr& imageRef, cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, image_transport::Publisher publish);
        std::vector<cv::DMatch> findMatches(const sensor_msgs::ImageConstPtr& lIm);
        void getCameraMatrix(ros::NodeHandle *nh);


};


}


#endif // VIEWER_H