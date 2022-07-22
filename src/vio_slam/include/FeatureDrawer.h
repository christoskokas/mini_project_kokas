#pragma once

#ifndef VIEWER_H
#define VIEWER_H

#include <Camera.h>

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
#include <opencv2/calib3d.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/foreach.hpp>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>

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

class Features
{
    private:

    public:
        cv::Mat image;
        cv::Mat descriptors;
        std::vector< cv::KeyPoint > keypoints;
        std::vector< pcl::PointXYZ > pointsPosition;
        void findFeatures();
        std::vector<cv::DMatch> findMatches(Features& secondImage, const std_msgs::Header& lIm, image_transport::Publisher& mImageMatches, bool LR);
};

class FeatureDrawer
{
    private:
        image_transport::ImageTransport m_it;
        image_transport::Publisher mImageMatches;
        message_filters::Subscriber<sensor_msgs::Image> leftIm;
        message_filters::Subscriber<sensor_msgs::Image> rightIm;
        message_filters::Synchronizer<MySyncPolicy> img_sync;
        ros::Publisher pose_pub;
        float sums[3] {};
        float sumsMovement[3] {};
        float previousSums[3] {};
        float previoussumsMovement[3] {};
        cv::Mat rmap[2][2];
        cv::Mat previouspoints3D;
        cv:: Mat R1, R2, P1, P2, Q;
        FeatureStrategy mFeatureMatchStrat;
        const Zed_Camera* zedcamera;
        bool firstImage {true};
        ros::Time prevTime;
    public:
        Features leftImage;
        Features rightImage;
        Features previousLeftImage;
        Features previousRightImage;
        double camera[6];
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d previousT = Eigen::Matrix4d::Identity();
        FeatureDrawer(ros::NodeHandle *nh, const Zed_Camera* zedptr);
        ~FeatureDrawer();
        void featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);
        void setUndistortMap(ros::NodeHandle *nh);
        cv::Mat setImage(const sensor_msgs::ImageConstPtr& imageRef);
        cv::Mat calculateFeaturePosition(const std::vector<cv::DMatch>& matches);
        void setPrevious(std::vector<cv::DMatch> matches, cv::Mat points3D);
        void allMatches(const std_msgs::Header& header);
        void clearFeaturePosition();
        void publishMovement(const std_msgs::Header& header);
        void printMat(cv::Mat matrix);
        void keepMatches(std::vector<cv::DMatch> matches, std::vector<cv::DMatch> matches2, bool left);
        // void findFeatures(const cv::Mat& imageRef, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, image_transport::Publisher publish);
        // std::vector<cv::DMatch> findMatches(const std_msgs::Header& lIm);


};


}


#endif // VIEWER_H