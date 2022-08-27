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
#include <opencv2/video/tracking.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/calib3d.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
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

class PointsWithIndexes
{
    private:

    public:
        std::vector < cv::Point2f > points;
        std::vector < bool > removed;
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
        std::vector<bool> close;
        std::vector<bool> statusOfKeys;
        std::vector< cv::KeyPoint > keypoints;
        std::vector< cv::KeyPoint > keypointsLR;
        std::vector< pcl::PointXYZ > pointsPosition;
        std::vector < cv::Point2f> inlierPoints;
        std::vector < int > indicesOfGrids;
        std_msgs::Header header;
        int updateFeatureRemoval(std::vector < int > indexes, std::vector < cv::Point2f >& points, int& count, bool& first);
        int findMinimumResponse(std::vector < int > indexes);
        void removeClosestNeighbors(std::vector < bool >& removed);
        void sortFeaturesKdTree();
        void removeOutliersOpticalFlow(std::vector < cv::Point2f>& pointL, std::vector < cv::Point2f>& pointpL, cv::Mat status);
        std::vector < cv::Point2f> opticalFlow(Features& prevImage, image_transport::Publisher& mImageMatches, bool left);
        std::vector< cv::KeyPoint > featuresAdaptiveThreshold(cv::Mat& patch, int step, unsigned int iterations);
        void findFeatures(bool LR);
        cv::Mat gridBasedFeatures(cv::Mat croppedImage, const int grid[2], cv::Size imgSize);
        void getFeatures(int rows, int cols,image_transport::Publisher& mImageMatches, bool left);
        void getDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        std::vector<cv::DMatch> getMatches(Features& secondImage, image_transport::Publisher& mImageMatches, std::vector<cv::KeyPoint>& previousleftKeypoints, bool LR);
        void findFeaturesTrial();
        void clearFeatures();
        void findORBFeatures(cv::Mat& image, std::vector< cv::KeyPoint >& keypoints, int numbOfFeatures, int edgeThreshold, int fastThreshold);
        void setImage(const sensor_msgs::ImageConstPtr& imageRef);
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
        cv::Mat rmap[2][2];
        cv::Mat previouspoints3D;
        cv:: Mat R1, R2, P1, P2, Q;
        FeatureStrategy mFeatureMatchStrat;
        const Zed_Camera* zedcamera;
        bool firstImage {true};
        ros::Time prevTime;
    public:
        Features leftImage;
        int count = 0;
        Features rightImage;
        Features previousLeftImage;
        Features previousRightImage;
        std::vector<cv::DMatch> previousMatches;
        std::vector<cv::KeyPoint> previousleftKeypoints;
        double camera[6];
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d previousT = Eigen::Matrix4d::Identity();
        std::vector<cv::DMatch> matchWithGridsUsingMask(Features& firstImage, Features& secondImage, int row, int col, int rows, int cols, bool LR);
        std::vector<cv::DMatch> matchEachGrid(Features& firstImage, Features& secondImage, int row, int col, int rows, int cols, bool LR);
        std::vector<cv::DMatch> matchesWithGrids(Features& firstImage, Features& secondImage, int rows, int cols, bool LR);
        std::vector<cv::DMatch> matchesLR(Features& leftImage, Features& rightImage);
        std::vector<cv::DMatch> findMatches(Features& firstImage, Features& secondImage, bool LR);
        std::vector< cv::DMatch > knnMatcher(const Features& firstImage, const Features& secondImage, const bool LR);
        std::vector< cv::DMatch > removeOutliersStereoMatch(const std::vector< cv::DMatch >& matches, const Features& leftImage, const Features& rightImage);
        std::vector< cv::DMatch > removeOutliersHomography(const std::vector< cv::DMatch >& matches, const Features& firstImage, const Features& secondImage);
        void positionOfMatchedFeatures(const std::vector<cv::DMatch>& matches, Features& leftImage, const Features& rightImage, const Features& previousLeftImage, const Features& previousRightImage);
        std::vector< cv::DMatch > loweRatioTest(std::vector< std::vector<cv::DMatch> >& knnmatches);
        void drawFeatureMatches(const std::vector<cv::DMatch>& matches, const Features& firstImage, const Features& secondImage);
        void ceresSolver(std::vector<cv::DMatch>& matches, const cv::Mat& points3D, const cv::Mat& prevpoints3D);
        cv::Mat featurePosition(std::vector < cv::Point2f>& pointsL, std::vector < cv::Point2f>& pointsR, std::vector<bool>& left, std::vector<bool>& right);
        FeatureDrawer(ros::NodeHandle *nh, const Zed_Camera* zedptr);
        ~FeatureDrawer();
        void featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);
        void setUndistortMap(ros::NodeHandle *nh);
        cv::Mat calculateFeaturePosition(const std::vector<cv::DMatch>& matches);
        void setPrevious(cv::Mat& points3D);
        void allMatches(const std_msgs::Header& header);
        void publishMovement();
        void matchTrial(const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& LpLmatches, const vio_slam::Features& secondImage);
        void keepMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& LpLmatches, const vio_slam::Features& secondImage, std::vector<cv::KeyPoint> tempKeypoints, const cv::Mat& points3D, const std_msgs::Header& header, bool left);
        void again();
        // void findFeatures(const cv::Mat& imageRef, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, image_transport::Publisher publish);
        // std::vector<cv::DMatch> findMatches(const std_msgs::Header& lIm);


};


}


#endif // VIEWER_H