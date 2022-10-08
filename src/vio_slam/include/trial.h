#pragma once

#ifndef TRIAL_H
#define TRIAL_H

#include "Camera.h"
#include "FeatureExtractor.h"
#include "Settings.h"
#include "Optimizer.h"
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
#include <boost/assign.hpp>
#include <boost/foreach.hpp>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <thread>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "ceres/ceres.h"
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <math.h> 
#include <future>

// typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

namespace vio_slam
{

bool kitti = false;


struct GoodFeatures
{
    std::vector<cv::Point2f> shiPoints;
    std::vector<int> class_id;
};

class ImageFrame
{
    public:
        // bool kitti = true;
        cv::Mat image, desc, realImage;
        std::vector < cv::KeyPoint > keypoints;
        std::vector<cv::Point2f> optPoints;
        std::vector<int> class_id;

        std_msgs::Header header;
        int rows {10};
        int cols {10};
        float averageDistance {0.0f};
        float averageAngle {0.0f};
        int totalNumber {1000};
        int numberPerCell {totalNumber/(rows*cols)};
        int numberPerCellFind {2*totalNumber/(rows*cols)};

        void setImage(const sensor_msgs::ImageConstPtr& imageRef);
        void getImage(int frameNumber, const char* whichImage);
        void rectifyImage(cv::Mat& map1, cv::Mat& map2);

        void opticalFlow(ImageFrame& prevImage,cv::Mat& status, cv::Mat& optFlow);


        void findDisparity(cv::Mat& otherImage, cv::Mat& disparity);

        void pointsToKeyPoints();
        

        void findFeaturesOnImage(int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2);

        void findFeaturesFAST();
        void findFeaturesFASTAdaptive();
        void findFeaturesORB();
        void findFeaturesORBAdaptive();
        void findFeaturesGoodFeatures();
        void findFeaturesGoodFeaturesGrid();
        void findFeaturesGoodFeaturesWithPast();

        void drawFeaturesWithLines(cv::Mat& outImage);

        void clone(const ImageFrame& second);
        
};

class RobustMatcher2 {
 private:
    // bool kitti = true;
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector;
    std::vector<int> pointsTimes;
    cv::Mat image, P1, P2, Q, R1, R2;
    // cv::Mat R1 = cv::Mat::zeros(cv::Size(3,3),CV_32F);
    // cv::Mat R2 = cv::Mat::zeros(cv::Size(3,3),CV_32F);
    // cv::Mat P1 = cv::Mat::zeros(cv::Size(4,3),CV_32F);
    // cv::Mat P2 = cv::Mat::zeros(cv::Size(4,3),CV_32F);
    // cv::Mat Q = cv::Mat::zeros(cv::Size(4,4),CV_32F);
    // cv::Mat R2(3, 3, CV_32F);
    // cv::Mat P1(3, 4, CV_32F);
    // cv::Mat P2(3, 4, CV_32F);
    // cv::Mat Q(4, 4, CV_32F);
    ImageFrame trialL, trialR;
    cv::Mat rmap[2][2];
    ImageFrame leftImage, rightImage, prevLeftImage, prevRightImage;
    clock_t start, total;
    Zed_Camera* zedcamera;
    // pointer to the feature descriptor extractor object
    // cv::Ptr<cv::DescriptorExtractor> extractor;
    float ratio; // max ratio between 1st and 2nd NN
    bool refineF; // if true will refine the F matrix
    bool firstImage{true};
    double distance; // min distance to epipolar
    double confidence; // confidence level (probability)
    double camera[6] = {0, 1, 2, 0, 0, 0};
    int rows {2};
    int cols {2};
    float averageDistance {0.0f};
    float averageAngle[4] {0.0f};
    int totalNumber {700};
    int numberPerCell {totalNumber/(rows*cols)};
    int numberPerCellFind {2*totalNumber/(rows*cols)};
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d previousT = Eigen::Matrix4d::Identity();


    // image_transport::ImageTransport m_it;
    // message_filters::Subscriber<sensor_msgs::Image> subLeftIm;
    // message_filters::Subscriber<sensor_msgs::Image> subRightIm;
    // message_filters::Synchronizer<MySyncPolicy> img_sync;


    // ros::Publisher posePublisher;
 public:
    RobustMatcher2(Zed_Camera* zedptr) : ratio(0.85f), refineF(false),
    confidence(0.99), distance(3.0) 
    {
        this->zedcamera = zedptr;
        if (!zedcamera->rectified)
        {
            undistortMap();
        }
        else
        {
            calcP1P2();
        }
        // cv::Mat rod;
        // cv::Rodrigues(R1, rod);
        // camera[0] = rod.at<double>(0);
        // camera[1] = rod.at<double>(1);
        // camera[2] = rod.at<double>(2);
        // camera[3] = 0.0;
        // camera[4] = 0.0;
        // camera[5] = 0.0;
        // previousT(1,1) = -1;
        // previousT(2,2) = -1;
        // testFeatureMatching();
        detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
        // testImageRectify();
        // testFeatureExtraction();

        // subLeftIm.subscribe(*nh, zedcamera->cameraLeft.path, 3);
        // subRightIm.subscribe(*nh, zedcamera->cameraRight.path, 3);
        // img_sync.registerCallback(boost::bind(&RobustMatcher2::ImagesCallback, this, _1, _2));

        // std::string position_path;
        // nh->getParam("ground_truth_path", position_path);
        // posePublisher = nh->advertise<nav_msgs::Odometry>(position_path,1);

    }

    void beginTest();

    void ImagesCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);

    void publishPose();

    void predictRightImagePoints(ImageFrame& left, ImageFrame& right);
    void removeLeftRightOutliers(ImageFrame& left, ImageFrame& right, cv::Mat& status);
    void opticalFlowRemoveOutliers(ImageFrame& first, ImageFrame& second, cv::Mat& status, bool LR);

    // void getImage(cv::Mat& image, cv::Mat& realImage, int frameNumber, const char* whichImage);

    // void findFeaturesOnImage(ImageFrame& camera, int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2);

    // void findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesORB(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesORBAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    

    void triangulatePointsOpt(ImageFrame& first, ImageFrame& second, cv::Mat& points3D);
    void triangulatePointsOptWithProjection(ImageFrame& first, ImageFrame& second, cv::Mat& points3D);
    
    void ceresSolver(cv::Mat& points3D, cv::Mat& prevPoints3D);
    void ceresSolverPnp(cv::Mat& points3D, cv::Mat& prevPoints3D);

    void reduceVector(std::vector<cv::Point2f> &v, cv::Mat& status);
    void reduceVectorInt(std::vector<int> &v, cv::Mat& status);
    void matchCrossRatio(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matches, bool LR);
    void symmetryTest(ImageFrame& first, ImageFrame& second, const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches);
    void ratioTest(std::vector<std::vector<cv::DMatch>>& matches);
    void classIdCheck(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesSym, std::vector < cv::DMatch >& matches, bool LR);
    void removeMatchesDistance(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesId, std::vector < cv::DMatch >& matches);

    
    void drawFeatureMatches(const std::vector<cv::DMatch>& matches, const ImageFrame& firstImage, const ImageFrame& secondImage, cv::Mat& outImage);
    void drawOpticalFlow(ImageFrame& prevImage, ImageFrame& curImage, cv::Mat& outImage);

    void undistortMap();
    void calcP1P2();

    float getAngleOfPoints(cv::Point2f& first, cv::Point2f& second);
    float getDistanceOfPoints(ImageFrame& first, ImageFrame& second, const cv::DMatch& match);
    float getDistanceOfPointsOptical(cv::Point2f& first, cv::Point2f& second);
    void findAverageDistanceOfPoints(ImageFrame& first, ImageFrame& second);


    void testImageRectify();
    void testFeatureExtraction();
    void testFeatureMatching();
    void testDisparityWithOpticalFlow();
    void testFeatureMatchingWithOpticalFlow();
    void testOpticalFlowWithPairs();
    void testFeatureExtractorClassWithCallback();
    void testFeatureExtractorClass();
    
};

} //namespace vio_slam
#endif // TRIAL_H