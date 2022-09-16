#pragma once

#ifndef TRIAL_H
#define TRIAL_H

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
#include <thread>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>


namespace vio_slam
{


class ImageFrame
{
    public:
        cv::Mat image, desc, realImage;
        std::vector < cv::KeyPoint > keypoints;
        std::vector<cv::Point2f> optPoints;
        int rows {5};
        int cols {5};
        float averageDistance {0.0f};
        int totalNumber {1000};
        int numberPerCell {totalNumber/(rows*cols)};
        int numberPerCellFind {2*totalNumber/(rows*cols)};

        void getImage(int frameNumber, const char* whichImage);
        void rectifyImage(cv::Mat& map1, cv::Mat& map2);

        void opticalFlow(ImageFrame& prevImage,cv::Mat& optFlow);
        void opticalFlowRemoveOutliers(std::vector < cv::Point2f>& pointL, std::vector < cv::Point2f>& pointpL, cv::Mat& status);

        void findDisparity(cv::Mat& otherImage, cv::Mat& disparity);

        void pointsToKeyPoints();

        void findFeaturesOnImage(int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2);

        void findFeaturesFAST();
        void findFeaturesFASTAdaptive();
        void findFeaturesORB();
        void findFeaturesORBAdaptive();
        void findFeaturesGoodFeatures();

        void drawFeaturesWithLines(cv::Mat& outImage);

        void clone(const ImageFrame& second);
        
};

class RobustMatcher2 {
 private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Mat image, P1, P2, Q, R1, R2;
    cv::Mat rmap[2][2];
    ImageFrame leftImage, rightImage, prevLeftImage, prevRightImage;
    clock_t start, total;
    const Zed_Camera* zedcamera;
    // pointer to the feature descriptor extractor object
    // cv::Ptr<cv::DescriptorExtractor> extractor;
    float ratio; // max ratio between 1st and 2nd NN
    bool refineF; // if true will refine the F matrix
    bool firstImage{true};
    double distance; // min distance to epipolar
    double confidence; // confidence level (probability)
    int rows {5};
    int cols {5};
    float averageDistance {0.0f};
    int totalNumber {1000};
    int numberPerCell {totalNumber/(rows*cols)};
    int numberPerCellFind {2*totalNumber/(rows*cols)};
 public:
    RobustMatcher2(const Zed_Camera* zedptr) : ratio(0.85f), refineF(false),
    confidence(0.99), distance(3.0) 
    {
        this->zedcamera = zedptr;
        if (!zedcamera->rectified)
        {
            undistortMap();
        }
        // testFeatureMatching();
        detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
        // testImageRectify();
        // testFeatureExtraction();
    }

    void beginTest();

    // void getImage(cv::Mat& image, cv::Mat& realImage, int frameNumber, const char* whichImage);

    // void findFeaturesOnImage(ImageFrame& camera, int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2);

    // void findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesORB(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    // void findFeaturesORBAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    

    


    void matchCrossRatio(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matches, bool LR);
    void symmetryTest(ImageFrame& first, ImageFrame& second, const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches);
    void ratioTest(std::vector<std::vector<cv::DMatch>>& matches);
    void classIdCheck(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesSym, std::vector < cv::DMatch >& matches, bool LR);
    void removeMatchesDistance(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesId, std::vector < cv::DMatch >& matches);

    
    void drawFeatureMatches(const std::vector<cv::DMatch>& matches, const ImageFrame& firstImage, const ImageFrame& secondImage, cv::Mat& outImage);

    void undistortMap();

    float getDistanceOfPoints(ImageFrame& first, ImageFrame& second, const cv::DMatch& match);

    void testImageRectify();
    void testFeatureExtraction();
    void testFeatureMatching();
    void testDisparityWithOpticalFlow();
    void testFeatureMatchingWithOpticalFlow();
    
};

} //namespace vio_slam
#endif // TRIAL_H