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

namespace vio_slam
{


class ImageFrame
{
    public:
        cv::Mat image;
};

class RobustMatcher {
 private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Mat image, P1, P2, Q, R1, R2;
    cv::Mat rmap[2][2];
    ImageFrame leftImage, rightImage;
    clock_t start, total;
    const Zed_Camera* zedcamera;
    // pointer to the feature descriptor extractor object
    // cv::Ptr<cv::DescriptorExtractor> extractor;
    float ratio; // max ratio between 1st and 2nd NN
    bool refineF; // if true will refine the F matrix
    double distance; // min distance to epipolar
    double confidence; // confidence level (probability)
    int rows {5};
    int cols {5};
    int totalNumber {2000};
    int numberPerCell {totalNumber/(rows*cols)};
    int numberPerCellFind {2*totalNumber/(rows*cols)};
 public:
    RobustMatcher(const Zed_Camera* zedptr) : ratio(0.85f), refineF(false),
    confidence(0.99), distance(3.0) 
    {
        this->zedcamera = zedptr;
        if (!zedcamera->rectified)
        {
            undistortMap();
        }
        testFeatureMatching();
        
        // testImageRectify();
        // testFeatureExtraction();
    }

    void getImage(cv::Mat& image, int frameNumber, const char* whichImage);

    void findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesORB(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesORBAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    

    void drawFeaturesWithLines(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& outImage);


    void rectifyImage(cv::Mat& image, cv::Mat& map1, cv::Mat& map2);
    void undistortMap();


    void testImageRectify();
    void testFeatureExtraction();
    void testFeatureMatching();
};

} //namespace vio_slam
#endif // TRIAL_H