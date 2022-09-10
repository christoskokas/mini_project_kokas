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


class RobustMatcher {
 private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Mat image;
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
    RobustMatcher() : ratio(0.85f), refineF(false),
    confidence(0.99), distance(3.0) 
    {
        std::string imagePath = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/city.jpg";
        image = cv::imread(imagePath,cv::IMREAD_COLOR);
        assert(!image.empty() && "Could not read the image");
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> keypoints;
        clock_t fastStart = clock();
        findFeatures(image, keypoints);
        clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
        cv::Mat fastImage;
        drawFeaturesWithLines(image, keypoints,fastImage);
        std::cout << "fast size : " << keypoints.size() << '\n';
        keypoints.clear();
        cv::imshow("fast features", fastImage);
        clock_t fastGridStart = clock();
        findFeaturesAdaptive(image, keypoints);
        clock_t fastGridTotalTime = double(clock() - fastGridStart) * 1000 / (double)CLOCKS_PER_SEC;
        cv::Mat fastAdaptiveImage;
        drawFeaturesWithLines(image, keypoints,fastAdaptiveImage);
        std::cout << "fast grid size : " << keypoints.size() << '\n';
        keypoints.clear();
        cv::imshow("fast features GRID", fastAdaptiveImage);
        clock_t ORBStart = clock();
        findFeaturesORB(image, keypoints);
        clock_t ORBTotalTime = double(clock() - ORBStart) * 1000 / (double)CLOCKS_PER_SEC;
        cv::Mat ORBImage;
        drawFeaturesWithLines(image, keypoints,ORBImage);
        std::cout << "ORB size : " << keypoints.size() << '\n';
        keypoints.clear();
        cv::imshow("ORB features", ORBImage);
        clock_t ORBGridStart = clock();
        findFeaturesORBAdaptive(image, keypoints);
        clock_t ORBGridTotalTime = double(clock() - ORBGridStart) * 1000 / (double)CLOCKS_PER_SEC;
        cv::Mat ORBAdaptiveImage;
        drawFeaturesWithLines(image, keypoints,ORBAdaptiveImage);
        std::cout << "ORB grid size : " << keypoints.size() << '\n';
        keypoints.clear();
        cv::imshow("ORB features GRID", ORBAdaptiveImage);
        std::cout << "\nFast Features Time      : " << fastTotalTime        << " milliseconds." << '\n';
        std::cout << "-------------------------\n";
        std::cout << "\nFast Grid Features Time : " << fastGridTotalTime    << " milliseconds." << '\n';
        std::cout << "-------------------------\n";
        std::cout << "\nORB Features Time       : " << ORBTotalTime         << " milliseconds." << '\n';
        std::cout << "-------------------------\n";
        std::cout << "\nORB Grid Features Time  : " << ORBGridTotalTime     << " milliseconds." << '\n';
        std::cout << "-------------------------\n";
        cv::waitKey(0);
        // detector = cv::ORB::create();
        // extractor = cv::ORB::create();
    }
    void drawFeaturesWithLines(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& outImage);
    void findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesORB(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    void findFeaturesORBAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    cv::Mat match(cv::Mat& image1,cv::Mat& image2, std::vector<cv::DMatch>& matches,std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2);
    cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,const std::vector<cv::KeyPoint>& keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& outMatches);
    void symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches);
    int ratioTest(std::vector<std::vector<cv::DMatch>>& matches);
};

#endif // TRIAL_H