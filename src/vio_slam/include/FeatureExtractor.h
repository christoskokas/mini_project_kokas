#pragma once

#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include "Settings.h"
#include <ros/ros.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <future>

namespace vio_slam
{


struct SubPixelPoints
{
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2d> points2D;
    std::vector<cv::Point2f> right;
    std::vector<int> indexes3D;
    std::vector<int> indexes2D;
    std::vector<float> depth;
    std::vector<bool> useable;

    void clone(SubPixelPoints& points);
    void clear();
    void add(SubPixelPoints& points);
    void addLeft(SubPixelPoints& points);

    template <typename T>
    void reduce(std::vector<T>& check)
    {
        reduceVectorTemp<cv::Point3d,T>(points3D,check);
        reduceVectorTemp<cv::Point2d,T>(points2D,check);
        reduceVectorTemp<cv::Point2f,T>(left,check);
        reduceVectorTemp<cv::Point2f,T>(right,check);
        reduceVectorTemp<float,T>(depth,check);
        reduceVectorTemp<bool,T>(useable,check);
    }

    template <typename T>
    void reduceWithInliers(std::vector<T>& check)
    {
        reduceVectorInliersTemp<cv::Point3d,T>(points3D,check);
        reduceVectorInliersTemp<cv::Point2f,T>(left,check);
        reduceVectorInliersTemp<float,T>(depth,check);
        reduceVectorInliersTemp<bool,T>(useable,check);
    }

    template <typename T>
    void reduceWithValue(std::vector<T>& check, const float value)
    {
        reduceVectorWithValue<cv::Point3d,T>(points3D,check, value);
        reduceVectorWithValue<cv::Point2d,T>(points2D,check, value);
        reduceVectorWithValue<cv::Point2f,T>(left,check, value);
        reduceVectorWithValue<cv::Point2f,T>(right,check, value);
        reduceVectorWithValue<float,T>(depth,check, value);
        reduceVectorWithValue<bool,T>(useable,check, value);
    }

};

struct StereoKeypoints
{
    std::vector <cv::KeyPoint> left;
    std::vector <cv::KeyPoint> right;
};

struct StereoDescriptors
{
    cv::Mat left;
    cv::Mat right;
};

class FeatureExtractor
{

    

    const int nFeatures;
    const int nLevels;
    const float imScale;
    const int edgeThreshold;
    const int patchSize;
    const int halfPatchSize {15};
    const int maxFastThreshold;
    const int minFastThreshold;
    const bool nonMaxSuppression;

#if KITTI_DATASET
    const int gridRows {30};
    const int gridCols {30};
#elif ZED_DATASET
    const int gridRows {10};
    const int gridCols {10};
#else
    const int gridRows {5};
    const int gridCols {5};
#endif

    const int gridsNumber {gridCols * gridRows};

    std::vector <cv::Mat> imagePyramid;
    std::vector < float > scalePyramid;
    std::vector < float > scaleInvPyramid;



    public:
        
        const int numberPerCell = nFeatures/(gridRows * gridCols);
        enum FeatureChoice
        {
            ORB,
            FAST
        };

        FeatureChoice choice;
        
        FeatureExtractor(const int _nfeatures = 2000, const int _nLevels = 5, const float _imScale = 1.3f, const int _edgeThreshold = 10, const int _patchSize = 31, const int _maxFastThreshold = 20, const int _minFastThreshold = 7, const bool _nonMaxSuppression = true);
        
        int getGridRows();
        int getGridCols();

        void findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);
        void findORBWithCV(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findORBGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);

        void extractFeatures(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractFeaturesClose(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractFeaturesMask(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const cv::Mat& mask);
        void extractFeaturesPop(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const std::vector<int>& pop);
        void extractORB(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractORBGrids(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);

        // void updatePoints(std::vector <cv::KeyPoint>& leftKeys, std::vector <cv::KeyPoint>& rightKeys, SubPixelPoints& points);


        void findFAST(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);
        void findFASTGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFASTGridsClose(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFASTGridsMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask);
        void findFASTGridsPop(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const std::vector<int>& pop);

        void computePyramid(cv::Mat& image);
        float computeOrientation(const cv::Mat& image, const cv::Point2f& point);
    
        void separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void separateImageSubPixel(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void getNonMaxSuppression(std::vector < cv::KeyPoint >& prevImageKeys, cv::KeyPoint& it);
        bool checkDistance(cv::KeyPoint& first, cv::KeyPoint& second, int distance);

        void highSpeedTest(const uchar* rowPtr, int pixels[25], const int fastThresh);
        void getPixelOffset(int pixels[25], int rowStride);
        int checkIntensities(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int thresh);
        float computeScore(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int fastThresh);


};

} // namespace vio_slam

#endif // FEATUREEXTRACTOR_H