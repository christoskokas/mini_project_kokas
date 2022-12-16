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
#include <numeric>

namespace vio_slam
{

struct TrackedKeys
{
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<cv::KeyPoint> rightKeyPoints;
    std::vector<cv::KeyPoint> predKeyPoints;
    std::vector<int> rightIdxs;
    std::vector<int> mapPointIdx;
    std::vector<int> matchedIdxs;
    std::vector<float> estimatedDepth;
    std::vector<float> angles;
    // std::vector<float> estimatedDepthNew;
    std::vector<bool> close;
    std::vector<bool> hasPrediction;
    std::vector<bool> inliers;
    std::vector<uchar> inliers2;
    std::vector<int> trackCnt;
    cv::Mat Desc, rightDesc;

    template <typename T>
    void reduce(std::vector<T>& check)
    {
        reduceVectorTemp<cv::KeyPoint,T>(keyPoints,check);
        reduceVectorTemp<cv::KeyPoint,T>(predKeyPoints,check);
        reduceVectorTemp<float,T>(estimatedDepth,check);
        reduceVectorTemp<int,T>(mapPointIdx,check);
        reduceVectorTemp<int,T>(trackCnt,check);
        reduceVectorTemp<bool,T>(close,check);
    }

    void add(cv::KeyPoint& kp, int mapIdx, float _depth, bool _close, int _trackCnt)
    {
        keyPoints.emplace_back(kp);
        mapPointIdx.emplace_back(mapIdx);
        estimatedDepth.emplace_back(_depth);
        close.emplace_back(_close);
        trackCnt.emplace_back(_trackCnt);
    }

    void reserve(size_t size)
    {
        keyPoints.reserve(size);
        predKeyPoints.reserve(size);
        rightIdxs.reserve(size);
        mapPointIdx.reserve(size);
        estimatedDepth.reserve(size);
        close.reserve(size);
        trackCnt.reserve(size);
    }

    void clear()
    {
        keyPoints.clear();
        predKeyPoints.clear();
        rightIdxs.clear();
        mapPointIdx.clear();
        estimatedDepth.clear();
        close.clear();
        trackCnt.clear();
    }

    void resize(size_t size)
    {
        estimatedDepth.resize(size, -1.0f);
        mapPointIdx.resize(size, -1);
        close.resize(size, false);
        trackCnt.resize(size, 0);
    }

    void clone(TrackedKeys& keysToClone)
    {
        keyPoints = keysToClone.keyPoints;
        rightKeyPoints = keysToClone.rightKeyPoints;
        Desc = keysToClone.Desc.clone();
        rightDesc = keysToClone.rightDesc.clone();
        mapPointIdx = keysToClone.mapPointIdx;
        estimatedDepth = keysToClone.estimatedDepth;
        close = keysToClone.close;
        trackCnt = keysToClone.trackCnt;
    }

    void getKeys(const TrackedKeys& keysToClone)
    {
        keyPoints = keysToClone.keyPoints;
        rightKeyPoints = keysToClone.rightKeyPoints;
        rightIdxs = keysToClone.rightIdxs;
        angles = keysToClone.angles;
        Desc = keysToClone.Desc.clone();
        rightDesc = keysToClone.rightDesc.clone();
        close = keysToClone.close;
        estimatedDepth = keysToClone.estimatedDepth;
    }
};

struct PointsWD
{
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2f> predLeft;
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point3d> prevPoints3D;
    std::vector<double> depth;

    template <typename T>
    void reduce(std::vector<T>& check)
    {
        reduceVectorTemp<cv::Point3d,T>(points3D,check);
        reduceVectorTemp<cv::Point3d,T>(prevPoints3D,check);
        reduceVectorTemp<cv::Point2f,T>(left,check);
        reduceVectorTemp<cv::Point2f,T>(predLeft,check);
        reduceVectorTemp<double,T>(depth,check);
    }
};

struct SubPixelPoints
{
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point3d> points3DCurr;
    std::vector<cv::Point3d> points3DStereo;
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2f> newPnts;
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
        reduceVectorTemp<cv::Point3d,T>(points3DCurr,check);
        reduceVectorTemp<cv::Point2d,T>(points2D,check);
        reduceVectorTemp<cv::Point2f,T>(left,check);
        reduceVectorTemp<cv::Point2f,T>(right,check);
        reduceVectorTemp<cv::Point2f,T>(newPnts,check);
        reduceVectorTemp<float,T>(depth,check);
        reduceVectorTemp<bool,T>(useable,check);
    }

    template <typename T>
    void reduceWithInliers(std::vector<T>& check)
    {
        reduceVectorInliersTemp<cv::Point3d,T>(points3D,check);
        reduceVectorInliersTemp<cv::Point3d,T>(points3DCurr,check);
        reduceVectorInliersTemp<cv::Point2f,T>(left,check);
        reduceVectorInliersTemp<cv::Point2f,T>(newPnts,check);
        // reduceVectorInliersTemp<float,T>(depth,check);
        // reduceVectorInliersTemp<bool,T>(useable,check);
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
    const int mnContr {100};

#if KITTI_DATASET
    const int gridRows {40};
    const int gridCols {40};
#elif ZED_DATASET
    const int gridRows {30};
    const int gridCols {30};
#else
    const int gridRows {20};
    const int gridCols {20};
#endif

    const int gridsNumber {gridCols * gridRows};

    std::vector<cv::Point> pattern;
    
    std::vector<int> umax;

    cv::Ptr<cv::ORB> detector;


    public:

        int maskRadius {5};
    
        std::vector <cv::Mat> imagePyramid;
        std::vector<int>scaledPatchSize;
        std::vector < float > scalePyramid;
        std::vector < float > scaleInvPyramid;
        std::vector < int > featurePerLevel;
        std::vector<std::vector<int>> KeyDestrib;
        // std::vector<std::vector<int>> KeyDestribRight;
        
        const int numberPerCell = nFeatures/(gridRows * gridCols);


        enum FeatureChoice
        {
            ORB,
            FAST
        };

        FeatureChoice choice;
        
        FeatureExtractor(const int _nfeatures = 2000, const int _nLevels = 8, const float _imScale = 1.2f, const int _edgeThreshold = 20, const int _patchSize = 31, const int _maxFastThreshold = 20, const int _minFastThreshold = 7, const bool _nonMaxSuppression = true);
        
        int getGridRows();
        int getGridCols();

        std::vector<cv::KeyPoint> ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints,
                         float tolerance, int cols, int rows);

        void extractKeysNew(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

        void computeKeypointsOld(cv::Mat& image, std::vector <cv::KeyPoint>& keypoints, cv::Mat& desc, const bool right);
        void computeKeypointsOld2(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point2f>& pnts, cv::Mat& descriptors, const bool right);
        void computeKeypoints(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point2f>& pnts, cv::Mat& descriptors, const bool right);
        void computeKeypointsORBNew(cv::Mat& image, std::vector<std::vector<cv::KeyPoint>>& allKeys);
        void computeKeypointsORBNewRight(cv::Mat& image, std::vector<std::vector<cv::KeyPoint>>& allKeys);
        void computeKeypointsFAST(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
        void computeKeypointsORB(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
        void computeDescriptorsFAST(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        void computeFASTandDesc(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point2f>& pnts, cv::Mat& descriptors);
        void computeORBandDesc(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        void computeORBandDescNoGrids(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

        void findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);
        void findORBWithCV(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findORBGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);

        void populateKeyDestrib(const std::vector<cv::Point2f>& pnts, std::vector<std::vector<int>>& keyDestrib);
        void populateKeyDestribFAST(const cv::Mat& image, const std::vector<cv::Point2f>& pnts, std::vector<std::vector<int>>& keyDestribution);

        void computeKeypointsORBLeft(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints);
        void computeKeypointsFASTLeft(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints);
        void setMask(const std::vector<cv::KeyPoint>& prevKeys, cv::Mat& mask);
        void extractLeftFeaturesORB(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& desc, TrackedKeys& prevKeys);
        void extractLeftFeaturesORBNoGrids(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& desc, TrackedKeys& prevKeys);
        void extractLeftFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& desc, TrackedKeys& prevKeys);
        void extractFeatures(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractFeaturesClose(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractFeaturesCloseMask(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const cv::Mat& mask);
        void extractFeaturesMask(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& desc);
        void extractFeaturesPop(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const std::vector<int>& pop);
        void extractORB(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);
        void extractORBGrids(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints);

        // void updatePoints(std::vector <cv::KeyPoint>& leftKeys, std::vector <cv::KeyPoint>& rightKeys, SubPixelPoints& points);


        void findFAST(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc);
        void findFASTGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFASTGridsClose(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFASTGridsCloseMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask);
        void findFASTGridsMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask);
        void findFASTGridsPop(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const std::vector<int>& pop);

        void computePyramid(const cv::Mat& image);
        float computeOrientation(const cv::Mat& image, const cv::Point2f& point);
        // static void computeOrbDescriptor(const cv::KeyPoint& kpt,const cv::Mat& img, const cv::Point* pattern, uchar* desc);
    
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