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
    std::vector<std::vector<std::vector<int>>> lkeyGrid;
    std::vector<std::vector<std::vector<int>>> rkeyGrid;
    std::vector<int> rightIdxs;
    std::vector<int> leftIdxs;
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
    float medianDepth;
    float xMult, yMult;
    int xGrids, yGrids;


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
        lkeyGrid = keysToClone.lkeyGrid;
        rkeyGrid = keysToClone.rkeyGrid;
        rightIdxs = keysToClone.rightIdxs;
        leftIdxs = keysToClone.leftIdxs;
        angles = keysToClone.angles;
        Desc = keysToClone.Desc.clone();
        rightDesc = keysToClone.rightDesc.clone();
        close = keysToClone.close;
        estimatedDepth = keysToClone.estimatedDepth;
        medianDepth = keysToClone.medianDepth;
        xMult = keysToClone.xMult;
        yMult = keysToClone.yMult;
        xGrids = keysToClone.xGrids;
        yGrids = keysToClone.yGrids;
    }
};

class FeatureExtractor
{

    const int nFeatures;
    const int edgeThreshold;
    const int patchSize;
    const int halfPatchSize {15};
    const int maxFastThreshold;
    const int minFastThreshold;
    const bool nonMaxSuppression;
    const int mnContr {0};

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

    std::vector<cv::Point> pattern;
    
    std::vector<int> umax;

    public:
        const float imScale;
        const int nLevels;

        int maskRadius {5};
    
        std::vector <cv::Mat> imagePyramid;
        std::vector<int>scaledPatchSize;
        std::vector < float > scalePyramid;
        std::vector < float > scaleInvPyramid;
        std::vector < float > sigmaFactor;
        std::vector < float > InvSigmaFactor;
        std::vector < int > featurePerLevel;
        
        
        FeatureExtractor(const int _nfeatures = 2000, const int _nLevels = 8, const float _imScale = 1.2f, const int _edgeThreshold = 19, const int _patchSize = 31, const int _maxFastThreshold = 20, const int _minFastThreshold = 7, const bool _nonMaxSuppression = true);
        
        int getGridRows();
        int getGridCols();

        std::vector<cv::KeyPoint> ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints,
                         float tolerance, int cols, int rows);

        void extractKeysNew(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

        void computeAllOrientations(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

        void computeKeypointsORBNew(cv::Mat& image, std::vector<std::vector<cv::KeyPoint>>& allKeys);

        void computePyramid(const cv::Mat& image);
        float computeOrientation(const cv::Mat& image, const cv::Point2f& point);
        // static void computeOrbDescriptor(const cv::KeyPoint& kpt,const cv::Mat& img, const cv::Point* pattern, uchar* desc);
    
};

} // namespace vio_slam

#endif // FEATUREEXTRACTOR_H