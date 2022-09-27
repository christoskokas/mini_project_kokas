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

namespace vio_slam
{

class FeatureExtractor
{

    enum FeatureChoice
    {
        ORB,
        FAST
    };
    

    
    const int nFeatures;
    const int edgeThreshold;
    const int maxFastThreshold;
    const int minFastThreshold;
    const bool nonMaxSuppression;

    FeatureChoice choice;


    public:
        FeatureExtractor(FeatureChoice _choice = FAST, const int _nfeatures = 1000, const int _edgeThreshold = 4, const int _maxFastThreshold = 20, const int _minFastThreshold = 6, const bool _nonMaxSuppression = true);
        
        void findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);

        void highSpeedTest(const uchar* rowPtr, int pixels[25], const int fastThresh);
        void getPixelOffset(int pixels[25], int rowStride);
        int checkIntensities(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int thresh);
        float computeScore(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int fastThresh);
    // FindFeatures orbs;


};

} // namespace vio_slam

#endif // FEATUREEXTRACTOR_H