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

namespace vio_slam
{

class FeatureExtractor
{

    

    const int nFeatures;
    const int nLevels;
    const float imScale;
    const int edgeThreshold;
    const int maxFastThreshold;
    const int minFastThreshold;
    const bool nonMaxSuppression;

    const int gridRows {5};
    const int gridCols {5};
    const int gridsNumber {gridCols * gridRows};
    const int numberPerCell = 2*nFeatures/(gridRows * gridCols * nLevels);

    std::vector <cv::Mat> imagePyramid;
    std::vector < float > scalePyramid;
    std::vector < float > scaleInvPyramid;



    public:
        
        enum FeatureChoice
        {
            ORB,
            FAST
        };

        FeatureChoice choice;
        
        FeatureExtractor(FeatureChoice _choice = ORB, const int _nfeatures = 1000, const int _nLevels = 8, const float _imScale = 1.2f, const int _edgeThreshold = 10, const int _maxFastThreshold = 20, const int _minFastThreshold = 6, const bool _nonMaxSuppression = true);
        
        void findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);
        void findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);

        void computePyramid(cv::Mat& image);

        void separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys);

        void highSpeedTest(const uchar* rowPtr, int pixels[25], const int fastThresh);
        void getPixelOffset(int pixels[25], int rowStride);
        int checkIntensities(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int thresh);
        float computeScore(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int fastThresh);
    // FindFeatures orbs;


};

} // namespace vio_slam

#endif // FEATUREEXTRACTOR_H