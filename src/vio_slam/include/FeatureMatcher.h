#pragma once

#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include "Settings.h"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace vio_slam
{

struct KeysWithIndx
{
    std::vector<std::vector < cv::KeyPoint > > keysPerY;
    std::vector<std::vector < int > > indexes;
};

class FeatureMatcher
{
    private:
        const int stereoYSpan;
        const int imageHeight;
        const int gridRows, gridCols;

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);
        void matchKeys(const std::vector < cv::KeyPoint >& leftKeys, const std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches);
        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        
    public:
        FeatureMatcher(const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 3);

        void stereoMatch(const std::vector<cv::KeyPoint>& leftKeys, const std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches);


};

} // namespace vio_slam

#endif // FEATUREMATCHER_H