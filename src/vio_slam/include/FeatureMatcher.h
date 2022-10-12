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

struct SubPixelPoints
{
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2f> right;

    SubPixelPoints(const int leftSize,const int rightSize)
    {
        left.reserve(leftSize);
        right.reserve(rightSize);
    }
};

struct MatchedKeysDist
{
    std::vector<int> dist;
    std::vector<int> lIdx;

    MatchedKeysDist(const int size, const int value1,const int value2)
    {
        std::vector<int> temp(size,value1);
        dist = temp;
        std::vector<int> temp2(size,value2);
        lIdx = temp2;
    }
};

class FeatureMatcher
{
    private:
        const int stereoYSpan;
        const int imageHeight;
        const int gridRows, gridCols;

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);
        void matchKeys(std::vector < cv::KeyPoint >& leftKeys, std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches);
        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        
    public:
        FeatureMatcher(const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 1);

        void stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches);
        void slidingWindowOpt(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys);


};

} // namespace vio_slam

#endif // FEATUREMATCHER_H