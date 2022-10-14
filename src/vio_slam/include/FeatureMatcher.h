#pragma once

#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include "Settings.h"
#include "Camera.h"
#include "FeatureExtractor.h"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

namespace vio_slam
{

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

        const Zed_Camera* zedptr;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 60, (0.01000000000000000021))};

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);
        void matchKeys(std::vector < cv::KeyPoint >& leftKeys, std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& tempMatches);

        void matchPoints(const StereoDescriptors& desc, const std::vector<std::vector < int > >& indexes, std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points, StereoKeypoints& keypoints);

        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        
    public:
        FeatureMatcher(const Zed_Camera* _zed, const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 1);

        void stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches, SubPixelPoints& points);

        void computeStereoMatches(const cv::Mat& leftImage, const cv::Mat& rightImage, const StereoDescriptors& desc, std::vector <cv::DMatch>& matches, SubPixelPoints& points, StereoKeypoints& keypoints);

        void slidingWindowOpt(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, SubPixelPoints& points);

        void slidingWindowOptimization(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points);

        void computeOpticalFlow(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, const std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints, std::vector <uchar>& status);

        void slidingWindowOptical(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints);

        void addUcharVectors(std::vector <uchar>& first, std::vector <uchar>& second);

};

} // namespace vio_slam

#endif // FEATUREMATCHER_H