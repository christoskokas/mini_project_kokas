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
        const int maxMatches {1000};
        const int mnDisp;
        const int closeNumber {100};
        const int thDist {75};


        const Zed_Camera* zedptr;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01000000000000000021))};

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);
        void matchKeys(std::vector < cv::KeyPoint >& leftKeys, std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& tempMatches);

        void matchPoints(const StereoDescriptors& desc, const std::vector<std::vector < int > >& indexes, std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points, StereoKeypoints& keypoints);

        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        
    public:
        const FeatureExtractor* feLeft, *feRight;

        FeatureMatcher(const Zed_Camera* _zed, const FeatureExtractor* _feLeft, const FeatureExtractor* _feRight, const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 2);

        void stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches, SubPixelPoints& points);

        void findStereoMatchesCloseFar(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void findStereoMatchesORB(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void findStereoMatchesORB2(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void matchORBPoints(TrackedKeys& prevLeftKeys, TrackedKeys& keysLeft);
        void destributeLeftKeys(TrackedKeys& keysLeft, std::vector<std::vector<std::vector<int>>>& leftIdxs, const int lnGrids, const int rnGrids);

        void findMatchesWD(const cv::Mat& lImage, const cv::Mat& rImage, const StereoDescriptors& desc, PointsWD& points, StereoKeypoints& keypoints);
        void findStereoMatchesFAST(const cv::Mat& lImage, const cv::Mat& rImage, const StereoDescriptors& desc, SubPixelPoints& points, StereoKeypoints& keypoints);
        void findStereoMatches(const StereoDescriptors& desc, SubPixelPoints& points, StereoKeypoints& keypoints);
        void findStereoMatchesClose(const StereoDescriptors& desc, SubPixelPoints& points, StereoKeypoints& keypoints);

        void computeStereoMatches(const cv::Mat& leftImage, const cv::Mat& rightImage, const StereoDescriptors& desc, std::vector <cv::DMatch>& matches, SubPixelPoints& points, StereoKeypoints& keypoints);
        void computeStereoMatchesClose(const cv::Mat& leftImage, const cv::Mat& rightImage, const StereoDescriptors& desc, std::vector <cv::DMatch>& matches, SubPixelPoints& points, StereoKeypoints& keypoints);

        void slidingWindowOpt(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, SubPixelPoints& points);

        void slidingWindowOptimization(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points);
        void slWinGF(const cv::Mat& leftImage, const cv::Mat& rightImage, SubPixelPoints& points);
        void slidingWindowOptimizationClose(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points);
        void checkDepthChange(const cv::Mat& leftImage, const cv::Mat& rightImage, SubPixelPoints& points);
        void pointUpdate(const cv::Mat& leftImage, const cv::Mat& rightImage, cv::Point2f& p1,  float& depth, bool& useable, cv::Point3d& p3d);

        void computeOpticalFlow(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, SubPixelPoints& prevPoints, SubPixelPoints& newPoints);
        void computeOpticalFlowWithSliding(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, SubPixelPoints& prevPoints, SubPixelPoints& newPoints);

        std::vector<bool> slidingWindowOptical(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints);
        std::vector<bool> slidingWindowOpticalFlow(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints);
        std::vector<bool> slidingWindowOpticalLR(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::Point2f>& leftPoints, std::vector<cv::Point2f>& rightPoints);
        std::vector<bool> slidingWindowOpticalBackUp(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints);
        void removeWithFund(SubPixelPoints& prevPoints, SubPixelPoints& points);
        void computeRightPoints(const SubPixelPoints& prevPoints, SubPixelPoints& points);
        void computeDepth(SubPixelPoints& prevPoints, SubPixelPoints& points);
        std::vector<bool> inlierDetection(std::vector < cv::Point3d>& first, std::vector < cv::Point3d>& second, std::vector <cv::Point2d>& toReduce);
        void outlierRejection(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, const cv::Mat& rightIm, SubPixelPoints& prevPoints, SubPixelPoints& points);


        double computeDistanceOf3DPoints(cv::Point3d& first, cv::Point3d& second);

        void addUcharVectors(std::vector <uchar>& first, std::vector <uchar>& second);

        std::vector<bool> getMaxClique( const std::vector<cv::Point3d>& ptsA, const std::vector<cv::Point3d>& ptsB );

        double computeDistanceThreshold(const cv::Point3d& p1, const cv::Point3d& p2, const double L1, const double L2);
        double computeDL(const cv::Point3d& p1,const cv::Point3d& p2, const double L);

        bool checkProjection(Eigen::Vector4d& point, cv::Point2d& kp);

};

} // namespace vio_slam

#endif // FEATUREMATCHER_H