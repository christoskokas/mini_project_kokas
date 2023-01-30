#pragma once

#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include "Settings.h"
#include "Camera.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

namespace vio_slam
{

class Map;
class MapPoint;
class KeyFrame;

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
        const int closeNumber {40};
        const int thDist {75};
        const int matchDist {50};
        const int matchDistConVel {50};
        const int matchDistProj {100};
        const int maxDistAng {25};
        // const int matchDistProj {40};

        // std::vector<std::vector<std::vector<int>>> leftIdxs;


        const Zed_Camera* zedptr;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01000000000000000021))};

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);
        void matchKeys(std::vector < cv::KeyPoint >& leftKeys, std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& tempMatches);

        void matchPoints(const StereoDescriptors& desc, const std::vector<std::vector < int > >& indexes, std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points, StereoKeypoints& keypoints);

        
    public:
        const FeatureExtractor* feLeft, *feRight;

        FeatureMatcher(const Zed_Camera* _zed, const FeatureExtractor* _feLeft, const FeatureExtractor* _feRight, const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 2);

        void getMatchIdxs(const cv::Point2f& predP, std::vector<int>& idxs, const TrackedKeys& keysLeft, const int predictedScale, const float radius, bool right);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        void matchLocalBA(std::vector<std::vector<std::pair<int, int>>>& matchedIdxs, KeyFrame* lastKF, KeyFrame* otherKF, const int aKFSize, const int timesGrid, bool first, std::vector<float>& keysAngles, const std::vector<cv::Point2f>& predPoints);

        void findStereoMatchesORB2R(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);


        int matchByProjectionR(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, const float rad);
        int matchByProjectionRPred(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, const float rad, std::vector<std::pair<cv::Point2f,cv::Point2f>>& prevKeyPositions, const bool pred);
        int matchByProjectionRPredLBA(const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<std::pair<float, float>>& keysAngles, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d);
        int matchByProjection(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB);
        
        int matchByProjectionPred(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const int timesGrid);
        int matchByProjectionPredWA(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, std::vector<cv::KeyPoint>& prevKeyPos, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const int timesGrid, const std::vector<float>& mapAngles);
        int matchByProjectionConVel(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const int timesGrid);
        
        int matchByProjectionConVelAng(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, std::vector<cv::KeyPoint>& prevKeyPos, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const int timesGrid, const std::vector<float>& mapAngles);
        int matchByProjectionConVelAngScale(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, std::vector<cv::KeyPoint>& prevKeyPos, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const int timesGrid, const std::vector<float>& mapAngles, const std::vector<int> scaleKeys);

        void stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches, SubPixelPoints& points);

        void findStereoMatchesCloseFar(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void findStereoMatchesORB(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void findStereoMatchesORB2(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);
        void matchORBPoints(TrackedKeys& prevLeftKeys, TrackedKeys& keysLeft);
        void destributeLeftKeys(TrackedKeys& keysLeft, std::vector<std::vector<std::vector<int>>>& leftIdxs, const int lnGrids, const int rnGrids);
        void destributeLeftKeysoct(TrackedKeys& keysLeft, std::vector<std::vector<std::vector<int>>>& leftIdxs, const int lnGrids, const int rnGrids);

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