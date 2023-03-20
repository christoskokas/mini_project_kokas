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

class FeatureMatcher
{
    private:
        const int stereoYSpan;
        const int imageHeight;
        const int gridRows, gridCols;
        const int maxMatches {1000};
        const int mnDisp;
        const int thDist {75};
        const int matchDist {50};
        const int matchDistConVel {50};
        const int matchDistProj {100};
        const float ratioProj {0.8};
        const int matchDistLBA {50};
        const float ratioLBA {0.6};
        const int maxDistAng {4};
        // const int matchDistProj {40};

        // std::vector<std::vector<std::vector<int>>> leftIdxs;


        const Zed_Camera* zedptr;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01000000000000000021))};

        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);

        
    public:
        const int closeNumber {40};
        const FeatureExtractor* feLeft, *feRight;

        FeatureMatcher(const Zed_Camera* _zed, const FeatureExtractor* _feLeft, const FeatureExtractor* _feRight, const int _imageHeight = 360, const int _gridRows = 5, const int _gridCols = 5, const int _stereoYSpan = 2);

        int matchByProjectionRPredLBAB(const Zed_Camera* zedCam, const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, const bool back);
        void getMatchIdxs(const cv::Point2f& predP, std::vector<int>& idxs, const TrackedKeys& keysLeft, const int predictedScale, const float radius, bool right);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        void findStereoMatchesORB2R(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);


        int matchByProjectionRPred(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, const float rad, const bool pred);
        int matchByProjectionRPredLBA(const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<std::pair<float, float>>& keysAngles, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, const bool pred);

};

} // namespace vio_slam

#endif // FEATUREMATCHER_H