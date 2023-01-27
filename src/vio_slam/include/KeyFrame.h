#pragma once

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Settings.h"
#include "Camera.h"
#include "FeatureExtractor.h"
#include "Map.h"
#include "opencv2/core.hpp"


namespace vio_slam
{

class Map;
class MapPoint;

class Frame
{
    private:

    public:

    Camera* lCam, rCam;
    Camera* lCamB, rCamB;

    CameraPose pose;
    CameraPose poseB;

    Eigen::Matrix4d Tc1c2;

    cv::Mat lIm, rIm;
    cv::Mat rLIm;
    cv::Mat lImB, rImB;
    cv::Mat rLImB;

    int rN, lN;
    int rNB, lNB;

    std::vector<cv::KeyPoint> lKeys, rKeys;
    std::vector<cv::KeyPoint> lKeysB, rKeysB;
    std::vector<float> estDepth;
    std::vector<float> estDepthB;
    cv::Mat lDesc, rDesc;
    cv::Mat lDescB, rDescB;

    std::vector<MapPoint*> lMapPoints;
    std::vector<MapPoint*> rMapPoints;
    std::vector<MapPoint*> lMapPointsB;
    std::vector<MapPoint*> rMapPointsB;

    Frame(){};

    Frame(Zed_Camera* zedPtr);

};

class KeyFrame
{
    private:

    public:
        CameraPose pose;
        cv::Mat leftIm, rightIm;
        cv::Mat rLeftIm;
        std::vector<cv::Point3d> points3D;
        std::vector<int> connections;
        std::vector<int> connectionWeights;
        std::vector<int> unMatchedF;
        std::vector<int> unMatchedFR;
        std::vector<float> scaleFactor;
        float logScale;
        int nScaleLev;
        TrackedKeys keys;
        Eigen::MatrixXd homoPoints3D;
        const int numb;
        const int frameIdx;
        int nKeysTracked {0};
        int closestKF {-1};
        bool visualize {true};
        std::vector<MapPoint*> localMapPoints;
        std::vector<MapPoint*> localMapPointsR;
        std::vector<MapPoint*> localMapPointsB;
        KeyFrame* KFBack;
        KeyFrame* KFFront;
        bool active {true};
        bool keyF {false};
        bool LBA {false};
        bool fixed {false};


        void eraseMPConnection(const int mpPos);
        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d _pose, const int _numb);
        KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        Eigen::Vector4d getWorldPosition(int idx);
        void getConnectedKFs(const Map* map, std::vector<KeyFrame*>& activeKF, const int N);

        Eigen::Matrix4d getPose();
};

class AllKeyFrames
{
    private:

    public:

        std::vector<KeyFrame*> allKeyFrames;


};

} // namespace vio_slam

#endif // KEYFRAME_H