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
        std::vector<bool> unMatchedF;
        TrackedKeys keys;
        Eigen::MatrixXd homoPoints3D;
        const int numb;
        const int frameIdx;
        int nKeysTracked {0};
        bool visualize {true};
        std::vector<MapPoint*> localMapPoints;
        bool active {true};
        bool keyF {false};
        bool LBA {false};



        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d _pose, const int _numb);
        KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        Eigen::Vector4d getWorldPosition(int idx);

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