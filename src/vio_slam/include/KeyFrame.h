#pragma once

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Settings.h"
#include "Camera.h"
#include "FeatureExtractor.h"
#include "opencv2/core.hpp"


namespace vio_slam
{

class KeyFrame
{
    private:

    public:
        CameraPose pose;
        cv::Mat leftIm, rightIm;
        std::vector<cv::Point3d> points3D;
        std::vector<int> connections;
        std::vector<int> connectionWeights;
        Eigen::MatrixXd homoPoints3D;
        const int numb;
        bool visualize {true};



        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d _pose, const int _numb);
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