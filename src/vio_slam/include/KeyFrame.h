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
        std::vector<cv::Point3d> points3D;
        Eigen::MatrixXd homoPoints3D;
        const int numb;

        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb = 0);
        KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb = 0);
        Eigen::Vector4d getWorldPosition(int idx);

        Eigen::Matrix4d getPose();
};

} // namespace vio_slam

#endif // KEYFRAME_H