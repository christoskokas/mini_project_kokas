#pragma once

#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H

#include "opencv2/core.hpp"
#include "Camera.h"
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace vio_slam
{

class PoseEstimator
{
    private:

        const Zed_Camera* zedcamera;

        float prevdt {};
        cv::Mat pprevR,pptra,prevR,ptra;
        void setPrevPrevR(cv::Mat& R);
        void setPrevPrevT(cv::Mat& T);
        void convertToEigenMat(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& transform);
    public:
        PoseEstimator(const Zed_Camera* _zedcamera);
        void predictPose(cv::Mat& Rvec, cv::Mat& Tvec, const float dt);
        void estimatePose(std::vector<cv::Point3d>& points3D, std::vector<cv::Point2d>& points2D, const float dt, Eigen::Matrix4d& transform);
        void initializePose(std::vector<cv::Point3d>& points3D, std::vector<cv::Point2d>& points2D, const float dt, Eigen::Matrix4d& transform);
        void setPrevR(cv::Mat& R);
        void setPrevT(cv::Mat& T);
        void setPrevDt(float dt);
    
};



} // namespace vio_slam


#endif // POSEESTIMATOR_H