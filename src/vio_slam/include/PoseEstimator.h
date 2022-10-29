#pragma once

#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H

#include "opencv2/core.hpp"
#include "Camera.h"
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

namespace vio_slam
{

class LKalmanFilter
{

    private:
        cv::KalmanFilter KF;

        const int nStates {18};
        const int nMeasurements {6};
        const int nInputs {0};

        const double dt;

        void initKalmanFilter(const double dt);

    public:

        const int minInliers {50};

        LKalmanFilter(const double _dt);

        void fillMeasurements( cv::Mat& measurements,const cv::Mat& translation_measured, const cv::Mat& rotation_measured);
        void updateKalmanFilter(cv::Mat &measurement,cv::Mat &translation_estimated, cv::Mat &rotation_estimated );

};

class PoseEstimator
{
    private:

        const Zed_Camera* zedcamera;

        float prevdt {};
        cv::Mat pprevR,pptra,prevR,ptra;
        void setPrevPrevR(cv::Mat& R);
        void setPrevPrevT(cv::Mat& T);
    public:
        void convertToEigenMat(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& transform);
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