#pragma once

#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H

#include "opencv2/core.hpp"

namespace vio_slam
{

class PoseEstimator
{
    private:
        float prevdt {};
        cv::Mat pprevR,pptra,prevR,ptra;
        void setPrevPrevR(cv::Mat& R);
        void setPrevPrevT(cv::Mat& T);
    public:
        PoseEstimator(){};
        void estimatePose(cv::Mat& Rvec, cv::Mat& Tvec, float dt);
        void setPrevR(cv::Mat& R);
        void setPrevT(cv::Mat& T);
        void setPrevDt(float dt);
    
};



} // namespace vio_slam


#endif // POSEESTIMATOR_H