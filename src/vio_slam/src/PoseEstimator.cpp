#include "PoseEstimator.h"

namespace vio_slam
{

void PoseEstimator::estimatePose(cv::Mat& Rvec, cv::Mat& Tvec, float dt)
{
    Rvec.at<double>(0,0) = (prevR.at<double>(0,0) - pprevR.at<double>(0,0))*dt/prevdt;
    Rvec.at<double>(0,1) = (prevR.at<double>(0,1) - pprevR.at<double>(0,1))*dt/prevdt;
    Rvec.at<double>(0,2) = (prevR.at<double>(0,2) - pprevR.at<double>(0,2))*dt/prevdt;
    Tvec.at<double>(0,0) = (ptra.at<double>(0,0) - pptra.at<double>(0,0))*dt/prevdt;
    Tvec.at<double>(0,1) = (ptra.at<double>(0,1) - pptra.at<double>(0,1))*dt/prevdt;
    Tvec.at<double>(0,2) = (ptra.at<double>(0,2) - pptra.at<double>(0,2))*dt/prevdt;
    prevdt = dt;
}

void PoseEstimator::setPrevDt(float dt)
{
    prevdt = dt;
}

void PoseEstimator::setPrevPrevR(cv::Mat& R)
{
    pprevR = R.clone();
}

void PoseEstimator::setPrevPrevT(cv::Mat& T)
{
    pptra = T.clone();
}

void PoseEstimator::setPrevR(cv::Mat& R)
{
    if (!prevR.empty())
        setPrevPrevR(prevR);
    prevR = R.clone();
}

void PoseEstimator::setPrevT(cv::Mat& T)
{
    if (!ptra.empty())
        setPrevPrevT(ptra);
    ptra = T.clone();
}

} // namespace vio_slam