#include "PoseEstimator.h"

namespace vio_slam
{

PoseEstimator::PoseEstimator(const Zed_Camera* _zedcamera) : zedcamera(_zedcamera)
{

}

void PoseEstimator::convertToEigenMat(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& transform)
{
    cv::Mat R;
    cv::Rodrigues(Rvec,R);

    Eigen::Matrix3d Reig;
    Eigen::Vector3d teig;
    cv::cv2eigen(R.t(),Reig);
    cv::cv2eigen(-tvec,teig);

    transform.setIdentity();
    transform.block<3,3>(0,0) = Reig;
    transform.block<3,1>(0,3) = teig;
}

void PoseEstimator::predictPose(cv::Mat& Rvec, cv::Mat& Tvec, const float dt)
{
    Rvec.at<double>(0,0) = (prevR.at<double>(0,0) - pprevR.at<double>(0,0))*dt/prevdt;
    Rvec.at<double>(0,1) = (prevR.at<double>(0,1) - pprevR.at<double>(0,1))*dt/prevdt;
    Rvec.at<double>(0,2) = (prevR.at<double>(0,2) - pprevR.at<double>(0,2))*dt/prevdt;
    Tvec.at<double>(0,0) = (ptra.at<double>(0,0) - pptra.at<double>(0,0))*dt/prevdt;
    Tvec.at<double>(0,1) = (ptra.at<double>(0,1) - pptra.at<double>(0,1))*dt/prevdt;
    Tvec.at<double>(0,2) = (ptra.at<double>(0,2) - pptra.at<double>(0,2))*dt/prevdt;
    prevdt = dt;
}

void PoseEstimator::estimatePose(std::vector<cv::Point3d>& points3D, std::vector<cv::Point2d>& points2D, const float dt, Eigen::Matrix4d& transform)
{
    cv::Mat Rvec(3,1,CV_64F), tvec(3,1,CV_64F);
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);

    predictPose(Rvec, tvec, dt);
    if (points3D.size() > 4)
        if (!cv::solvePnP(points3D,points2D,zedcamera->cameraLeft.cameraMatrix,dist, Rvec,tvec,true,cv::SOLVEPNP_ITERATIVE))
            Logging("SolvePnP Failed, keep device steady!","",3);
        // if (!cv::solvePnPRansac(points3D,points2D,zedcamera->cameraLeft.cameraMatrix,dist, Rvec,tvec,true,cv::SOLVEPNP_ITERATIVE,8))
        //     Logging("SolvePnP Failed, keep device steady!","",3);

    setPrevR(Rvec);
    setPrevT(tvec);
    setPrevDt(dt);

    convertToEigenMat(Rvec,tvec,transform);

}

void PoseEstimator::initializePose(std::vector<cv::Point3d>& points3D, std::vector<cv::Point2d>& points2D, const float dt, Eigen::Matrix4d& transform)
{
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    cv::Mat Rvec,tvec;

    bool sPnP {false};
    if (points3D.size() > 4)
    {
        // if (!cv::solvePnPRansac(points3D,points2D,zedcamera->cameraLeft.cameraMatrix,dist, Rvec,tvec,true,cv::SOLVEPNP_ITERATIVE,8))
        //     Logging("SolvePnP Failed, keep device steady!","",3);
        if (!cv::solvePnP(points3D,points2D,zedcamera->cameraLeft.cameraMatrix,dist, Rvec,tvec,false,cv::SOLVEPNP_ITERATIVE))
            Logging("Initialization Failed, keep device steady!","",3);
        else
        {
            setPrevR(Rvec);
            setPrevT(tvec);
            setPrevDt(dt);
            convertToEigenMat(Rvec, tvec, transform);
        }
    }
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