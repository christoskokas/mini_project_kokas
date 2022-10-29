#include "PoseEstimator.h"

namespace vio_slam
{

LKalmanFilter::LKalmanFilter(const double _dt) : dt(_dt)
{
    initKalmanFilter(dt);
}

void LKalmanFilter::initKalmanFilter(double dt)
{
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
                    /* DYNAMIC MODEL */
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
        /* MEASUREMENT MODEL */
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

void LKalmanFilter::fillMeasurements( cv::Mat& measurements,const cv::Mat& translation_measured, const cv::Mat& rotation_measured)
{
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = rotation_measured.at<double>(0);      // roll
    measurements.at<double>(4) = rotation_measured.at<double>(1);      // pitch
    measurements.at<double>(5) = rotation_measured.at<double>(2);      // yaw
}

void LKalmanFilter::updateKalmanFilter(cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated )
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    cv::Rodrigues(eulers_estimated, rotation_estimated);
}

PoseEstimator::PoseEstimator(const Zed_Camera* _zedcamera) : zedcamera(_zedcamera)
{

}

void PoseEstimator::convertToEigenMat(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& transform)
{
    // cv::Mat R;
    // cv::Rodrigues(Rvec,R);

    Eigen::Matrix3d Reig;
    Eigen::Vector3d teig;
    cv::cv2eigen(Rvec.t(),Reig);
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