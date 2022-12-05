#include "Conversions.h"

namespace vio_slam
{

// static Eigen::Matrix3d Converter::convertCVRotToEigen(cv::Mat& Rot)
// {
//     cv::Rodrigues(Rot,Rot);
//     Eigen::Matrix3d RotEig;
//     cv::cv2eigen(Rot, RotEig);
//     return RotEig;
// }

// static Eigen::Vector3d convertCVTraToEigen(cv::Mat& Tra)
// {
//     Eigen::Vector3d traEig;
//     cv::cv2eigen(Tra, traEig);
//     return traEig;
// }

// static Eigen::Matrix4d convertRTtoPose(cv::Mat& Rot, cv::Mat& Tra)
// {
//     Eigen::Vector3d traEig;
//     cv::cv2eigen(Tra, traEig);
//     cv::Rodrigues(Rot,Rot);
//     Eigen::Matrix3d RotEig;
//     cv::cv2eigen(Rot, RotEig);

//     Eigen::Matrix4d convPose = Eigen::Matrix4d::Identity();
//     convPose.block<3,3>(0,0) = RotEig;
//     convPose.block<3,1>(0,3) = traEig;

//     return convPose;
// }

} // namespace vio_slam