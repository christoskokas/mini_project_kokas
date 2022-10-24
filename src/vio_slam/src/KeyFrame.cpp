#include "KeyFrame.h"

namespace vio_slam
{

KeyFrame::KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb) : numb(_numb)
{
    points3D = points;
    pose.setPose(poseT);
    homoPoints3D = _homoPoints3D;
}

Eigen::Vector4d KeyFrame::getWorldPosition(int idx)
{
    return pose.pose * homoPoints3D.row(idx).transpose();
}

};