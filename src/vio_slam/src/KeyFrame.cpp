#include "KeyFrame.h"

namespace vio_slam
{

KeyFrame::KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb) : numb(_numb)
{
    points3D = points;
    pose.setPose(poseT);
    homoPoints3D = _homoPoints3D;
}

KeyFrame::KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb) : numb(_numb)
{
    points3D = points;
    pose.setPose(poseT);
    const size_t end {points3D.size()};
    Eigen::MatrixX4d temp(end,4);
    for (size_t i {0}; i<end;i++)
    {
        temp(i,0) = points3D[i].x;
        temp(i,1) = points3D[i].y;
        temp(i,2) = points3D[i].z;
        temp(i,3) = 1;
    }
    homoPoints3D = temp;
}

Eigen::Vector4d KeyFrame::getWorldPosition(int idx)
{
    return pose.pose * homoPoints3D.row(idx).transpose();
}

Eigen::Matrix4d KeyFrame::getPose()
{
    return pose.pose;
}

};