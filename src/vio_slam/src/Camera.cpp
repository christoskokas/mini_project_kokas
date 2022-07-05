#include "Camera.h"



namespace vio_slam
{

Zed_Camera::Zed_Camera(ros::NodeHandle *nh, bool rectified)
{
    this->rectified = rectified;
    setCameraValues(nh);
    setCameraMatrices(nh);

}

void Zed_Camera::setCameraMatrices(ros::NodeHandle* nh)
{
    std::vector< double > transf;
    nh->getParam("Stereo/T_c1_c2/data", transf);
    cameraLeft.cameraMatrix = (cv::Mat_<double>(3,3) << cameraLeft.fx, 0.0f, cameraLeft.cx, 0.0f, cameraLeft.fy, cameraLeft.cy, 0.0f, 0.0f, 1);
    cameraLeft.distCoeffs = (cv::Mat_<double>(1,5) << cameraLeft.k1, cameraLeft.k2, cameraLeft.p1, cameraLeft.p2, cameraLeft.k3);
    cameraRight.cameraMatrix = (cv::Mat_<double>(3,3) << cameraRight.fx, 0.0f, cameraRight.cx, 0.0f, cameraRight.fy, cameraRight.cy, 0.0f, 0.0f, 1);
    cameraRight.distCoeffs = (cv::Mat_<double>(1,5) << cameraRight.k1, cameraRight.k2, cameraRight.p1, cameraRight.p2, cameraRight.k3);
    double translate[3][1] = {{transf[3]}, {transf[7]}, {transf[11]}};
    double rotate[3][3] = {{transf[0],transf[1],transf[2]},{transf[4],transf[5],transf[6]},{transf[8],transf[9],transf[10]}};
    sensorsTranslate = (cv::Mat_<double>(3,1) << transf[3], transf[7], transf[11]);
    sensorsRotate = (cv::Mat_<double>(3,3) << transf[0], transf[1], transf[2], transf[4], transf[5], transf[6], transf[8], transf[9], transf[10]);
}

void Zed_Camera::setCameraValues(ros::NodeHandle* nh)
{
    nh->getParam("/Camera/width",mWidth);
    nh->getParam("/Camera/height",mHeight);
    nh->getParam("/Camera/fps",mFps);
    nh->getParam("/Camera/bl",mBaseline);
    cameraLeft.setIntrinsicValues(nh, "Camera_l");
    cameraRight.setIntrinsicValues(nh, "Camera_r");
}

Zed_Camera::~Zed_Camera()
{

}

void Zed_Camera::GetResolution()
{
    ROS_INFO("Height : [%d], Width : [%d]", mHeight, mWidth);
}

Camera::Camera(ros::NodeHandle *nh)
{

}

Camera::~Camera()
{
    
}


float Camera::GetFx()
{
    return fx;
}

void Camera::setIntrinsicValues(ros::NodeHandle* nh, const std::string& cameraPath)
{
    nh->getParam(cameraPath + "/fx",fx);
    nh->getParam(cameraPath + "/fy",fy);
    nh->getParam(cameraPath + "/cx",cx);
    nh->getParam(cameraPath + "/cy",cy);
    nh->getParam(cameraPath + "/k1",k1);
    nh->getParam(cameraPath + "/k2",k2);
    nh->getParam(cameraPath + "/p1",p1);
    nh->getParam(cameraPath + "/p2",p2);
    nh->getParam(cameraPath + "/k3",k3);
    nh->getParam(cameraPath + "_path", path);
}

} //namespace vio_slam

