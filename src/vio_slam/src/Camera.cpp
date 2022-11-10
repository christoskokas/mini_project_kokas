#include "Camera.h"



namespace vio_slam
{

void CameraPose::separatePose()
{
    Rv = pose.topLeftCorner<3,3>();
    tv = pose.topRightCorner<3,1>();
}

void CameraPose::setVel(const double _vx, const double _vy, const double _vz)
{
    vx = _vx;
    vy = _vy;
    vz = _vz;
}

CameraPose::CameraPose(Eigen::Matrix4d _pose, std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp) : pose(_pose), timestamp(_timestamp)
{
    poseInverse = Eigen::Matrix4d::Identity();
}

void CameraPose::setPose(Eigen::Matrix4d poseT)
{
    pose = poseT;
    separatePose();
    // Eigen::Matrix4d temp = poseT.inverse();
    // poseInverse = temp;
    timestamp = std::chrono::high_resolution_clock::now();
}

void CameraPose::setInvPose(Eigen::Matrix4d poseT)
{
    poseInverse = poseT;
}

Zed_Camera::Zed_Camera(ConfigFile& yamlFile)
{
    confFile = &yamlFile;
    this->rectified = confFile->getValue<bool>("rectified");
    numOfFrames = confFile->getValue<int>("numOfFrames");
#if KITTI_DATASET
    seq = confFile->getValue<std::string>("seq");
#endif
    setCameraValues();
    setCameraMatrices();

}

void Zed_Camera::setCameraMatrices()
{

    std::vector < double > transf {confFile->getValue<std::vector<double>>("Stereo","T_c1_c2","data")};

    // std::vector< double > transf;
    // nh->getParam("Stereo/T_c1_c2/data", transf);
    cameraLeft.cameraMatrix = (cv::Mat_<double>(3,3) << cameraLeft.fx, 0.0f, cameraLeft.cx, 0.0f, cameraLeft.fy, cameraLeft.cy, 0.0f, 0.0f, 1);
    cameraLeft.distCoeffs = (cv::Mat_<double>(1,5) << cameraLeft.k1, cameraLeft.k2, cameraLeft.p1, cameraLeft.p2, cameraLeft.k3);
    cameraRight.cameraMatrix = (cv::Mat_<double>(3,3) << cameraRight.fx, 0.0f, cameraRight.cx, 0.0f, cameraRight.fy, cameraRight.cy, 0.0f, 0.0f, 1);
    cameraRight.distCoeffs = (cv::Mat_<double>(1,5) << cameraRight.k1, cameraRight.k2, cameraRight.p1, cameraRight.p2, cameraRight.k3);
    double translate[3][1] = {{transf[3]}, {transf[7]}, {transf[11]}};
    double rotate[3][3] = {{transf[0],transf[1],transf[2]},{transf[4],transf[5],transf[6]},{transf[8],transf[9],transf[10]}};
    sensorsTranslate = (cv::Mat_<double>(3,1) << transf[3], transf[7], transf[11]);
    sensorsRotate = (cv::Mat_<double>(3,3) << transf[0], transf[1], transf[2], transf[4], transf[5], transf[6], transf[8], transf[9], transf[10]);
}

void Zed_Camera::setCameraValues()
{
    // nh->getParam("/Camera/width",mWidth);
    // nh->getParam("/Camera/height",mHeight);
    // nh->getParam("/Camera/fps",mFps);
    // nh->getParam("/Camera/bl",mBaseline);
    mWidth = confFile->getValue<int>("Camera","width");
    mHeight = confFile->getValue<int>("Camera","height");
    mFps = confFile->getValue<float>("Camera","fps");
    if ( rectified )
    {
         cameraLeft.setIntrinsicValuesR("Camera_l",confFile);
        cameraRight.setIntrinsicValuesR("Camera_r",confFile);
        mBaseline = setBaseline();
    }
    else
    {
        cameraLeft.setIntrinsicValuesUnR("Camera_l",confFile);
        cameraRight.setIntrinsicValuesUnR("Camera_r",confFile);
        mBaseline = confFile->getValue<float>("Camera","bl");
    }
}

float Zed_Camera::setBaseline()
{
    std::vector < float > P {confFile->getValue<std::vector<float>>("Camera_r","P","data")};

    return -P[3]/(float)cameraLeft.fx;
}

Zed_Camera::~Zed_Camera()
{

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

void Camera::setIntrinsicValuesUnR(const std::string& cameraPath, ConfigFile* confFile)
{
    fx = confFile->getValue<double>(cameraPath,"fx");
    fy = confFile->getValue<double>(cameraPath,"fy");
    cx = confFile->getValue<double>(cameraPath,"cx");
    cy = confFile->getValue<double>(cameraPath,"cy");
    k1 = confFile->getValue<double>(cameraPath,"k1");
    k2 = confFile->getValue<double>(cameraPath,"k2");
    p1 = confFile->getValue<double>(cameraPath,"p1");
    p2 = confFile->getValue<double>(cameraPath,"p2");
    k3 = confFile->getValue<double>(cameraPath,"k3");
    path = confFile->getValue<std::string>(cameraPath + "_path");
    // nh->getParam(cameraPath + "/fx",fx);
    // nh->getParam(cameraPath + "/fy",fy);
    // nh->getParam(cameraPath + "/cx",cx);
    // nh->getParam(cameraPath + "/cy",cy);
    // nh->getParam(cameraPath + "/k1",k1);
    // nh->getParam(cameraPath + "/k2",k2);
    // nh->getParam(cameraPath + "/p1",p1);
    // nh->getParam(cameraPath + "/p2",p2);
    // nh->getParam(cameraPath + "/k3",k3);
    // nh->getParam(cameraPath + "_path", path);
}

void Camera::setIntrinsicValuesR(const std::string& cameraPath, ConfigFile* confFile)
{
    std::vector < double > P {confFile->getValue<std::vector<double>>(cameraPath,"P","data")};

    fx = P[0];
    fy = P[5];
    cx = P[2];
    cy = P[6];
    k1 = 0;
    k2 = 0;
    p1 = 0;
    p2 = 0;
    k3 = 0;
    // nh->getParam(cameraPath + "/fx",fx);
    // nh->getParam(cameraPath + "/fy",fy);
    // nh->getParam(cameraPath + "/cx",cx);
    // nh->getParam(cameraPath + "/cy",cy);
    // nh->getParam(cameraPath + "/k1",k1);
    // nh->getParam(cameraPath + "/k2",k2);
    // nh->getParam(cameraPath + "/p1",p1);
    // nh->getParam(cameraPath + "/p2",p2);
    // nh->getParam(cameraPath + "/k3",k3);
    // nh->getParam(cameraPath + "_path", path);
}

} //namespace vio_slam

