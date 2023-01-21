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
{}

void CameraPose::setPose(const Eigen::Matrix4d& poseT)
{
    pose = poseT;
    poseInverse = poseT.inverse();
    separatePose();
    // Eigen::Matrix4d temp = poseT.inverse();
    // poseInverse = temp;
    timestamp = std::chrono::high_resolution_clock::now();
}

Eigen::Matrix4d CameraPose::getPose() const
{
    return pose;
}

Eigen::Matrix4d CameraPose::getInvPose() const
{
    return poseInverse;
}

void CameraPose::setPose(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose)
{
    refPose = _refPose;
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::changePose(Eigen::Matrix4d& _keyPose)
{
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::setInvPose(const Eigen::Matrix4d poseT)
{
    poseInverse = poseT;
    pose = poseT.inverse();
}

Zed_Camera::Zed_Camera(ConfigFile* yamlFile)
{
    confFile = yamlFile;
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
    extrinsics(0,3) = (double)mBaseline;
}

float Zed_Camera::setBaseline()
{

#if KITTI_DATASET
    std::vector < float > P {confFile->getValue<std::vector<float>>("Camera_r","P","data")};
    float bl = -P[3]/(float)cameraLeft.fx;
#else
    float bl = confFile->getValue<float>("Camera","bl");
#endif
    return bl;
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
    std::vector < double > Rt = confFile->getValue<std::vector<double>>(cameraPath,"R","data");
    std::vector < double > Pt = confFile->getValue<std::vector<double>>(cameraPath,"P","data");
    std::vector < double > Dt = confFile->getValue<std::vector<double>>(cameraPath,"D","data");
    std::vector < double > Kt = confFile->getValue<std::vector<double>>(cameraPath,"K","data");

    R = (cv::Mat_<double>(3,3) << Rt[0], Rt[1], Rt[2], Rt[3], Rt[4], Rt[5], Rt[6], Rt[7], Rt[8]);
    P = (cv::Mat_<double>(3,4) << Pt[0], Pt[1], Pt[2], Pt[3], Pt[4], Pt[5], Pt[6], Pt[7], Pt[8], Pt[9], Pt[10], Pt[11]);
    K = (cv::Mat_<double>(3,3) << Kt[0], Kt[1], Kt[2], Kt[3], Kt[4], Kt[5], Kt[6], Kt[7], Kt[8]);
    D = (cv::Mat_<double>(1,5) << Dt[0], Dt[1], Dt[2], Dt[3], Dt[4]);
    cameraMatrix = K;
    std::cout << cameraPath<< " cameraMatrix " << cameraMatrix << std::endl;
    // std::cout << cameraPath<< " P " << P << std::endl;
    // std::cout << cameraPath<< " P " << P << std::endl;

    intrisics(0,0) = fx;
    intrisics(1,1) = fy;
    intrisics(0,2) = cx;
    intrisics(1,2) = cy;

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

#if KITTI_DATASET
    std::vector < double > P {confFile->getValue<std::vector<double>>(cameraPath,"P","data")};
    fx = P[0];
    fy = P[5];
    cx = P[2];
    cy = P[6];
#else
    fx = confFile->getValue<double>(cameraPath,"fx");
    fy = confFile->getValue<double>(cameraPath,"fy");
    cx = confFile->getValue<double>(cameraPath,"cx");
    cy = confFile->getValue<double>(cameraPath,"cy");
#endif
    k1 = 0;
    k2 = 0;
    p1 = 0;
    p2 = 0;
    k3 = 0;

    intrisics(0,0) = fx;
    intrisics(1,1) = fy;
    intrisics(0,2) = cx;
    intrisics(1,2) = cy;
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

