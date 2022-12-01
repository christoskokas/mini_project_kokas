#include "System.h"


namespace vio_slam
{

System::System(std::string& confFile)
{
    mConf = new ConfigFile(confFile.c_str());

    mZedCamera = new Zed_Camera(mConf);

    mFrame = new Frame;

    mRb = new RobustMatcher2(mZedCamera);

    map = new Map();

    // Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera);

    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb);

    // vio_slam::Frame frame;
    // vio_slam::RobustMatcher2 rb(zedptr);
    // std::thread worker(&vio_slam::Frame::pangoQuit, frame, zedptr);
    // std::thread tester(&vio_slam::RobustMatcher2::beginTest, &rb);
}

void System::SLAM()
{
    Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera);

    Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb, map);

    Visual->join();
    Tracking->join();
}

} // namespace vio_slam