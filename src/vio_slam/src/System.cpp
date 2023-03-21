#include "System.h"


namespace vio_slam
{

System::System(ConfigFile* _mConf)
{
    mConf = _mConf;

    mZedCamera = new Zed_Camera(mConf);

    mFrame = new ViewFrame;

    map = new Map();

    featTracker = new FeatureTracker(mZedCamera,map);

    feLeft = new FeatureExtractor(nFeatures);
    feRight = new FeatureExtractor(nFeatures);

    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight);

    localMap = new LocalMapper(map, mZedCamera, fm);
    loopCl = new LocalMapper(map, mZedCamera, fm);

    Visual = new std::thread(&vio_slam::ViewFrame::pangoQuit, mFrame, mZedCamera, map);

    LocalMapping = new std::thread(&vio_slam::LocalMapper::beginLocalMapping, localMap);

    LoopClosure = new std::thread(&vio_slam::LocalMapper::beginLoopClosure, loopCl);
    
    // Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera);

    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb);

    // vio_slam::Frame frame;
    // vio_slam::RobustMatcher2 rb(zedptr);
    // std::thread worker(&vio_slam::Frame::pangoQuit, frame, zedptr);
    // std::thread tester(&vio_slam::RobustMatcher2::beginTest, &rb);
}

System::System(ConfigFile* _mConf, bool multi)
{
    mConf = _mConf;

    mZedCamera = new Zed_Camera(mConf, false);
    mZedCameraB = new Zed_Camera(mConf, true);

    mFrame = new ViewFrame;

    map = new Map();

    featTracker = new FeatureTracker(mZedCamera, mZedCameraB,map);

    feLeft = new FeatureExtractor(nFeatures);
    feRight = new FeatureExtractor(nFeatures);

    feLeftB = new FeatureExtractor(nFeatures);
    feRightB = new FeatureExtractor(nFeatures);

    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight);

    localMap = new LocalMapper(map, mZedCamera, mZedCameraB, fm);
    loopCl = new LocalMapper(map, mZedCamera, mZedCameraB, fm);

    Visual = new std::thread(&vio_slam::ViewFrame::pangoQuitMulti, mFrame, mZedCamera, mZedCameraB, map);

    LocalMapping = new std::thread(&vio_slam::LocalMapper::beginLocalMappingB, localMap);

    LoopClosure = new std::thread(&vio_slam::LocalMapper::beginLoopClosureB, loopCl);
    
}

void System::saveTrajectory(const std::string& filepath)
{
    std::vector<KeyFrame*>& allFrames = map->allFramesPoses;
    KeyFrame* closeKF = allFrames[0];
    std::ofstream datafile(filepath);
    std::vector<KeyFrame*>::iterator it;
    std::vector<KeyFrame*>::const_iterator end(allFrames.end());
    for ( it = allFrames.begin(); it != end; it ++)
    {
        KeyFrame* candKF = *it;
        Eigen::Matrix4d matT;
        if ( candKF->keyF )
        {
            matT = candKF->pose.pose;
            closeKF = candKF;
        }
        else
        {
            matT = (closeKF->pose.getPose() * candKF->pose.refPose);
        }
        Eigen::Matrix4d mat = matT.transpose();
        for (int32_t i{0}; i < 12; i ++)
        {
            if ( i == 0 )
                datafile << mat(i);
            else
                datafile << " " << mat(i);
        }
        datafile << '\n';
    }
    datafile.close();

}

void System::saveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition)
{
    std::vector<KeyFrame*>& allFrames = map->allFramesPoses;
    KeyFrame* closeKF = allFrames[0];
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    std::vector<KeyFrame*>::iterator it;
    std::vector<KeyFrame*>::const_iterator end(allFrames.end());
    for ( it = allFrames.begin(); it != end; it ++)
    {
        KeyFrame* candKF = *it;
        Eigen::Matrix4d matT;
        if ( candKF->keyF )
        {
            matT = candKF->pose.pose;
            closeKF = candKF;
        }
        else
        {
            matT = (closeKF->pose.getPose() * candKF->pose.refPose);
        }
        Eigen::Matrix4d mat = matT.transpose();
        for (int32_t i{0}; i < 12; i ++)
        {
            if ( i == 0 )
                datafile << mat(i);
            else
                datafile << " " << mat(i);
            if ( i == 3 || i == 7 || i == 11 )
                datafilePos << mat(i) << " ";
        }
        datafile << '\n';
        datafilePos << '\n';
    }
    datafile.close();
    datafilePos.close();

}

void System::trackNewImage(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb)
{
    // std::thread track(&FeatureTracker::TrackImageT, featTracker,std::ref(imLRect), std::ref(imRRect), std::ref(frameNumb));
    // track.join();
    featTracker->TrackImageT(imLRect, imRRect, frameNumb);
}

void System::trackNewImageMutli(const cv::Mat& imLRect, const cv::Mat& imRRect, const cv::Mat& imLRectB, const cv::Mat& imRRectB, const int frameNumb)
{
    // std::thread track(&FeatureTracker::TrackImageTB, featTracker,std::ref(imLRect), std::ref(imRRect),std::ref(imLRectB), std::ref(imRRectB), std::ref(frameNumb));
    // track.join();
    featTracker->TrackImageTB(imLRect, imRRect, imLRectB, imRRectB, frameNumb);
}

void System::exitSystem()
{
    mFrame->stopRequested = true;
    localMap->stopRequested = true;
    loopCl->stopRequested = true;
    Visual->join();
    LocalMapping->join();
    LoopClosure->join();
}

} // namespace vio_slam