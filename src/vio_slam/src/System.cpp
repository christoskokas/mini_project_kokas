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

    featTracker = new FeatureTracker(mZedCamera,map);

    feLeft = new FeatureExtractor(nFeatures);
    feRight = new FeatureExtractor(nFeatures);

    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight, feLeft->getGridRows(), feLeft->getGridCols());

    localMap = new LocalMapper(map, mZedCamera, fm);
    // Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera);

    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb);

    // vio_slam::Frame frame;
    // vio_slam::RobustMatcher2 rb(zedptr);
    // std::thread worker(&vio_slam::Frame::pangoQuit, frame, zedptr);
    // std::thread tester(&vio_slam::RobustMatcher2::beginTest, &rb);
}

void System::setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<bool>& MPsMatches)
{
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for ( size_t i{0}, end{MPsOutliers.size()}; i < end; i++)
    {
        MapPoint* mp = activeMPs[i];
        if ( MPsMatches[i] )
            mp->unMCnt = 0;
        else
            mp->unMCnt++;
        if ( !MPsOutliers[i] && mp->unMCnt < 20 )
            continue;
        mp->SetIsOutlier( true );
    }
}

void System::SLAM()
{
    Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera, map);

    LocalMapping = new std::thread(&vio_slam::LocalMapper::beginLocalMapping, localMap);

    const int nFrames {mZedCamera->numOfFrames};
    std::vector<std::string>leftImagesStr, rightImagesStr;
    leftImagesStr.reserve(nFrames);
    rightImagesStr.reserve(nFrames);

    const std::string imagesPath = mConf->getValue<std::string>("imagesPath");

    const std::string leftPath = imagesPath + "left/";
    const std::string rightPath = imagesPath + "right/";
    const std::string fileExt = mConf->getValue<std::string>("fileExtension");

    const size_t imageNumbLength = 6;

    for ( size_t i {0}; i < nFrames; i++)
    {
        std::string frameNumb = std::to_string(i);
        std::string frameStr = std::string(imageNumbLength - std::min(imageNumbLength, frameNumb.length()), '0') + frameNumb;
        leftImagesStr.emplace_back(leftPath + frameStr + fileExt);
        rightImagesStr.emplace_back(rightPath + frameStr + fileExt);
    }

    cv::Mat rectMap[2][2];
    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[1][0], rectMap[1][1]);
        std::cout << "mZedCamera->cameraRight.P " << mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3) << std::endl;
        std::cout << "mZedCamera->cameraRight.cameraMatrix " << mZedCamera->cameraRight.cameraMatrix << std::endl;

    }

    Eigen::Matrix4d prevWPose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prevWPoseInv = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d currCameraPose = Eigen::Matrix4d::Identity();

    for ( size_t i{0}; i < nFrames; i++)
    {

        // if ( i%3 != 0 && i != 0  && i != 1 )
        //     continue;

        cv::Mat imageLeft = cv::imread(leftImagesStr[i],cv::IMREAD_COLOR);
        cv::Mat imageRight = cv::imread(rightImagesStr[i],cv::IMREAD_COLOR);
        // std::cout << "channels" <<imageLeft.channels() << std::endl;

        cv::Mat imLRect, imRRect;

        if ( !mZedCamera->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
            cv::remap(imageRight, imRRect, rectMap[1][0], rectMap[1][1], cv::INTER_LINEAR);
            cv::imshow("right rect", imRRect);
            cv::imshow("left rect", imLRect);
            cv::imshow("right", imageRight);
            cv::imshow("left", imageLeft);
            cv::waitKey(1);
        }
        else
        {
            imLRect = imageLeft.clone();
            imRRect = imageRight.clone();
        }

        std::vector<MapPoint*> activeMpsTemp;

        std::vector<bool> MPsOutliers;
        std::vector<bool> MPsMatches;

        Eigen::Matrix4d estimPose = featTracker->TrackImage(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, i);

        setActiveOutliers(activeMpsTemp,MPsOutliers, MPsMatches);

        currCameraPose = estimPose;

        prevWPose = mZedCamera->cameraPose.pose;
        prevWPoseInv = mZedCamera->cameraPose.poseInverse;
        // referencePose = lastKFPoseInv * poseEst;
        // Eigen::Matrix4d lKFP = activeKeyFrames.front()->pose.pose;
        // zedPtr->cameraPose.setPose(referencePose, lKFP);
        mZedCamera->cameraPose.setPose(estimPose);
        mZedCamera->cameraPose.setInvPose(estimPose.inverse());
        predNPose = estimPose * (prevWPoseInv * estimPose);
        predNPoseInv = predNPose.inverse();

    }


    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb, map);




    Visual->join();
    Tracking->join();
    LocalMapping->join();
}

} // namespace vio_slam