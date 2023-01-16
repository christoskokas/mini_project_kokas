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
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.cameraMatrix, mZedCamera->cameraLeft.distCoeffs, R1, mZedCamera->cameraLeft.cameraMatrix, cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.cameraMatrix, mZedCamera->cameraRight.distCoeffs, R2, mZedCamera->cameraRight.cameraMatrix, cv::Size(width, height), CV_32F, rectMap[1][0], rectMap[1][1]);
    }



    for ( size_t i{0}; i < nFrames; i++)
    {
        cv::Mat imageLeft = cv::imread(leftImagesStr[i],cv::IMREAD_COLOR);
        cv::Mat imageRight = cv::imread(rightImagesStr[i],cv::IMREAD_COLOR);

        cv::Mat imLRect, imRRect;

        if ( !mZedCamera->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
            cv::remap(imageRight, imRRect, rectMap[1][0], rectMap[1][1], cv::INTER_LINEAR);
        }
        else
        {
            imLRect = imageLeft.clone();
            imRRect = imageRight.clone();
        }

        featTracker->TrackImage(imLRect, imRRect, i);

    }




    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb, map);




    Visual->join();
    Tracking->join();
    LocalMapping->join();
}

} // namespace vio_slam