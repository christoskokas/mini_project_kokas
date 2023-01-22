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

void System::insertKF(KeyFrame* kF, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsN, const Eigen::Matrix4d& estimPose, const int nStereo, const int nMono)
{
    const double fx {mZedCamera->cameraLeft.fx};
    const double fy {mZedCamera->cameraLeft.fy};
    const double cx {mZedCamera->cameraLeft.cx};
    const double cy {mZedCamera->cameraLeft.cy};
    TrackedKeys& keysLeft = kF->keys;
    kF->keyF = true;
    kF->unMatchedF.resize(keysLeft.keyPoints.size(), -1);
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
    activeMapPoints.reserve(activeMapPoints.size() + keysLeft.keyPoints.size());
    std::lock_guard<std::mutex> lock(map->mapMutex);
    int trckedKeys {0};
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        // if ( keysLeft.close[i] >  )
        if ( matchedIdxsN[i] >= 0 )
        {
            MapPoint* mp = activeMapPoints[matchedIdxsN[i]];
            if ( mp->GetIsOutlier() || !mp->GetInFrame() )
                continue;
            mp->kFWithFIdx.insert(std::pair<KeyFrame*, size_t>(kF, i));
            mp->desc.push_back(keysLeft.Desc.row(i));
            if ( keysLeft.estimatedDepth[i] > 0 )
                mp->desc.push_back(keysLeft.rightDesc.row(keysLeft.rightIdxs[i]));
            mp->addTCnt();
            kF->localMapPoints[i] = mp;
            kF->unMatchedF[i] = mp->kdx;
            trckedKeys++;
            continue;
        }
        if ( nStereo > minNStereo )
            continue;
        if ( keysLeft.close[i] || ( nMono <= minNMono && keysLeft.estimatedDepth[i] > 0) )
        {
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = estimPose * p;
            MapPoint* mp = new MapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], keysLeft.close[i], map->kIdx, map->pIdx);
            mp->desc.push_back(keysLeft.rightDesc.row(keysLeft.rightIdxs[i]));
            mp->kFWithFIdx.insert(std::pair<KeyFrame*, size_t>(kF, i));
            // kF->unMatchedF[i] = false;
            // kF->localMapPoints.emplace_back(mp);
            kF->localMapPoints[i] = mp;
            // kF->unMatchedF[i] = mp->kdx;
            // if ( keysLeft.close[i] )
            // {
            mp->added = true;
            activeMapPoints.emplace_back(mp);
            map->addMapPoint(mp);
            trckedKeys ++;
            // }
        }
    }
    kF->nKeysTracked = trckedKeys;
    // kF->keys.getKeys(keysLeft);
    map->addKeyFrame(kF);
    map->activeKeyFrames.insert(map->activeKeyFrames.begin(),kF);
    // if ( activeKeyFrames.size() > actvKFMaxSize )
    // {
    //     // removeKeyFrame(activeKeyFrames);
    //     activeKeyFrames.back()->active = false;
    //     activeKeyFrames.resize(actvKFMaxSize);
    // }
    // referencePose = Eigen::Matrix4d::Identity();
    // zedPtr->cameraPose.refPose = referencePose;
    if ( map->activeKeyFrames.size() > 3 )
        map->keyFrameAdded = true;
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
        // std::cout << "mZedCamera->cameraRight.P " << mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3) << std::endl;
        // std::cout << "mZedCamera->cameraRight.cameraMatrix " << mZedCamera->cameraRight.cameraMatrix << std::endl;

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
            // cv::imshow("right rect", imRRect);
            // cv::imshow("left rect", imLRect);
            // cv::imshow("right", imageRight);
            // cv::imshow("left", imageLeft);
            // cv::waitKey(1);
        }
        else
        {
            imLRect = imageLeft.clone();
            imRRect = imageRight.clone();
        }

        std::vector<MapPoint*> activeMpsTemp;

        std::vector<bool> MPsOutliers;
        std::vector<bool> MPsMatches;
        std::vector<int> matchedIdxsN;


        int nStereo {0}, nMono {0};
        bool newKF {false};
        std::pair<KeyFrame*, Eigen::Matrix4d> trckF;

        // Eigen::Matrix4d estimPose = featTracker->TrackImage(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, i, newKF);
        trckF = featTracker->TrackImageT(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, newKF, i, matchedIdxsN, nStereo, nMono);
        Eigen::Matrix4d estimPose = trckF.second;
        KeyFrame* kFCandF = trckF.first;

        setActiveOutliers(activeMpsTemp,MPsOutliers, MPsMatches);

        if ( newKF )
        {
            insertKF(kFCandF, map->activeMapPoints,matchedIdxsN, estimPose, nStereo, nMono);
            map->keyFrameAddedMain = true;

        }

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