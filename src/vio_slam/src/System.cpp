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

    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight, feLeft->getGridRows(), feLeft->getGridCols());

    localMap = new LocalMapper(map, mZedCamera, fm);

    Visual = new std::thread(&vio_slam::ViewFrame::pangoQuit, mFrame, mZedCamera, map);

    LocalMapping = new std::thread(&vio_slam::LocalMapper::beginLocalMapping, localMap);
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

    mZedCamera = new Zed_Camera(mConf);
    mZedCameraB = new Zed_Camera(mConf);

    mFrame = new ViewFrame;

    map = new Map();
    mapB = new Map();

    featTracker = new FeatureTracker(mZedCamera,map);
    featTrackerB = new FeatureTracker(mZedCameraB,mapB);

    feLeft = new FeatureExtractor(nFeatures/2);
    feRight = new FeatureExtractor(nFeatures/2);

    feLeftB = new FeatureExtractor(nFeatures/2);
    feRightB = new FeatureExtractor(nFeatures/2);

    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight, feLeft->getGridRows(), feLeft->getGridCols());
    fmB = new FeatureMatcher(mZedCameraB, feLeftB, feRightB, mZedCameraB->mHeight, feLeftB->getGridRows(), feLeftB->getGridCols());

    localMap = new LocalMapper(map, mZedCamera, fm, mapB, mZedCameraB, fmB);

    Visual = new std::thread(&vio_slam::ViewFrame::pangoQuit, mFrame, mZedCamera, map);

    LocalMapping = new std::thread(&vio_slam::LocalMapper::beginLocalMappingB, localMap);
    // Visual = new std::thread(&vio_slam::Frame::pangoQuit, mFrame, mZedCamera);

    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb);

    // vio_slam::Frame frame;
    // vio_slam::RobustMatcher2 rb(zedptr);
    // std::thread worker(&vio_slam::Frame::pangoQuit, frame, zedptr);
    // std::thread tester(&vio_slam::RobustMatcher2::beginTest, &rb);
}

void System::setActiveOutliers(Map* map, std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<bool>& MPsMatches)
{
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for ( size_t i{0}, end{MPsOutliers.size()}; i < end; i++)
    {
        MapPoint*& mp = activeMPs[i];
        if ( MPsMatches[i] )
            mp->unMCnt = 0;
        else
            mp->unMCnt++;
        if ( !MPsOutliers[i] && mp->unMCnt < 20 )
        {
            continue;
        }
        // mp->desc.release();
        mp->SetIsOutlier( true );
        // std::unordered_map<KeyFrame*,size_t>::iterator it;
        // std::unordered_map<KeyFrame*,size_t>::const_iterator endmp(mp->kFWithFIdx.end());
        // for ( it = mp->kFWithFIdx.begin(); it != endmp; it++)
        // {
        //     KeyFrame* kF = it->first;
        //     size_t keyPos = it->second;
        //     kF->eraseMPConnection(keyPos);
        //     // kF->localMapPoints[keyPos] = nullptr;
        //     // kF->unMatchedF[keyPos] = -1;
        // }
        // delete mp;
        // mp = nullptr;
    }
}

void System::insertKF(Map* map, KeyFrame* kF, std::vector<int>& matchedIdxsN, const Eigen::Matrix4d& estimPose, const int nStereo, const int nMono, const bool front, const bool addKF)
{
    const double fx {mZedCamera->cameraLeft.fx};
    const double fy {mZedCamera->cameraLeft.fy};
    const double cx {mZedCamera->cameraLeft.cx};
    const double cy {mZedCamera->cameraLeft.cy};
    TrackedKeys& keysLeft = kF->keys;
    if ( addKF )
        kF->keyF = true;
    kF->unMatchedF.resize(keysLeft.keyPoints.size(), -1);
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
    std::vector<MapPoint*>& activeMapPoints = map->activeMapPoints;
    activeMapPoints.reserve(activeMapPoints.size() + keysLeft.keyPoints.size());
    std::lock_guard<std::mutex> lock(map->mapMutex);
    int trckedKeys {0};
    int newStereoKeys{0};
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        // if ( keysLeft.close[i] >  )
        if ( matchedIdxsN[i] >= 0 )
        {
            MapPoint* mp = activeMapPoints[matchedIdxsN[i]];
            if ( mp->GetIsOutlier() || !mp->GetInFrame() || !mp)
            {
                matchedIdxsN[i] = -1;
                continue;
            }
            mp->kFWithFIdx.insert(std::pair<KeyFrame*, size_t>(kF, i));
            mp->desc.push_back(keysLeft.Desc.row(i));
            if ( keysLeft.estimatedDepth[i] > 0 )
                mp->desc.push_back(keysLeft.rightDesc.row(keysLeft.rightIdxs[i]));
            mp->addTCnt();
            kF->localMapPoints[i] = mp;
            mp->lastObsKF = kF;
            mp->lastObsL = keysLeft.keyPoints[i];
            kF->unMatchedF[i] = mp->kdx;
            trckedKeys++;
            continue;
        }
        if ( nStereo > minNStereo )
            continue;
        if ( keysLeft.close[i] /* && newStereoKeys < 100 *//*  || ( nMono <= minNMono && keysLeft.estimatedDepth[i] > 0)  */)
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
            mp->lastObsKF = kF;
            mp->lastObsL = keysLeft.keyPoints[i];
            // kF->unMatchedF[i] = mp->kdx;
            // if ( keysLeft.close[i] )
            // {
            mp->added = true;
            activeMapPoints.emplace_back(mp);
            map->addMapPoint(mp);
            trckedKeys ++;
            newStereoKeys++;
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
    // if ( map->activeKeyFrames.size() > 3 && front )
    //     map->keyFrameAdded = true;
}

void System::drawTrackedKeys(KeyFrame* kF, std::vector<int> matched, const char* com, cv::Mat& im)
{
    TrackedKeys& tKeys = kF->keys;
    std::vector<cv::KeyPoint> keys;
    keys.reserve(tKeys.keyPoints.size());
    for ( size_t i{0}, end{ tKeys.keyPoints.size()}; i < end; i++)
    {
        if ( matched[i] >= 0 )
            keys.emplace_back(tKeys.keyPoints[i]);
    }

    cv::Mat outIm = im.clone();
    for (auto& key:keys)
    {
        cv::circle(outIm, key.pt,2,cv::Scalar(0,255,0));

    }
    cv::imshow(com, outIm);
    cv::waitKey(1);
}

void System::SLAM()
{
    

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
        // trckF = std::thread lel(&vio_slam::FeatureTracker::TrackImageT,)

        Eigen::Matrix4d estimPose = Eigen::Matrix4d::Identity();
        KeyFrame* kFCandF;

        // featTracker->TrackImageT(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, newKF, i, matchedIdxsN, nStereo, nMono, kFCandF, estimPose);
        FeatTrack = new std::thread(&vio_slam::FeatureTracker::TrackImageT, featTracker, std::ref(imLRect), std::ref(imRRect), std::ref(currCameraPose), std::ref(predNPoseInv), std::ref(activeMpsTemp), std::ref(MPsOutliers), std::ref(MPsMatches), std::ref(newKF), std::ref(i), std::ref(matchedIdxsN), std::ref(nStereo), std::ref(nMono), std::ref(kFCandF), std::ref(estimPose));
        FeatTrack->join();
        // Eigen::Matrix4d estimPose = trckF.second;
        // KeyFrame* kFCandF = trckF.first;

        setActiveOutliers(map, activeMpsTemp,MPsOutliers, MPsMatches);

        if ( i == 0 )
        {
            
            continue;
        }

        if ( newKF )
        {
            insertKF(map, kFCandF, matchedIdxsN, estimPose, nStereo, nMono, true, true);
            if ( map->activeKeyFrames.size() > 4 )
                map->keyFrameAdded = true;

        }

        // drawTrackedKeys(kFCandF, matchedIdxsN, "tracked Keys", imLRect);

        // currCameraPose = estimPose;

        // prevWPose = mZedCamera->cameraPose.pose;
        // prevWPoseInv = mZedCamera->cameraPose.poseInverse;
        // // referencePose = lastKFPoseInv * poseEst;
        // // Eigen::Matrix4d lKFP = activeKeyFrames.front()->pose.pose;
        // // zedPtr->cameraPose.setPose(referencePose, lKFP);
        // mZedCamera->cameraPose.setPose(estimPose);
        // mZedCamera->cameraPose.setInvPose(estimPose.inverse());
        // predNPose = estimPose * (prevWPoseInv * estimPose);
        // predNPoseInv = predNPose.inverse();

    }


    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb, map);




    Visual->join();
    // Tracking->join();
    LocalMapping->join();
}

void System::MultiSLAM()
{
    

    const int nFrames {mZedCamera->numOfFrames};
    std::vector<std::string>leftImagesStr, rightImagesStr, leftImagesStrB, rightImagesStrB;
    leftImagesStr.reserve(nFrames);
    rightImagesStr.reserve(nFrames);
    leftImagesStrB.reserve(nFrames);
    rightImagesStrB.reserve(nFrames);

    const std::string imagesPath = mConf->getValue<std::string>("imagesPath");

    std::vector<float> Tc1_c2 = mConf->getValue<std::vector<float>>("Multi", "T_c1_c2", "data");

    Eigen::Matrix4d transfC1C2;
    transfC1C2 << Tc1_c2[0],Tc1_c2[1],Tc1_c2[2],Tc1_c2[3],Tc1_c2[4],Tc1_c2[5],Tc1_c2[6],Tc1_c2[7],Tc1_c2[8],Tc1_c2[9],Tc1_c2[10],Tc1_c2[11],Tc1_c2[12],Tc1_c2[13],Tc1_c2[14],Tc1_c2[15];

    Logging("Tc1c2", transfC1C2, 3);

    mZedCameraB->cameraPose.setPose(transfC1C2);
    Eigen::Matrix4d transfC1C2inv = transfC1C2.inverse();

    const std::string leftPath = imagesPath + "left/";
    const std::string rightPath = imagesPath + "right/";
    const std::string leftPathB = imagesPath + "leftBack/";
    const std::string rightPathB = imagesPath + "rightBack/";
    const std::string fileExt = mConf->getValue<std::string>("fileExtension");

    const size_t imageNumbLength = 6;

    for ( size_t i {0}; i < nFrames; i++)
    {
        std::string frameNumb = std::to_string(i);
        std::string frameStr = std::string(imageNumbLength - std::min(imageNumbLength, frameNumb.length()), '0') + frameNumb;
        leftImagesStr.emplace_back(leftPath + frameStr + fileExt);
        rightImagesStr.emplace_back(rightPath + frameStr + fileExt);
        leftImagesStrB.emplace_back(leftPathB + frameStr + fileExt);
        rightImagesStrB.emplace_back(rightPathB + frameStr + fileExt);
    }

    cv::Mat rectMap[2][2], rectMapB[2][2];
    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[1][0], rectMap[1][1]);
        cv::initUndistortRectifyMap(mZedCameraB->cameraLeft.K, mZedCameraB->cameraLeft.D, mZedCameraB->cameraLeft.R, mZedCameraB->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMapB[0][0], rectMapB[0][1]);
        cv::initUndistortRectifyMap(mZedCameraB->cameraRight.K, mZedCameraB->cameraRight.D, mZedCameraB->cameraRight.R, mZedCameraB->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMapB[1][0], rectMapB[1][1]);
        // std::cout << "mZedCamera->cameraRight.P " << mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3) << std::endl;
        // std::cout << "mZedCamera->cameraRight.cameraMatrix " << mZedCamera->cameraRight.cameraMatrix << std::endl;

    }

    Eigen::Matrix4d prevWPose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prevWPoseInv = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d currCameraPose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prevWPoseB = mZedCameraB->cameraPose.pose;
    Eigen::Matrix4d prevWPoseInvB = mZedCameraB->cameraPose.pose;
    Eigen::Matrix4d predNPoseB = mZedCameraB->cameraPose.pose;
    Eigen::Matrix4d predNPoseInvB = mZedCameraB->cameraPose.pose;
    Eigen::Matrix4d currCameraPoseB = mZedCameraB->cameraPose.pose;

    Sophus::Vector6d displacement;
    

    for ( size_t i{0}; i < nFrames; i++)
    {

        // if ( i%3 != 0 && i != 0  && i != 1 )
        //     continue;

        cv::Mat imageLeft = cv::imread(leftImagesStr[i],cv::IMREAD_COLOR);
        cv::Mat imageRight = cv::imread(rightImagesStr[i],cv::IMREAD_COLOR);
        cv::Mat imageLeftB = cv::imread(leftImagesStrB[i],cv::IMREAD_COLOR);
        cv::Mat imageRightB = cv::imread(rightImagesStrB[i],cv::IMREAD_COLOR);
        // std::cout << "channels" <<imageLeft.channels() << std::endl;

        cv::Mat imLRect, imRRect, imLRectB, imRRectB;

        if ( !mZedCamera->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
            cv::remap(imageRight, imRRect, rectMap[1][0], rectMap[1][1], cv::INTER_LINEAR);
            cv::remap(imageLeftB, imLRectB, rectMapB[0][0], rectMapB[0][1], cv::INTER_LINEAR);
            cv::remap(imageRightB, imRRectB, rectMapB[1][0], rectMapB[1][1], cv::INTER_LINEAR);
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
            imLRectB = imageLeftB.clone();
            imRRectB = imageRightB.clone();
        }

        std::vector<MapPoint*> activeMpsTemp;
        std::vector<MapPoint*> activeMpsTempB;

        std::vector<bool> MPsOutliers;
        std::vector<bool> MPsMatches;
        std::vector<int> matchedIdxsN;
        std::vector<bool> MPsOutliersB;
        std::vector<bool> MPsMatchesB;
        std::vector<int> matchedIdxsNB;


        int nStereo {0}, nMono {0};
        int nStereoB {0}, nMonoB {0};
        bool newKF {false};
        bool newKFB {false};

        // Eigen::Matrix4d estimPose = featTracker->TrackImage(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, i, newKF);
        // trckF = std::thread lel(&vio_slam::FeatureTracker::TrackImageT,)

        Eigen::Matrix4d estimPose = Eigen::Matrix4d::Identity();
        KeyFrame* kFCandF;
        Eigen::Matrix4d estimPoseB = Eigen::Matrix4d::Identity();
        KeyFrame* kFCandFB;

        // featTracker->TrackImageT(imLRect, imRRect, currCameraPose, predNPoseInv, activeMpsTemp, MPsOutliers, MPsMatches, newKF, i, matchedIdxsN, nStereo, nMono, kFCandF, estimPose);
        FeatTrack = new std::thread(&vio_slam::FeatureTracker::TrackImageT, featTracker, std::ref(imLRect), std::ref(imRRect), std::ref(currCameraPose), std::ref(predNPoseInv), std::ref(activeMpsTemp), std::ref(MPsOutliers), std::ref(MPsMatches), std::ref(newKF), std::ref(i), std::ref(matchedIdxsN), std::ref(nStereo), std::ref(nMono), std::ref(kFCandF), std::ref(estimPose));
        FeatTrackB = new std::thread(&vio_slam::FeatureTracker::TrackImageT, featTrackerB, std::ref(imLRectB), std::ref(imRRectB), std::ref(currCameraPoseB), std::ref(predNPoseInvB), std::ref(activeMpsTempB), std::ref(MPsOutliersB), std::ref(MPsMatchesB), std::ref(newKFB), std::ref(i), std::ref(matchedIdxsNB), std::ref(nStereoB), std::ref(nMonoB), std::ref(kFCandFB), std::ref(estimPoseB));
        FeatTrack->join();
        FeatTrackB->join();
        // Eigen::Matrix4d estimPose = trckF.second;
        // KeyFrame* kFCandF = trckF.first;

        setActiveOutliers(map, activeMpsTemp,MPsOutliers, MPsMatches);
        setActiveOutliers(mapB, activeMpsTempB,MPsOutliersB, MPsMatchesB);

        if ( i == 0 )
        {
            KeyFrame* KF = map->activeKeyFrames.front();
            KeyFrame* KFB = mapB->activeKeyFrames.front();
            KF->KFBack = KFB;
            KFB->KFBack = KF;
            KF->localMapPointsB = KFB->localMapPoints;
            continue;
        }

       

        if ( i == 1 )
        {
            prevV = estimPose.block<3,1>(0,3);
            prevVB = estimPoseB.block<3,1>(0,3);
            prevT = estimPose.block<3,1>(0,3);
            prevTB = estimPoseB.block<3,1>(0,3);
        }
        else
        {
            // if ( prevV.cwiseAbs().sum() > 0.05 )
            //     if (!checkDisplacement(estimPose, estimPoseB, predNPose, transfC1C2inv, transfC1C2))
            //     {
            //         Logging("Pose Estimation Failed, Retrying..","",3);
            //         continue;
            //     }
            // else
            // {
            //     prevV = (estimPose.block<3,1>(0,3) - prevT).cwiseAbs();
            //     prevT = estimPose.block<3,1>(0,3);
            //     prevVB = (estimPoseB.block<3,1>(0,3) - prevTB).cwiseAbs();
            //     prevTB = estimPoseB.block<3,1>(0,3);
            // }
        }

        Eigen::Matrix4d realPoseB = estimPoseB * transfC1C2inv;
        // Logging("Front Pose", estimPose,3);
        // Logging("Back Pose", estimPoseB,3);
        // Logging("realPoseB", realPoseB,3);

        Eigen::Matrix4d newPose = changePosesFromBoth(estimPose, realPoseB);

        estimPose = newPose;
        estimPoseB = newPose * transfC1C2;

        kFCandF->pose.setPose(estimPose);
        kFCandFB->pose.setPose(estimPoseB);

        currCameraPose = estimPose;
        currCameraPoseB = estimPoseB;

        if ( newKF /* || newKFB */)
        {
            insertKF(map, kFCandF, matchedIdxsN, estimPose, nStereo, nMono, true, true);
            insertKF(mapB, kFCandFB, matchedIdxsNB, estimPoseB, nStereoB, nMonoB, false, true);
            kFCandF->KFBack = kFCandFB;
            kFCandFB->KFBack = kFCandF;
            kFCandF->localMapPointsB = kFCandFB->localMapPoints;
            if ( map->activeKeyFrames.size() > 4 )
                map->keyFrameAdded = true;

        }
        else if ( newKFB )
        {
            insertKF(mapB, kFCandFB, matchedIdxsNB, estimPoseB, nStereoB, nMonoB, false, false);
        }
        drawTrackedKeys(kFCandF, matchedIdxsN, "tracked Keys", imLRect);

        // Eigen::Matrix4d poseDif = estimPose * mZedCamera->cameraPose.poseInverse;

        // Eigen::Matrix3d Rdif = poseDif.block<3,3>(0,0);
        // Eigen::Matrix<double,3,1> tdif =  poseDif.block<3,1>(0,3);

        // Sophus::SE3<double> se3t(Rdif, tdif);
        // // Sophus::SE3;

        // Sophus::Vector6d prevDisp = displacement;

        // displacement = se3t.log();




        prevWPose = mZedCamera->cameraPose.pose;
        prevWPoseInv = mZedCamera->cameraPose.poseInverse;
        mZedCamera->cameraPose.setPose(estimPose);
        mZedCamera->cameraPose.setInvPose(estimPose.inverse());
        predNPose = estimPose * (prevWPoseInv * estimPose);
        predNPoseInv = predNPose.inverse();

        prevWPoseB = mZedCameraB->cameraPose.pose;
        prevWPoseInvB = mZedCameraB->cameraPose.poseInverse;
        mZedCameraB->cameraPose.setPose(estimPoseB);
        mZedCameraB->cameraPose.setInvPose(estimPoseB.inverse());
        predNPoseB = estimPoseB * (prevWPoseInvB * estimPoseB);
        predNPoseInvB = predNPoseB.inverse();

    }


    // Tracking = new std::thread(&vio_slam::RobustMatcher2::beginTest, mRb, map);




    Visual->join();
    // Tracking->join();
    LocalMapping->join();
}

bool System::checkDisplacement(Eigen::Matrix4d& estimPose, Eigen::Matrix4d& estimPoseB, Eigen::Matrix4d& predPose, Eigen::Matrix4d& transfC1C2Inv, Eigen::Matrix4d& transfC1C2)
{
    Eigen::Vector3d newV = (estimPose.block<3,1>(0,3) - prevT).cwiseAbs();
    Eigen::Vector3d newVB = (estimPoseB.block<3,1>(0,3) - prevTB).cwiseAbs();
    double perc = ((newV - prevV).cwiseAbs()).sum()/newV.sum();
    double percB = ((newVB - prevVB).cwiseAbs()).sum()/newVB.sum();
    if ( perc > maxPerc && percB > maxPerc )
    {
        return false;
        estimPose = predPose;
        estimPoseB = estimPose * transfC1C2;
    }
    else if ( perc > maxPerc )
    {
        estimPose = estimPoseB * transfC1C2Inv;
    }
    else if ( percB > maxPerc )
    {
        estimPoseB = estimPose * transfC1C2;
    }
    prevV = (estimPose.block<3,1>(0,3) - prevT).cwiseAbs();
    prevT = estimPose.block<3,1>(0,3);
    prevVB = (estimPoseB.block<3,1>(0,3) - prevTB).cwiseAbs();
    prevTB = estimPoseB.block<3,1>(0,3);
    return true;
}

Eigen::Matrix4d System::changePosesFromBoth(Eigen::Matrix4d& estimPose, Eigen::Matrix4d& estimPoseB)
{
    Eigen::Matrix3d Rot = estimPose.block<3,3>(0,0);
    Eigen::Vector3d t = estimPose.block<3,1>(0,3);
    Eigen::Quaterniond q(Rot);
    Eigen::Matrix3d RotB = estimPoseB.block<3,3>(0,0);
    Eigen::Vector3d tB = estimPoseB.block<3,1>(0,3);
    Eigen::Quaterniond qB(RotB);
    Eigen::Quaterniond qres = q.slerp(0.5, qB);
    Eigen::Vector3d tnew = (t + tB)/2;

    Eigen::Matrix4d newPose = Eigen::Matrix4d::Identity();
    newPose.block<3,3>(0,0) = qres.toRotationMatrix();
    newPose.block<3,1>(0,3) = tnew;

    // Logging("NEWPOSE", newPose,3);
    return newPose;
    
}

} // namespace vio_slam