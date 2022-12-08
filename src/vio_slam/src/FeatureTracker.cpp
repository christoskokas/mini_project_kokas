#include "FeatureTracker.h"

namespace vio_slam
{

void ImageData::setImage(const int frameNumber, const char* whichImage, const std::string& seq)
{
    std::string imagePath;
    std::string first;
    std::string second, format;
    std::string t = whichImage;
#if KITTI_DATASET
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/kitti/" + seq + "/";
    second = "/00";
    format = ".png";
#elif ZED_DATASET
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/zed_exp/";
    second = "/" + t + "00";
    format = ".png";
#else
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/";
    second = "/frame";
    format = ".jpg";
#endif

    if (frameNumber > 999)
    {
        imagePath = first + t + second + std::to_string(frameNumber/(int)(pow(10,3))%10) + std::to_string(frameNumber/(int)(pow(10,2))%10) + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
        int i{};
    }
    else if (frameNumber > 99)
    {
        imagePath = first + t + second + "0" + std::to_string(frameNumber/(int)(pow(10,2))%10) + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
    }
    else if (frameNumber > 9)
    {
        imagePath = first + t + second + "00" + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
    }
    else
    {
        imagePath = first + t + second + "000" + std::to_string(frameNumber) + format;
    }
    im = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    rIm = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void ImageData::rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2)
{
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
}

FeatureData::FeatureData(Zed_Camera* _zedPtr) : zedPtr(_zedPtr), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

void FeatureData::compute3DPoints(SubPixelPoints& prePnts, const int keyNumb)
{
    const size_t end{prePnts.left.size()};

    const size_t start{prePnts.points3D.size()};

    prePnts.points3D.reserve(end);
    for (size_t i = start; i < end; i++)
    {   

        const double zp = (double)prePnts.depth[i];
        const double xp = (double)(((double)prePnts.left[i].x-cx)*zp/fx);
        const double yp = (double)(((double)prePnts.left[i].y-cy)*zp/fy);
        Eigen::Vector4d p4d(xp,yp,zp,1);
        p4d = zedPtr->cameraPose.pose * p4d;
        prePnts.points3D.emplace_back(p4d(0),p4d(1),p4d(2));
        
    }
}

FeatureTracker::FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr, Map* _map) : zedPtr(_zedPtr), map(_map), fm(zedPtr, &feLeft, &feRight, zedPtr->mHeight,feLeft.getGridRows(), feLeft.getGridCols()), pE(zedPtr), fd(zedPtr), dt(1.0f/(double)zedPtr->mFps), lkal(dt), datafile(filepath), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{
    rmap[0][0] = _rmap[0][0];
    rmap[0][1] = _rmap[0][1];
    rmap[1][0] = _rmap[1][0];
    rmap[1][1] = _rmap[1][1];
    K(0,0) = fx;
    K(1,1) = fy;
    K(0,2) = cx;
    K(1,2) = cy;
}

void FeatureTracker::setMask(const SubPixelPoints& prePnts, cv::Mat& mask)
{
    // const int rad {3};
    mask = cv::Mat(zedPtr->mHeight, zedPtr->mWidth, CV_8UC1, cv::Scalar(255));

    std::vector<cv::Point2f>::const_iterator it, end{prePnts.left.end()};
    for (it = prePnts.left.begin();it != end; it++)
    {
        if (mask.at<uchar>(*it) == 255)
        {
            cv::circle(mask, *it, maskRadius, 0, cv::FILLED);
        }
    }

}

void FeatureTracker::setPopVec(const SubPixelPoints& prePnts, std::vector<int>& pop)
{
    const int gRows {fe.getGridRows()};
    const int gCols {fe.getGridCols()};
    pop.resize(gRows * gCols);
    const int wid {(int)zedPtr->mWidth/gCols + 1};
    const int hig {(int)zedPtr->mHeight/gRows + 1};
    std::vector<cv::Point2f>::const_iterator it, end(prePnts.left.end());
    for (it = prePnts.left.begin(); it != end; it ++)
    {
        const int w {(int)it->x/wid};
        const int h {(int)it->y/hig};
        pop[(int)(w + h*gCols)] += 1;
    }
}

void FeatureTracker::stereoFeaturesPop(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    std::vector<int> pop;
    setPopVec(prePnts, pop);
    fe.extractFeaturesPop(leftIm, rightIm, desc, keys, pop);
    fm.computeStereoMatches(leftIm, rightIm, desc, matches, pnts, keys);
    // std::vector<uchar> inliers;
    // if ( pnts.left.size() >  6)
    // {
    //     cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);

    //     pnts.reduce<uchar>(inliers);
    //     reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
    // }
    Logging("matches size", matches.size(),1);

#if KEYSIM
    drawKeys("left", pLIm.rIm, keys.left);
    drawKeys("right", pRIm.rIm, keys.right);
#endif


#if MATCHESIM
    drawMatches(pLIm.rIm, pnts, matches);
#endif
}

void FeatureTracker::stereoFeaturesMask(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;

    extractFAST(leftIm, rightIm, keys, desc, prePnts.left);

    fm.findStereoMatchesFAST(leftIm, rightIm, desc,pnts, keys);

#if KEYSIM
    drawKeys("left", pLIm.rIm, keys.left);
    drawKeys("right", pRIm.rIm, keys.right);
#endif

    Logging("matches size", pnts.left.size(),3);
#if MATCHESIM
    drawPointsTemp<cv::Point2f,cv::Point2f>("Matches",pLIm.rIm,pnts.left, pnts.right);
#endif
}

void FeatureTracker::stereoFeaturesClose(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;

    extractFAST(leftIm, rightIm, keys, desc, prePnts.left);

    // fm.findStereoMatchesClose(desc,pnts, keys);
    fm.findStereoMatchesFAST(leftIm, rightIm, desc,pnts, keys);

    // std::vector<uchar> inliers;
    // cv::Mat F = cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);
    // cv::correctMatches(F,pnts.left, pnts.right,pnts.left, pnts.right);

    // pnts.reduce<uchar>(inliers);
    // reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
#if MATCHESIM
    drawPointsTemp<cv::Point2f,cv::Point2f>("Matches",pLIm.rIm,pnts.left, pnts.right);
#endif
}

void FeatureTracker::extractORB(cv::Mat& leftIm, cv::Mat& rightIm, StereoKeypoints& keys, StereoDescriptors& desc)
{
    Timer orb("ORB");
    std::thread extractLeft(&FeatureExtractor::computeKeypoints, std::ref(feLeft), std::ref(leftIm), std::ref(keys.left), std::ref(prePnts.left), std::ref(desc.left), 0);
    std::thread extractRight(&FeatureExtractor::computeKeypoints, std::ref(feRight), std::ref(rightIm), std::ref(keys.right), std::ref(prePnts.left),std::ref(desc.right), 1);
    extractLeft.join();
    extractRight.join();
}

void FeatureTracker::extractORBStereoMatch(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft)
{
    Timer orb("ORB");
    std::vector<cv::KeyPoint> rightKeys, temp;
    cv::Mat rightDesc;
    std::thread extractLeft(&FeatureExtractor::extractKeysNew, std::ref(feLeft), std::ref(leftIm), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc));
    std::thread extractRight(&FeatureExtractor::extractKeysNew, std::ref(feRight), std::ref(rightIm), std::ref(rightKeys),std::ref(rightDesc));
    extractLeft.join();
    extractRight.join();



    fm.findStereoMatchesORB2(lIm.im, rIm.im, rightDesc, rightKeys, keysLeft);

    keysLeft.mapPointIdx.resize(keysLeft.keyPoints.size(), -1);
    keysLeft.trackCnt.resize(keysLeft.keyPoints.size(), 0);

    drawKeys("left Keys", lIm.rIm, keysLeft.keyPoints);

    drawKeyPointsCloseFar("new method", lIm.rIm, keysLeft, rightKeys);
}

void FeatureTracker::extractFAST(const cv::Mat& leftIm, const cv::Mat& rightIm, StereoKeypoints& keys, StereoDescriptors& desc, const std::vector<cv::Point2f>& prevPnts)
{
    Timer fast("FAST");
    std::thread extractLeft(&FeatureExtractor::computeFASTandDesc, std::ref(feLeft), std::ref(leftIm), std::ref(keys.left), std::ref(prevPnts), std::ref(desc.left));
    std::thread extractRight(&FeatureExtractor::computeFASTandDesc, std::ref(feRight), std::ref(rightIm), std::ref(keys.right), std::ref(prevPnts),std::ref(desc.right));
    // std::thread extractLeft(&FeatureExtractor::extractFeaturesMask, std::ref(feLeft), std::ref(leftIm), std::ref(keys.left), std::ref(desc.left));
    // std::thread extractRight(&FeatureExtractor::extractFeaturesMask, std::ref(feRight), std::ref(rightIm), std::ref(keys.right),std::ref(desc.right));
    extractLeft.join();
    extractRight.join();
}

void FeatureTracker::stereoFeatures(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    extractFAST(leftIm, rightIm, keys, desc, prePnts.left);

    // fm.findStereoMatches(desc,pnts, keys);
    fm.findStereoMatchesFAST(leftIm, rightIm, desc,pnts, keys);

    // reduceStereoKeys<bool>(stereoKeys, inliersL, inliersR);
    // fm.computeStereoMatches(leftIm, rightIm, desc, matches, pnts, keys);
    // std::vector<uchar> inliers;
    // cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);

    // pnts.reduce<uchar>(inliers);
    // reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
    Logging("matches size", matches.size(),1);
#if MATCHESIM
    drawPointsTemp<cv::Point2f,cv::Point2f>("Matches",lIm.rIm,pnts.left, pnts.right);
#endif
}

void FeatureTracker::stereoFeaturesGoodFeatures(cv::Mat& leftIm, cv::Mat& rightIm, SubPixelPoints& pnts, const SubPixelPoints& prePnts)
{
    const size_t gfcurCount {prePnts.left.size()};
    if ( curFrame != 0)
    {
        cv::Mat mask;
        setMask(prePnts, mask);
        cv::goodFeaturesToTrack(leftIm,pnts.left,gfmxCount - gfcurCount, 0.01, gfmnDist, mask);
    }
    else
        cv::goodFeaturesToTrack(leftIm,pnts.left,gfmxCount - gfcurCount, 0.01, gfmnDist);


    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(leftIm, rightIm, pnts.left, pnts.right, status, err, cv::Size(21,21), 1,criteria);

    reduceVectorTemp<cv::Point2f,uchar>(pnts.left,status);
    reduceVectorTemp<cv::Point2f,uchar>(pnts.right,status);

    std::vector<uchar>inliers;
    cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);

    pnts.reduce<uchar>(inliers);

    fm.slWinGF(leftIm, rightIm,pnts);

    Logging("matches size", pnts.left.size(),1);
#if MATCHESIM
    drawMatchesGoodFeatures(lIm.rIm, pnts);
#endif
}

void FeatureTracker::initializeTracking()
{
    // gridTraX.resize(gridVelNumb * gridVelNumb);
    // gridTraY.resize(gridVelNumb * gridVelNumb);
    startTime = std::chrono::high_resolution_clock::now();
    setLRImages(0);
    std::vector<cv::DMatch> matches;
    stereoFeatures(lIm.im, rIm.im, matches,pnts);
    cv::Mat rot = (cv::Mat_<double>(3,3) << 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
    uStereo = pnts.left.size();
    uMono = uStereo;
    pE.setPrevR(rot);
    cv::Mat tr = (cv::Mat_<double>(3,1) << 0.0,0.0,0.0);
    pE.setPrevT(tr);
    setPreInit();
    fd.compute3DPoints(prePnts, keyNumb);
    uStereo = prePnts.points3D.size();
    keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
    keyNumb++;
#if SAVEODOMETRYDATA
    saveData();
#endif
    // addFeatures = checkFeaturesArea(prePnts);
}

void FeatureTracker::initializeTrackingGoodFeatures()
{
    // gridTraX.resize(gridVelNumb * gridVelNumb);
    // gridTraY.resize(gridVelNumb * gridVelNumb);
    startTime = std::chrono::high_resolution_clock::now();
    setLRImages(0);
    std::vector<cv::DMatch> matches;
    stereoFeaturesGoodFeatures(lIm.im, rIm.im,pnts, prePnts);
    setPreInit();
    fd.compute3DPoints(prePnts, keyNumb);
    uStereo = prePnts.points3D.size();
    keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
    keyNumb++;
#if SAVEODOMETRYDATA
    saveData();
#endif
    // addFeatures = checkFeaturesArea(prePnts);
}

void FeatureTracker::beginTracking(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        curFrame = frame;
        setLRImages(frame);
        if (addFeatures || uStereo < mnSize)
        {
            zedPtr->addKeyFrame = true;
            updateKeys(frame);
            fd.compute3DPoints(prePnts, keyNumb);
            keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
            keyNumb ++;
            
        }
        opticalFlow();

        // Logging("addf", addFeatures,3);
        Logging("ustereo", uStereo,3);

        // getSolvePnPPoseWithEss();

        // getPoseCeres();
        getPoseCeresNew();

        setPre();

        addFeatures = checkFeaturesAreaCont(prePnts);
    }
    datafile.close();
}

void FeatureTracker::beginTrackingTrial(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        curFrame = frame;
        setLRImages(frame);
        fm.checkDepthChange(pLIm.im,pRIm.im,prePnts);
        if ( addFeatures || uStereo < mnSize )
        {
            zedPtr->addKeyFrame = true;
            updateKeys(frame);
            fd.compute3DPoints(prePnts, keyNumb);
            keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
            keyNumb ++;
            
        }
        
        // opticalFlow();
        opticalFlowPredict();

        // Logging("addf", addFeatures,3);

        // getSolvePnPPoseWithEss();

        // getPoseCeres();
        getPoseCeresNew();

        setPreTrial();

        addFeatures = checkFeaturesArea(prePnts);
        // addFeatures = checkFeaturesAreaCont(prePnts);
        Logging("ustereo", uStereo,3);
        Logging("umono", uMono,3);
    }
    datafile.close();
}

void FeatureTracker::beginTrackingTrialClose(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        curFrame = frame;
        setLRImages(frame);
        // fm.checkDepthChange(pLIm.im,pRIm.im,prePnts);
        if ( (addFeatures || uStereo < mnSize || cv::norm(pTvec)*zedPtr->mFps > highSpeed) && ( uStereo < mxSize) )
        {
            // Logging("ptvec",pTvec,3);
            // Logging("cv::norm(pTvec)",cv::norm(pTvec),3);

            zedPtr->addKeyFrame = true;
            if ( uMono > mxMonoSize )
                updateKeysClose(frame);
            else
                updateKeys(frame);
            fd.compute3DPoints(prePnts, keyNumb);
            keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
            keyNumb ++;
            
        }
        // std::vector<cv::Point3d> p3D;
        // pointsInFrame(p3D);
        

        // std::vector<cv::Point2f> pPnts, curPnts;

        // optFlow(p3D, pPnts, curPnts);

        // estimatePose(p3D,curPnts);

        if ( curFrame == 1 )
            opticalFlow();
        else
            opticalFlowPredict();
        getPoseCeresNew();


        // Logging("addf", addFeatures,3);

        // getSolvePnPPoseWithEss();

        // getPoseCeres();

        setPreTrial();

        addFeatures = checkFeaturesArea(prePnts);
        // addFeatures = checkFeaturesAreaCont(prePnts);
        Logging("ustereo", uStereo,3);
        Logging("umono", uMono,3);
    }
    datafile.close();
}

void FeatureTracker::findStereoFeatures(cv::Mat& leftIm, cv::Mat& rightIm, SubPixelPoints& pnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    std::vector<cv::Point2f> temp;

    // extractORB(leftIm, rightIm, keys, desc);
    // fm.findStereoMatches(desc,pnts, keys);


    extractFAST(leftIm, rightIm, keys, desc, temp);
    fm.findStereoMatchesFAST(leftIm, rightIm, desc,pnts, keys);

    // float min {leftIm.cols};
    // float max {0};
    // std::vector<bool>inBox(pnts.left.size(),true);
    // std::vector<cv::Point2f>::const_iterator it, end(pnts.left.end());
    // for (it = pnts.left.begin(); it != end; it++)
    // {
    //     const float& px = it->x;
    //     if ( px > max )
    //         max = px;
    //     else if (px < min)
    //         min = px;
    // }

    // float difMax {leftIm.cols - max};
    // float maxDist {0};
    // if ( min > difMax )
    //     maxDist = min;
    // else
    //     maxDist = difMax;

    // float maxDistR = leftIm.cols - maxDist;

    // size_t boxC {0};

    // for (it = pnts.left.begin(); it != end; it++, boxC ++)
    // {
    //     const float& px = it->x;
    //     if ( px < maxDist || px > maxDistR)
    //         inBox[boxC] = false;
    // }
    // pnts.reduce<bool>(inBox);

    Logging("matches size", pnts.left.size(),3);
#if MATCHESIM
    drawPointsTemp<cv::Point2f,cv::Point2f>("Matches",lIm.rIm,pnts.left, pnts.right);
#endif
}

void FeatureTracker::triangulate3DPoints(SubPixelPoints& pnts)
{
    const size_t end{pnts.left.size()};

    pnts.points3D.reserve(end);
    for (size_t i = 0; i < end; i++)
    {

        const double zp = (double)pnts.depth[i];
        const double xp = (double)(((double)pnts.left[i].x-cx)*zp/fx);
        const double yp = (double)(((double)pnts.left[i].y-cy)*zp/fy);
        pnts.points3D.emplace_back(xp, yp, zp);
        // Eigen::Vector4d p4d(xp,yp,zp,1);
        // p4d = zedPtr->cameraPose.pose * p4d;
        // pnts.points3D.emplace_back(p4d(0),p4d(1),p4d(2));

    }
}

void FeatureTracker::setPre3DPnts(SubPixelPoints& prePnts, SubPixelPoints& pnts)
{
    const size_t end {pnts.points3D.size()};
    const size_t res { end + prePnts.points3D.size()};

    cv::Mat mask;
    setMask(prePnts, mask);

    prePnts.points3D.reserve(res);
    prePnts.left.reserve(res);

    for ( size_t iP = 0; iP < end; iP++ )
    {
        if (mask.at<uchar>(pnts.left[iP]) == 0)
            continue;
        const double x = pnts.points3D[iP].x;
        const double y = pnts.points3D[iP].y;
        const double z = pnts.points3D[iP].z;
        Eigen::Vector4d p4d(x,y,z,1);
        p4d = zedPtr->cameraPose.pose * p4d;
        prePnts.points3D.emplace_back(p4d(0), p4d(1), p4d(2));
        prePnts.left.emplace_back(pnts.left[iP]);
    }
}

void FeatureTracker::setPreviousValuesIni()
{
    setPreLImage();
    setPreRImage();
    setPre3DPnts(prePnts, pnts);
    pnts.clear();
}

void FeatureTracker::setPreviousValues()
{
    setPreLImage();
    setPreRImage();
    prePnts.left = prePnts.newPnts;
    // fm.checkDepthChange(pLIm.im, pRIm.im,prePnts);
    setPre3DPnts(prePnts, pnts);
    pnts.clear();
    prePnts.newPnts.clear();
}

bool FeatureTracker::inBorder(cv::Point3d& p3d, cv::Point2d& p2d)
{
    Eigen::Vector4d point(p3d.x, p3d.y, p3d.z,1);
    point = zedPtr->cameraPose.poseInverse * point;
    const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;
    const double invZ = 1.0f/pointZ;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};


    const int min {0};
    const int maxW {zedPtr->mWidth};
    const int maxH {zedPtr->mHeight};

    if ( u < min || u > maxW || v < min || v > maxH)
        return false;

    p3d = cv::Point3d(pointX, pointY, pointZ);

    p2d = cv::Point2d(u,v);

    return true;
}

void FeatureTracker::checkInBorder(SubPixelPoints& pnts)
{
    const size_t end {pnts.points3D.size()};
    std::vector<bool> in;
    in.resize(end,false);
    pnts.points3DCurr = pnts.points3D;
    for (size_t i{0}; i < end; i++)
    {
        cv::Point2d pd((double)pnts.left[i].x, (double)pnts.left[i].y);
        if ( inBorder(pnts.points3DCurr[i], pd) )
        {
            cv::Point2f pf((float)pd.x, (float)pd.y);
            // if ( pointsDist(pf,pnts.left[i]) <= 16.0 )
            // {
                // pnts.left[i] = pf;
                in[i] = true;
            // }
        }

    }
    pnts.reduce<bool>(in);
}

void FeatureTracker::calcOpticalFlow(SubPixelPoints& pnts)
{
    Timer optical("optical");
    std::vector<float> err, err1;
    std::vector <uchar>  inliers, inliers2;
    std::vector<cv::Point3d> p3D;
    cv::Mat fIm, sIm, rightIm;
    fIm = pLIm.im;
    sIm = lIm.im;
    rightIm = rIm.im;
    checkInBorder(pnts);



    if ( curFrame == 1 )
    {
        cv::calcOpticalFlowPyrLK(fIm, sIm, pnts.left, pnts.newPnts, inliers, err1,cv::Size(21,21),3, criteria);
    }
    else
    {
        predictNewPnts(pnts, false);
// #if OPTICALIM
//     if ( new3D )
//         drawOptical("pred new", pLIm.rIm,pnts.left, pnts.newPnts);
//     else
//         drawOptical("pred old", pLIm.rIm,pnts.left, pnts.newPnts);
// #endif
        cv::calcOpticalFlowPyrLK(fIm, sIm, pnts.left, pnts.newPnts, inliers, err,cv::Size(21,21),3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    }
    std::vector<cv::Point2f> temp = pnts.left;
    cv::calcOpticalFlowPyrLK(sIm, fIm, pnts.newPnts, temp, inliers2, err,cv::Size(21,21),3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i {0}; i < pnts.left.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],pnts.left[i]) <= 0.25)
            inliers[i] = true;
        else
            inliers[i] = false;
    }

    pnts.reduce<uchar>(inliers);

    inliers.clear();
    inliers2.clear();

    cv::calcOpticalFlowPyrLK(lIm.im, rightIm, pnts.newPnts, pnts.right, inliers, err,cv::Size(21,21),3, criteria);

    temp = pnts.newPnts;
    cv::calcOpticalFlowPyrLK(rightIm, lIm.im, pnts.right, temp, inliers2, err,cv::Size(21,21),3, criteria);

    pnts.depth = std::vector<float>(pnts.newPnts.size(),-1.0f);
    pnts.points3DStereo = std::vector<cv::Point3d>(pnts.newPnts.size(),cv::Point3d(0,0,-1.0f));

    for (size_t i {0}; i < pnts.newPnts.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],pnts.newPnts[i]) <= 0.25 && abs(pnts.newPnts[i].y - pnts.right[i].y) < 2.0f)
        {
            const double zp = fx * zedPtr->mBaseline/((double)pnts.newPnts[i].x - (double)pnts.right[i].x);
            if ( zp > 0.0f )
            {
                const double xp = (double)(((double)pnts.newPnts[i].x-cx)*zp/fx);
                const double yp = (double)(((double)pnts.newPnts[i].y-cy)*zp/fy);
                pnts.points3DStereo[i] = cv::Point3d(xp, yp, zp);
            }
        }
            // pnts.depth[i] = pnts.newPnts[i].x - pnts.right[i].x;
    }



    // inliers.clear();
    // cv::findFundamentalMat(pnts.left, pnts.newPnts,inliers, cv::FM_RANSAC, 2, 0.99);

    // pnts.reduce<uchar>(inliers);


}

void FeatureTracker::calcOptical(SubPixelPoints& pnts, const bool new3D)
{
    Timer optical("optical");
    std::vector<float> err, err1;
    std::vector <uchar>  inliers, inliers2;
    std::vector<cv::Point3d> p3D;
    cv::Mat fIm, sIm, rightIm;
    if ( new3D )
    {
        fIm = lIm.im;
        sIm = pLIm.im;
        rightIm = pRIm.im;
        pnts.points3DCurr = pnts.points3D;
    }
    else
    {
        fIm = pLIm.im;
        sIm = lIm.im;
        rightIm = rIm.im;
        checkInBorder(pnts);
    }



    if ( curFrame == 1 )
    {
        cv::calcOpticalFlowPyrLK(fIm, sIm, pnts.left, pnts.newPnts, inliers, err1,cv::Size(21,21),3, criteria);
    }
    else
    {
        predictNewPnts(pnts, new3D);
// #if OPTICALIM
//     if ( new3D )
//         drawOptical("pred new", pLIm.rIm,pnts.left, pnts.newPnts);
//     else
//         drawOptical("pred old", pLIm.rIm,pnts.left, pnts.newPnts);
// #endif
        cv::calcOpticalFlowPyrLK(fIm, sIm, pnts.left, pnts.newPnts, inliers, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    }
    std::vector<cv::Point2f> temp = pnts.left;
    cv::calcOpticalFlowPyrLK(sIm, fIm, pnts.newPnts, temp, inliers2, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i {0}; i < pnts.left.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],pnts.left[i]) <= 0.25)
            inliers[i] = true;
        else
            inliers[i] = false;
    }

    pnts.reduce<uchar>(inliers);

    // inliers.clear();
    // cv::findFundamentalMat(pnts.left, pnts.newPnts,inliers, cv::FM_RANSAC, 2, 0.99);

    // pnts.reduce<uchar>(inliers);


}

bool FeatureTracker::predProj(const cv::Point3d& p3d, cv::Point2d& p2d, const bool new3D)
{
    // Logging("key",keyFrameNumb,3);
    Eigen::Vector4d point(p3d.x, p3d.y, p3d.z,1);
    // Logging("point",point,3);
    // point = zedPtr->cameraPose.poseInverse * point;
    if ( !new3D )
        point = predNPoseInv * point;
    else
        point = poseEstFrame * point;
    // Logging("point",point,3);
    // Logging("zedPtr",zedPtr->cameraPose.poseInverse,3);
    // Logging("getPose",keyframes[keyFrameNumb].getPose(),3);
    const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;
    const double invZ = 1.0f/pointZ;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};

    const int off {10};
    const int min {-off};
    const int maxW {zedPtr->mWidth + off};
    const int maxH {zedPtr->mHeight + off};

    if ( u < min || u > maxW || v < min || v > maxH)
        return false;

    p2d = cv::Point2d(u,v);
    return true;
}

void FeatureTracker::predictNewPnts(SubPixelPoints& pnts, const bool new3D)
{
    const size_t end {pnts.points3D.size()};
    pnts.newPnts.resize(end);
    std::vector<bool> in;
    in.resize(end,true);
    for (size_t i{0}; i < end; i++)
    {
        cv::Point2d pd((double)pnts.left[i].x, (double)pnts.left[i].y);
        if ( predProj(pnts.points3D[i], pd, new3D) )
            pnts.newPnts[i] = cv::Point2f((float)pd.x, (float)pd.y);
        else
            in[i] = false;

    }
    pnts.reduce<bool>(in);
}

void FeatureTracker::solvePnPIni(SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec, const bool new3D)
{
    std::vector<int>idxs;


    cv::solvePnPRansac(pnts.points3DCurr, pnts.newPnts ,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true,100, 4.0f, 0.99, idxs);
    // cv::solvePnP(pnts.points3DCurr, pnts.newPnts ,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true);

    // pnts.reduceWithInliers<int>(idxs);

    if ( new3D )
    {
        cv::Mat Rot;
        cv::Rodrigues(Rvec, Rot);
        cv::transpose(Rot, Rot);
        cv::Rodrigues(Rot, Rvec);
        tvec = -tvec;
    }
}

void FeatureTracker::checkRotTra(cv::Mat& Rvec, cv::Mat& tvec,cv::Mat& RvecN, cv::Mat& tvecN)
{
    const double R1 = cv::norm(Rvec,pRvec);
    const double R2 = cv::norm(RvecN,pRvec);
    const double T1 = cv::norm(tvec,pTvec);
    const double T2 = cv::norm(tvecN,pTvec);

    if ( (T1 > 1.0f && T2 > 1.0f) || (R1 > 0.5f && R2 > 0.5f))
    {
        tvec = pTvec.clone();
        Rvec = pRvec.clone();
    }
    else if ( T1 > 1.0f && T2 < 1.0f  && R2 < 0.5f )
    {
        tvec = tvecN.clone();
        Rvec = RvecN.clone();
    }
    else if (T2 < 1.0f && T1 < 1.0f  && R1 < 0.5f && R2 < 0.5f)
    {
        tvec.at<double>(0) =  (tvec.at<double>(0) +  tvecN.at<double>(0)) / 2.0f;
        tvec.at<double>(1) =  (tvec.at<double>(1) +  tvecN.at<double>(1)) / 2.0f;
        tvec.at<double>(2) =  (tvec.at<double>(2) +  tvecN.at<double>(2)) / 2.0f;
        Rvec.at<double>(0) =  (Rvec.at<double>(0) +  RvecN.at<double>(0)) / 2.0f;
        Rvec.at<double>(1) =  (Rvec.at<double>(1) +  RvecN.at<double>(1)) / 2.0f;
        Rvec.at<double>(2) =  (Rvec.at<double>(2) +  RvecN.at<double>(2)) / 2.0f;
    }
    else if ( T2 > 1.0f && T1 < 1.0f  && R1 < 0.5f )
    {
        tvec = tvec.clone();
        Rvec = Rvec.clone();
    }
    else
    {
        tvec = pTvec.clone();
        Rvec = pRvec.clone();
    }

    pTvec = tvec.clone();
    pRvec = Rvec.clone();

}

void FeatureTracker::estimatePoseN()
{
    cv::Mat Rvec = pRvec.clone();
    cv::Mat tvec = pTvec.clone();

    // cv::Mat RvecN = pRvec.clone();
    // cv::Mat tvecN = pTvec.clone();

    // optWithSolve(prePnts, Rvec, tvec, false);

    // calcOptical(prePnts, false);
    calcOpticalFlow(prePnts);

    if (curFrame == 1)
    {
        solvePnPIni(prePnts, Rvec, tvec, false);
        Eigen::Vector3d tra;
        Eigen::Matrix3d Rot;

        cv::Rodrigues(Rvec, Rvec);
        cv::cv2eigen(Rvec, Rot);
        cv::cv2eigen(tvec, tra);

        
        poseEstFrame.block<3, 3>(0, 0) = Rot.transpose();
        poseEstFrame.block<3, 1>(0, 3) = - tra;
    }

    // std::thread prevPntsThread(&FeatureTracker::optWithSolve, this, std::ref(prePnts), std::ref(Rvec), std::ref(tvec), false);
    // std::thread pntsThread(&FeatureTracker::optWithSolve, this, std::ref(pnts), std::ref(RvecN), std::ref(tvecN), true);

    // prevPntsThread.join();
    // pntsThread.join();

#if OPTICALIM
    // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
    drawOptical("old", pLIm.rIm,prePnts.left, prePnts.newPnts);
#endif

    // checkRotTra(Rvec, tvec, RvecN, tvecN);
    // poseEstKal(Rvec, tvec, uStereo);

    optimizePose(prePnts, pnts, Rvec, tvec);


}

bool FeatureTracker::checkOutlier(const Eigen::Matrix4d& estimatedP, const cv::Point3d& p3d, const cv::Point2f& obs, const double thres, const float weight, cv::Point2f& out2d)
{
    Eigen::Vector4d p4d(p3d.x, p3d.y, p3d.z,1);
    p4d = estimatedP * p4d;
    const double invZ = 1.0f/p4d(2);

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    const double u {fx*p4d(0)*invZ + cx};
    const double v {fy*p4d(1)*invZ + cy};

    const double errorU = weight * ((double)obs.x - u);
    const double errorV = weight * ((double)obs.y - v);

    const double error = (errorU * errorU + errorV * errorV);
    out2d = cv::Point2f((float)u, (float)v);
    if (error > thres)
        return false;
    else
        return true;
}

int FeatureTracker::checkOutliers(const Eigen::Matrix4d& estimatedP, const std::vector<cv::Point3d>& p3d, const std::vector<cv::Point2f>& obs, std::vector<bool>& inliers, const double thres, const std::vector<float>& weights)
{
    // std::vector<cv::Point2f>out2d;
    int nOut = 0;
    for (size_t i {0}; i < p3d.size(); i++)
    {
        cv::Point2f out;
        if ( !checkOutlier(estimatedP, p3d[i],obs[i], thres, weights[i],out))
        {
            nOut++;
            inliers[i] = false;
        }
        else
            inliers[i] = true;
        // out2d.push_back(out);
    }

// #if PROJECTIM
//     // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
//     drawOptical("reproj", pLIm.rIm,obs, out2d);
//     cv::waitKey(waitTrials);
// #endif
    return nOut;
}

bool FeatureTracker::checkOutlierMap(const Eigen::Matrix4d& estimatedP, Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const float weight, cv::Point2f& out2d)
{
    p4d = estimatedP * p4d;
    const double invZ = 1.0f/p4d(2);

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    const double u {fx*p4d(0)*invZ + cx};
    const double v {fy*p4d(1)*invZ + cy};

    const double errorU = weight * ((double)obs.x - u);
    const double errorV = weight * ((double)obs.y - v);

    const double error = (errorU * errorU + errorV * errorV);
    out2d = cv::Point2f((float)u, (float)v);
    if (error > thres)
        return false;
    else
        return true;
}

bool FeatureTracker::checkOutlierMap3d(const Eigen::Matrix4d& estimatedP, Eigen::Vector4d& p4d, const double thres, const float weight,  Eigen::Vector4d& obs)
{
    p4d = estimatedP * p4d;
    const double invZ = 1.0f/p4d(2);

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    const double u {fx*p4d(0)*invZ + cx};
    const double v {fy*p4d(1)*invZ + cy};

    const double errorX = weight * ((double)obs(0) - p4d(0));
    const double errorY = weight * ((double)obs(1) - p4d(1));
    const double errorZ = weight * ((double)obs(2) - p4d(2));

    const double error = (errorX * errorX + errorY * errorY + errorZ * errorZ);
    if (error > thres)
        return false;
    else
        return true;
}

int FeatureTracker::checkOutliersMap(const Eigen::Matrix4d& estimatedP, TrackedKeys& prevKeysLeft, TrackedKeys& newKeys, std::vector<bool>& inliers, const double thres, const std::vector<float>& weights)
{
    // std::vector<cv::Point2f>out2d;
    int nOut = 0;
    for (size_t i {0}, end{prevKeysLeft.keyPoints.size()}; i < end; i++)
    {
        if (prevKeysLeft.mapPointIdx[i] < 0 || prevKeysLeft.matchedIdxs[i] < 0)
            continue;
        MapPoint* mp = map->mapPoints[prevKeysLeft.mapPointIdx[i]];
        if ( !mp->GetInFrame() || mp->GetIsOutlier())
        {
            inliers[i] = false;
            continue;
        }
        cv::Point2f out;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        p4d = zedPtr->cameraPose.poseInverse * p4d;
        if ( newKeys.estimatedDepth[prevKeysLeft.matchedIdxs[i]] <= 0)
        {
            if ( !checkOutlierMap(estimatedP, p4d, prevKeysLeft.predKeyPoints[i].pt, thres, weights[i],out))
            {
                nOut++;
                inliers[i] = false;
            }
            else
                inliers[i] = true;

        }
        else
        {
            Eigen::Vector4d np4d;
            get3dFromKey(np4d, newKeys.keyPoints[prevKeysLeft.matchedIdxs[i]], newKeys.estimatedDepth[prevKeysLeft.matchedIdxs[i]]);
            if ( !checkOutlierMap3d(estimatedP, p4d, thres, weights[i],np4d))
            {
                nOut++;
                inliers[i] = false;
            }
            else
                inliers[i] = true;

        }
        // out2d.push_back(out);

    }

// #if PROJECTIM
//     // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
//     drawOptical("reproj", pLIm.rIm,obs, out2d);
//     cv::waitKey(waitTrials);
// #endif
    return nOut;
}

void FeatureTracker::calcWeights(const SubPixelPoints& pnts, std::vector<float>& weights)
{
    const size_t end {pnts.points3DCurr.size()};
    weights.resize(end, 1.0f);
    // float leftN {0};
    // float rightN {0};
    // const int off {10};
    // const float mid {(float)zedPtr->mWidth/2.0f};
    // for (size_t i {0}; i < end; i++)
    // {
    //     const float& px = pnts.newPnts[i].x;
    //     if ( px < mid )
    //         leftN += mid - px;
    //     else if ( px > mid )
    //         rightN += px - mid;
    // }
    // Logging("LeftN", leftN,3);
    // Logging("rightN", rightN,3);
    // const float multL {(float)rightN/(float)leftN};
    // const float multR {(float)leftN/(float)rightN};
    // float sum {0};
    // const float vd {zedPtr->mBaseline * 40};
    // const float sig {vd};
    // for (size_t i {0}; i < end; i++)
    // {
    //     // const float& px = pnts.newPnts[i].x;
    //     // if ( px < mid )
    //     //     weights[i] *= multL;
    //     // else if ( px > mid )
    //     //     weights[i] *= multR;
    //     const float& depth  = (float)pnts.points3DCurr[i].z;
    //     if ( depth > vd)
    //     {
    //         float prob = norm_pdf(depth, vd, sig);
    //         weights[i] *= 2 * prob * vd;
    //     }
    //     // sum += weights[i];
    // }

    // float aver {sum/(float)end};
    // float xd {0};
    // for (size_t i {0}; i < end; i++)
    // {
    //     weights[i] /= aver;
    //     xd += weights[i];
    // }

    // Logging("summm", xd/end,3);


}

void FeatureTracker::optimizePose(SubPixelPoints& prePnts, SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec)
{

    std::vector<bool> inliers(prePnts.points3DCurr.size(),true);
    std::vector<bool> prevInliers(prePnts.points3DCurr.size(),true);

    std::vector<float> weights;
    calcWeights(prePnts, weights);

    std::vector<double>thresholds = {15.6f,9.8f,7.815f,7.815f};
    // bool rerun {false};
    int nIn = prePnts.points3DCurr.size();
    bool rerun = true;

    Eigen::Matrix4d prevCalcPose = poseEstFrame;

    for (size_t times = 0; times < 4; times++)
    {
        ceres::Problem problem;

        Eigen::Vector3d frame_tcw;
        Eigen::Quaterniond frame_qcw;

        Eigen::Matrix4d frame_pose = poseEstFrame;
        Eigen::Matrix3d frame_R;
        frame_R = frame_pose.block<3, 3>(0, 0);
        frame_tcw = frame_pose.block<3, 1>(0, 3);
        frame_qcw = Eigen::Quaterniond(frame_R);
        
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fx;
        K(1,1) = fy;
        K(0,2) = cx;
        K(1,2) = cy;
        ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
        // problem.AddParameterBlock(cameraR,4);
        // problem.AddParameterBlock(cameraT,3);
        for (size_t i{0}, end{prePnts.points3DCurr.size()}; i < end; i++)
        {
            if ( !inliers[i] )
                continue;

            Eigen::Vector2d obs((double)prePnts.newPnts[i].x, (double)prePnts.newPnts[i].y);
            Eigen::Vector3d point(prePnts.points3DCurr[i].x, prePnts.points3DCurr[i].y, prePnts.points3DCurr[i].z);
            ceres::CostFunction* costf = OptimizePose::Create(K, point, obs, weights[i]);
            
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.use_explicit_schur_complement = true;
        
        options.max_num_iterations = 100;
        // options.max_solver_time_in_seconds = 0.05;

        // options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // Logging("sum",summary.FullReport(),3);
        Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = R.transpose();
        pose.block<3, 1>(0, 3) = - frame_tcw;
        // zedPtr->cameraPose.setPose(pose);
        // zedPtr->cameraPose.setInvPose(pose.inverse());

        poseEstFrame = pose;
        Eigen::Matrix4d pFInv = pose.inverse();
        int nOut = checkOutliers(pFInv, prePnts.points3DCurr, prePnts.newPnts, inliers, thresholds[times], weights);
        int nInAfter = nIn - nOut;

        // if ( nInAfter < (nIn / 2))
        // {
        //     // if more than half then keep the other bunch
        //     // that means that we take for granted that more than half are right.

        //     // You can change all the inliers to the opposite (false = true, true = false)
        //     // And rerun the last loop with the new inliers


        //     if ( rerun )
        //     {
        //         for (size_t i{0}, end{prePnts.points3DCurr.size()}; i < end; i++)
        //         {
        //             if ( inliers[i] )
        //                 inliers[i] = false;
        //             else
        //                 inliers[i] = true;
        //         }
        //         times = times - 1;
        //         poseEstFrame = prevCalcPose;
        //         prevInliers = inliers;
        //         rerun = false;
        //         continue;

        //     }
        //     else
        //     {
        //         poseEstFrame = prevCalcPose;
        //         inliers = prevInliers;
        //     }
            

        //     break;
        // }
        // else
        // {
            prevCalcPose = poseEstFrame;
            prevInliers = inliers;
        // }

    }

    prePnts.reduce<bool>(prevInliers);

#if PROJECTIM
    // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
    drawOptical("optimization", pLIm.rIm,prePnts.left, prePnts.newPnts);
    // cv::waitKey(waitTrials);
#endif

    publishPose();
    // float sum {0};
    // const float middd = (float)zedPtr->mWidth/2.0f;
    // const size_t sizeee = prePnts.points3DCurr.size();
    // for ( size_t iS {0}; iS < sizeee; iS++ )
    //     sum += (prePnts.left[iS].x - middd);

    // leftRight += sum/ sizeee;

    // Logging("LEEEEEEEEEEEEEEEFTTTTTTTTTT", leftRight,3);


    // Eigen::Matrix3d Reig = temp.block<3, 3>(0, 0);
    // Eigen::Vector3d Teig = temp.block<3, 1>(0, 3);
    // cv::Mat Rot, tra;

    // cv::eigen2cv(Reig, Rot);
    // cv::eigen2cv(Teig, tra);

    // cv::Rodrigues(Rot, Rot);

    // poseEstKal(Rot, tra, uStereo);

    

    // poseEst = poseEst * poseEstFrame;
    // poseEstFrameInv = poseEstFrame.inverse();
    // prevWPose = zedPtr->cameraPose.pose;
    // prevWPoseInv = zedPtr->cameraPose.poseInverse;
    // zedPtr->cameraPose.setPose(poseEst);
    // zedPtr->cameraPose.setInvPose(poseEst.inverse());
    // predNPose = poseEst * (prevWPoseInv * poseEst);
    // predNPoseInv = predNPose.inverse();
    // options.gradient_tolerance = 1e-16;
    // options.function_tolerance = 1e-16;
    // options.parameter_tolerance = 1e-16;
    // double cost {0.0};
    // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
    // Logging("cost ", summary.final_cost,3);
    // Logging("R bef", Rvec,3);
    // Logging("T bef", tvec,3);
    // Rvec.at<double>(0) = euler[0];
    // Rvec.at<double>(1) = euler[1];
    // Rvec.at<double>(2) = euler[2];
    // tvec.at<double>(0) = cameraT[0];
    // tvec.at<double>(1) = cameraT[1];
    // tvec.at<double>(2) = cameraT[2];
    // Logging("R after", Rvec,3);
    // Logging("T after", tvec,3);
}

void FeatureTracker::get3dFromKey(Eigen::Vector4d& pnt4d, const cv::KeyPoint& pnt, const float depth)
{
    const double zp = (double)depth;
    const double xp = (double)(((double)pnt.pt.x-cx)*zp/fx);
    const double yp = (double)(((double)pnt.pt.y-cy)*zp/fy);
    pnt4d = Eigen::Vector4d(xp, yp, zp,1);
}

void FeatureTracker::optimizePoseCeres(TrackedKeys& prevKeys, TrackedKeys& newKeys)
{


    const size_t prevS { prevKeys.keyPoints.size()};
    const size_t newS { newKeys.keyPoints.size()};
    const size_t startPrev {newS - prevS};
    prevKeys.inliers.resize(prevS,true);
    std::vector<bool> inliers(prevS,true);
    std::vector<bool> prevInliers(prevS,true);

    std::vector<float> weights;
    // calcWeights(prePnts, weights);
    weights.resize(newS, 1.0f);

    std::vector<double>thresholds = {15.6f,9.8f,7.815f,7.815f};
    // bool rerun {false};
    int nIn = prevS;
    bool rerun = true;

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0,0) = fx;
    K(1,1) = fy;
    K(0,2) = cx;
    K(1,2) = cy;
    Eigen::Matrix4d prevCalcPose = poseEstFrameInv;
    std::vector<cv::Point2f> calc, predictedd;
    for (size_t times = 0; times < 4; times++)
    {
        std::vector<cv::Point2f> found, observ;
        ceres::Problem problem;
        Eigen::Vector3d frame_tcw;
        Eigen::Quaterniond frame_qcw;

        Eigen::Matrix4d frame_pose = poseEstFrameInv;
        Eigen::Matrix3d frame_R;
        frame_R = frame_pose.block<3, 3>(0, 0);
        frame_tcw = frame_pose.block<3, 1>(0, 3);
        frame_qcw = Eigen::Quaterniond(frame_R);
        
        ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(7.815f));
        int count {0};
        for (size_t i{0}, end{prevKeys.keyPoints.size()}; i < end; i++)
        {
            if (prevKeys.mapPointIdx[i] < 0 || prevKeys.matchedIdxs[i] < 0)
                continue;
            MapPoint* mp = map->mapPoints[prevKeys.mapPointIdx[i]];
            if ( !mp->GetInFrame() || mp->GetIsOutlier())
                continue;
            if ( !inliers[i] )
                continue;
            if ( newKeys.estimatedDepth[prevKeys.matchedIdxs[i]] > 0)
                continue;
            count ++;
            Eigen::Vector2d obs((double)prevKeys.predKeyPoints[i].pt.x, (double)prevKeys.predKeyPoints[i].pt.y);
            observ.push_back(prevKeys.predKeyPoints[i].pt);
            Eigen::Vector4d point = mp->getWordPose4d();
            point = zedPtr->cameraPose.poseInverse * point;
            Eigen::Vector3d point3d(point(0), point(1),point(2));
            ceres::CostFunction* costf = OptimizePose::Create(K, point3d, obs, (double)weights[i]);
            Eigen::Vector3d pmoved;
            pmoved = K * point3d;
            found.push_back(cv::Point2f((float)pmoved(0)/(float)pmoved(2), (float)pmoved(1)/(float)pmoved(2)));
            
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }

        for (size_t i{0}, end{prevKeys.keyPoints.size()}; i < end; i++)
        {
            if (prevKeys.mapPointIdx[i] < 0 || prevKeys.matchedIdxs[i] < 0)
                continue;
            MapPoint* mp = map->mapPoints[prevKeys.mapPointIdx[i]];
            if ( !mp->GetInFrame() || mp->GetIsOutlier())
                continue;
            if ( !inliers[i] )
                continue;
            if ( newKeys.estimatedDepth[prevKeys.matchedIdxs[i]] <= 0)
                continue;
            count ++;

            Eigen::Vector4d np4d;
            get3dFromKey(np4d, newKeys.keyPoints[prevKeys.matchedIdxs[i]], newKeys.estimatedDepth[prevKeys.matchedIdxs[i]]);
            observ.push_back(newKeys.keyPoints[prevKeys.matchedIdxs[i]].pt);
            // np4d = zedPtr->cameraPose.pose * np4d;
            Eigen::Vector3d obs(np4d(0), np4d(1),np4d(2));
            Eigen::Vector4d point = mp->getWordPose4d();
            point = zedPtr->cameraPose.poseInverse * point;
            Eigen::Vector3d point3d(point(0), point(1),point(2));
            ceres::CostFunction* costf = OptimizePoseICP::Create(K, point3d, obs, (double)weights[i]);
            Eigen::Vector3d pmoved;
            pmoved = K * point3d;
            found.push_back(cv::Point2f((float)pmoved(0)/(float)pmoved(2), (float)pmoved(1)/(float)pmoved(2)));
            
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.use_explicit_schur_complement = true;
        
        options.max_num_iterations = 100;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = R.transpose();
        pose.block<3, 1>(0, 3) = - frame_tcw;

        Eigen::Matrix4d pFInv = pose.inverse();
        int nOut = checkOutliersMap(pFInv, prevKeys, newKeys, inliers, thresholds[times], weights);
        // int nInAfter = nIn - nOut;
        Logging("POSE EST", pose,3);
        prevCalcPose = pose;
        prevInliers = inliers;
        drawOptical("optimization", pLIm.rIm, observ, found);
        cv::waitKey(1);

    }
    poseEstFrame = prevCalcPose;
    for ( size_t i {0}; i < inliers.size(); i++)
    {
        if ( !inliers[i] )
        {
            if (prevKeys.mapPointIdx[i] < 0)
                    continue;
            MapPoint* mp = map->mapPoints[prevKeys.mapPointIdx[i]];
            mp->SetIsOutlier(true);
            
            newKeys.close[prevKeys.matchedIdxs[i]] = false;
        }
        else
            if ( prevKeys.mapPointIdx[i] >= 0)
            {
                if ( prevKeys.matchedIdxs[i] >= 0)
                {
                    MapPoint* mp = map->mapPoints[prevKeys.mapPointIdx[i]];
                    Eigen::Vector4d point = mp->getWordPose4d();
                    point = zedPtr->cameraPose.poseInverse * point;
                    point = poseEstFrame.inverse() * point;
                    Eigen::Vector3d point3d(point(0), point(1),point(2));
                    point3d = K * point3d;
                    calc.push_back(cv::Point2f((float)point3d(0)/(float)point3d(2), (float)point3d(1)/(float)point3d(2)));
                    predictedd.push_back(newKeys.keyPoints[prevKeys.matchedIdxs[i]].pt);
                }
            }
    }
    drawOptical("reproj erro", pLIm.rIm, predictedd, calc);
    cv::waitKey(1);
    // prevKeys.reduce<bool>(prevInliers);
    // prePnts.reduce<bool>(prevInliers);

#if PROJECTIM
    // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
#endif

    publishPoseCeres();
}

void FeatureTracker::optimizePoseORB(TrackedKeys& prevKeys, TrackedKeys& newKeys)
{


    const size_t prevS { prevKeys.keyPoints.size()};
    const size_t newS { newKeys.keyPoints.size()};
    const size_t startPrev {newS - prevS};
    prevKeys.inliers.resize(prevS,true);
    std::vector<bool> inliers(prevS,true);
    std::vector<bool> prevInliers(prevS,true);

    std::vector<float> weights;
    // calcWeights(prePnts, weights);
    weights.resize(newS, 1.0f);

    std::vector<double>thresholds = {15.6f,9.8f,7.815f,7.815f};
    // bool rerun {false};
    int nIn = prevS;
    bool rerun = true;

    Eigen::Matrix4d prevCalcPose = poseEst;

    std::vector<cv::Point2f> found, observ;
    for (size_t times = 0; times < 1; times++)
    {
        ceres::Problem problem;
        Eigen::Vector3d frame_tcw;
        Eigen::Quaterniond frame_qcw;

        Eigen::Matrix4d frame_pose = poseEst;
        Eigen::Matrix3d frame_R;
        frame_R = frame_pose.block<3, 3>(0, 0);
        frame_tcw = frame_pose.block<3, 1>(0, 3);
        frame_qcw = Eigen::Quaterniond(frame_R);
        
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fx;
        K(1,1) = fy;
        K(0,2) = cx;
        K(1,2) = cy;
        ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(7.815f));
        for (size_t i{0}, end{prevKeys.keyPoints.size()}; i < end; i++)
        {
            if ( !prevKeys.close[i] )
                continue;
            MapPoint* mp = map->mapPoints[prevKeys.mapPointIdx[i]];
            if ( !mp->GetInFrame() || mp->GetIsOutlier())
                continue;
            Eigen::Vector2d obs((double)prevKeys.predKeyPoints[i].pt.x, (double)prevKeys.predKeyPoints[i].pt.y);
            observ.push_back(prevKeys.predKeyPoints[i].pt);
            Eigen::Vector3d point = mp->getWordPose3d();
            ceres::CostFunction* costf = OptimizePose::Create(K, point, obs, weights[i]);
            Eigen::Vector3d pmoved = frame_R * point + frame_tcw;
            pmoved = K * pmoved;
            found.push_back(cv::Point2f((float)pmoved(0)/(float)pmoved(2), (float)pmoved(1)/(float)pmoved(2)));
            
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }

        // for (size_t i{0}, end{prevKeys.keyPoints.size()}; i < end; i++)
        // {
        //     if ( !inliers[i] )
        //         continue;

        //     Eigen::Vector2d obs((double)prePnts.newPnts[i].x, (double)prePnts.newPnts[i].y);
        //     Eigen::Vector3d point(prePnts.points3DCurr[i].x, prePnts.points3DCurr[i].y, prePnts.points3DCurr[i].z);
        //     ceres::CostFunction* costf = OptimizePose::Create(K, point, obs, weights[i]);
            
        //     problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

        //     problem.SetManifold(frame_qcw.coeffs().data(),
        //                                 quaternion_local_parameterization);
        // }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.use_explicit_schur_complement = true;
        
        options.max_num_iterations = 100;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = R.transpose();
        pose.block<3, 1>(0, 3) = - frame_tcw;

        poseEst = pose;
        Eigen::Matrix4d pFInv = pose.inverse();
        int nOut = checkOutliersMap(pFInv, prevKeys, newKeys, inliers, thresholds[times], weights);
        int nInAfter = nIn - nOut;
        Logging("POSE EST", pose,3);
        prevCalcPose = poseEst;
        prevInliers = inliers;

    }
    // prevKeys.reduce<bool>(prevInliers);
    // prePnts.reduce<bool>(prevInliers);

#if PROJECTIM
    // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
    drawOptical("optimization", pLIm.rIm, observ, found);
    cv::waitKey(waitTrials);
#endif

    publishPoseCeres();
}

void FeatureTracker::optWithSolve(SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec, const bool new3D)
{
    calcOptical(pnts, new3D);
    solvePnPIni(pnts, Rvec, tvec, new3D);

}

void FeatureTracker::computeStereoMatches(TrackedKeys& keysLeft, TrackedKeys& prevLeftKeys)
{
    std::vector<cv::Point2f> tobeRemoved;

    std::vector<cv::KeyPoint> rightKeys;
    cv::Mat rightDesc;

    Timer fast("FAST");
    std::thread extractLeft(&FeatureExtractor::extractLeftFeatures, std::ref(feLeft), std::ref(lIm.im), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc), std::ref(prevLeftKeys));
    std::thread extractRight(&FeatureExtractor::computeFASTandDesc, std::ref(feRight), std::ref(rIm.im), std::ref(rightKeys), std::ref(tobeRemoved),std::ref(rightDesc));

    extractLeft.join();
    extractRight.join();

    fm.findStereoMatchesCloseFar(lIm.im, rIm.im, rightDesc, rightKeys, keysLeft);

    drawKeyPointsCloseFar("new method", lIm.rIm, keysLeft, rightKeys);
}

void FeatureTracker::computeStereoMatchesORB(TrackedKeys& keysLeft, TrackedKeys& prevLeftKeys)
{
    Timer both ("Both");
    std::vector<cv::KeyPoint> rightKeys;
    cv::Mat rightDesc;

    Timer orb("orb");
    std::thread extractLeft(&FeatureExtractor::extractLeftFeaturesORB, std::ref(feLeft), std::ref(lIm.im), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc), std::ref(prevLeftKeys));
    std::thread extractRight(&FeatureExtractor::computeORBandDesc, std::ref(feRight), std::ref(rIm.im), std::ref(rightKeys),std::ref(rightDesc));

    extractLeft.join();
    extractRight.join();

    fm.findStereoMatchesCloseFar(lIm.im, rIm.im, rightDesc, rightKeys, keysLeft);

    drawKeyPointsCloseFar("new method", lIm.rIm, keysLeft, rightKeys);
}

void FeatureTracker::drawKeyPointsCloseFar(const char* com, const cv::Mat& im, const TrackedKeys& keysLeft, const std::vector<cv::KeyPoint>& right)
{
        cv::Mat outIm = im.clone();
        const size_t end {keysLeft.keyPoints.size()};
        for (size_t i{0};i < end; i ++ )
        {
            if ( keysLeft.estimatedDepth[i] > 0)
            {
                cv::circle(outIm, keysLeft.keyPoints[i].pt,2,cv::Scalar(0,255,0));
                cv::line(outIm, keysLeft.keyPoints[i].pt, right[keysLeft.rightIdxs[i]].pt,cv::Scalar(0,0,255));
                cv::circle(outIm, right[keysLeft.rightIdxs[i]].pt,2,cv::Scalar(255,0,0));
            }
        }
        cv::imshow(com, outIm);
        cv::waitKey(waitImClo);

}

void FeatureTracker::drawLeftMatches(const char* com, const cv::Mat& im, const TrackedKeys& prevKeysLeft, const TrackedKeys& keysLeft)
{
        cv::Mat outIm = im.clone();
        const size_t end {prevKeysLeft.keyPoints.size()};
        for (size_t i{0};i < end; i ++ )
        {
            if ( prevKeysLeft.matchedIdxs[i] >= 0)
            {
                cv::circle(outIm, prevKeysLeft.keyPoints[i].pt,2,cv::Scalar(0,255,0));
                cv::line(outIm, prevKeysLeft.keyPoints[i].pt, keysLeft.keyPoints[prevKeysLeft.matchedIdxs[i]].pt,cv::Scalar(0,0,255));
                cv::circle(outIm, keysLeft.keyPoints[prevKeysLeft.matchedIdxs[i]].pt,2,cv::Scalar(255,0,0));
            }
        }
        cv::imshow(com, outIm);
        cv::waitKey(waitImClo);

}

void FeatureTracker::Track(const int frames)
{
    for (curFrame = 0; curFrame < frames; curFrame++)
    {

        zedPtr->addKeyFrame = true;
        setLRImages(curFrame);

    //     Eigen::Vector4d p(0,0,0,1);
    // map->addMapPoint(p);

    // Eigen::Vector4d p2(500,432,5234,3211);
    // map->addMapPoint(p2);

        findStereoFeatures(lIm.im, rIm.im, pnts);

        triangulate3DPoints(pnts);

        if ( curFrame == 0 )
        {
            setPreviousValuesIni();
            continue;
        }


        estimatePoseN();

        setPreviousValues();

        // calcOptical(pnts, true);
        // calcOptical(prePnts, false);

        // solvePnPIni(prePnts, Rvec, tvec, false);
        // solvePnPIni(pnts, Rvec, tvec, true);




        // if ( curFrame == 1 )
        //     opticalFlow();
        // else
        //     opticalFlowPredict();
        // getPoseCeresNew();

        // setPreTrial();

        // Logging("ustereo", uStereo,3);
        // Logging("umono", uMono,3);
    }
    datafile.close();
}

void FeatureTracker::predictPntsLeft(TrackedKeys& keysLeft)
{
    std::vector<bool>inliers (keysLeft.keyPoints.size(),true);
    for ( size_t i{0}, end {keysLeft.keyPoints.size()}; i < end; i++)
    {
        if (keysLeft.mapPointIdx[i] >= 0)
        {
            cv::Point2f predPnt;
            if ( getPredInFrame(predNPoseInv, map->mapPoints[keysLeft.mapPointIdx[i]], predPnt))
                keysLeft.predKeyPoints[i].pt = predPnt;
            else
                inliers[i] = false;
        }
    }
    keysLeft.reduce<bool>(inliers);
}

void FeatureTracker::computeOpticalLeft(TrackedKeys& keysLeft)
{
    const size_t keysLeftSize {keysLeft.keyPoints.size()};
    std::vector<float> err;
    std::vector<uchar> inliers, inliers2;
    std::vector<cv::Point2f> keysp2f, predkeysp2f;
    keysLeft.predKeyPoints = keysLeft.keyPoints;
    if ( curFrame == 1 )
    {

        cv::KeyPoint::convert(keysLeft.keyPoints, keysp2f);
        cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, keysp2f, predkeysp2f, inliers, err,cv::Size(21,21),3, criteria);

    }
    else
    {
        predictPntsLeft(keysLeft);
        keysp2f.reserve(keysLeft.keyPoints.size());
        predkeysp2f.reserve(keysLeft.keyPoints.size());
        for (size_t i{0}, end(keysLeft.keyPoints.size()); i < end; i++)
        {
            keysp2f.emplace_back(keysLeft.keyPoints[i].pt);
            predkeysp2f.emplace_back(keysLeft.predKeyPoints[i].pt);
        }
        // cv::KeyPoint::convert(keysLeft.keyPoints, keysp2f);
        // cv::KeyPoint::convert(keysLeft.predKeyPoints, predkeysp2f);

        drawPointsTemp<cv::Point2f,cv::Point2f>("optical", pLIm.rIm, keysp2f, predkeysp2f);


        cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, keysp2f, predkeysp2f, inliers, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    std::vector<cv::Point2f> temp = keysp2f;
    cv::calcOpticalFlowPyrLK(lIm.im, pLIm.im, predkeysp2f, temp, inliers2, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i {0}; i < pnts.left.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],keysp2f[i]) <= 0.25)
            inliers[i] = true;
        else
            inliers[i] = false;
    }

    const int w {zedPtr->mWidth};
    const int h {zedPtr->mHeight};

    for ( size_t i{0}, end{prevLeftPnts.predKeyPoints.size()}; i < end; i++)
    {
        const cv::Point2f& p2f = predkeysp2f[i];
        if (inliers[i] )
        {
            if (p2f.x > 0 && p2f.x < w && p2f.y > 0 && p2f.y < h)
            {
                prevLeftPnts.predKeyPoints[i].pt = predkeysp2f[i];
            }
            else
                inliers[i] = false;
        }
    }
    prevLeftPnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::Point2f,uchar>(predkeysp2f, inliers);
    reduceVectorTemp<cv::Point2f,uchar>(keysp2f, inliers);
    inliers.clear();
    cv::findFundamentalMat(keysp2f, predkeysp2f,inliers);
    prevLeftPnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::Point2f,uchar>(predkeysp2f, inliers);
    reduceVectorTemp<cv::Point2f,uchar>(keysp2f, inliers);
    size_t s {keysp2f.size()};
    size_t s2 {predkeysp2f.size()};


}

void setMaskOfIdxs(cv::Mat& mask, const TrackedKeys& keysLeft)
{
    std::vector<cv::KeyPoint>::const_iterator it, end(keysLeft.keyPoints.end());
    for ( it = keysLeft.keyPoints.begin(); it != end; it++)
    {

    }
}

void FeatureTracker::addMapPnts(TrackedKeys& keysLeft)
{
    const size_t prevE { prevLeftPnts.keyPoints.size()};
    const size_t newE { keysLeft.keyPoints.size()};
    const size_t start {newE - prevE};
    prevLeftPnts.keyPoints.reserve(keysLeft.keyPoints.size());
    // cv::Mat mask = cv::Mat(zedPtr->mHeight , zedPtr->mWidth , CV_16UC1, cv::Scalar(0));

    for (size_t i{start}, end {newE}; i < end; i++)
    {

        if ( keysLeft.estimatedDepth[i] > 0)
        {
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = zedPtr->cameraPose.pose * p;
            if ( prevLeftPnts.mapPointIdx[i - start] >= 0)
            {
                Eigen::Vector4d mp = map->mapPoints[prevLeftPnts.mapPointIdx[i-start]]->getWordPose4d();
                p(0) = (p(0) + mp(0)) / 2;
                p(1) = (p(1) + mp(1)) / 2;
                p(2) = (p(2) + mp(2)) / 2;
                map->mapPoints[prevLeftPnts.mapPointIdx[i-start]]->updateMapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i]);
            }
            else
            {
                prevLeftPnts.mapPointIdx[i - start] = map->pIdx;
                map->addMapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], keysLeft.close[i]);
            }
            prevLeftPnts.estimatedDepth[i - start] = keysLeft.estimatedDepth[i];
            prevLeftPnts.close[i - start] = keysLeft.close[i];
        }
        prevLeftPnts.keyPoints[i - start] = keysLeft.keyPoints[i];
        prevLeftPnts.trackCnt[i - start] ++;
    }
    for (size_t i{0}, end {start}; i < end; i++)
    {

        if ( keysLeft.estimatedDepth[i] > 0 )
        {
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = zedPtr->cameraPose.pose * p;
            prevLeftPnts.add(keysLeft.keyPoints[i], map->pIdx, keysLeft.estimatedDepth[i], keysLeft.close[i], 0 );
            map->addMapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], keysLeft.close[i]);
            // Here check if another point is close to see if that point is 3d
            
        }
        else
        {
            prevLeftPnts.add(keysLeft.keyPoints[i], -1, -1.0f, keysLeft.close[i], 0 );

        }
    }
    map->addKeyFrame(zedPtr->cameraPose.pose);
}

bool FeatureTracker::getPredInFrame(const Eigen::Matrix4d& predPose, MapPoint* mp, cv::Point2f& predPnt)
{
    if ( mp->GetInFrame() && !mp->GetIsOutlier())
    {
        Eigen::Vector4d p4d = predPose * mp->getWordPose4d();

        if (p4d(2) <= 0.0f )
        {
            mp->SetInFrame(false);
            return false;
        }

        const double invfx = 1.0f/fx;
        const double invfy = 1.0f/fy;


        double u {fx*p4d(0)/p4d(2) + cx};
        double v {fy*p4d(1)/p4d(2) + cy};

        const int h = zedPtr->mHeight;
        const int w = zedPtr->mWidth;

        if ( u < 0 || u > w  || v < 0 || v > h )
        {
            mp->SetInFrame(false);
            return false;
        }
        else
        {
            predPnt = cv::Point2f((float)u, (float)v);
            return true;
        }
    }
    return false;
}

bool FeatureTracker::getPoseInFrame(const Eigen::Matrix4d& pose, const Eigen::Matrix4d& predPose, MapPoint* mp, cv::Point2f& pnt, cv::Point2f& predPnt)
{
    if ( mp->GetInFrame() && !mp->GetIsOutlier())
    {
        Eigen::Vector4d p4d = pose * mp->getWordPose4d();
        Eigen::Vector4d predP4d = predPose * mp->getWordPose4d();
        

        if (p4d(2) <= 0.0f || predP4d(2) <= 0.0f)
        {
            mp->SetInFrame(false);
            return false;
        }

        const double invfx = 1.0f/fx;
        const double invfy = 1.0f/fy;


        double u {fx*p4d(0)/p4d(2) + cx};
        double v {fy*p4d(1)/p4d(2) + cy};

        double pu {fx*predP4d(0)/predP4d(2) + cx};
        double pv {fy*predP4d(1)/predP4d(2) + cy};

        const int h = zedPtr->mHeight;
        const int w = zedPtr->mWidth;

        if ( u < 0 || u > w || pu < 0 || pu > w || v < 0 || v > h || pv < 0 || pv > h)
        {
            mp->SetInFrame(false);
            return false;
        }
        else
        {
            pnt = cv::Point2f((float)u, (float)v);
            predPnt = cv::Point2f((float)pu, (float)pv);
            return true;
        }
    }
    return false;
}

void FeatureTracker::addStereoPnts()
{

    // std::unordered_map<unsigned long, MapPoint*>::const_iterator end(map->mapPoints.end());
    // std::unordered_map<unsigned long, MapPoint*>::iterator it;
    // for ( it = map->mapPoints.begin(); it!= end; it++)
    // {
    //     cv::Point2f pnt, predPnt;
    //     if ( getPoseInFrame(zedPtr->cameraPose.pose, predNPoseInv, it->second, pnt, predPnt))
    //     {
    //         const cv::KeyPoint& kp = it->second->obs[0];
    //         prevLeftPnts.keyPoints.emplace_back(pnt, kp.size, kp.angle, kp.response, kp.octave);
    //         prevLeftPnts.predPos.emplace_back(predPnt);
    //     }
    // }
}

void FeatureTracker::getNewMatchedPoints(TrackedKeys& keysMatched, TrackedKeys& newPnts)
{
    const size_t prevTrS { prevLeftPnts.keyPoints.size()};
    const size_t keysMS { keysMatched.keyPoints.size()};

    const size_t start { keysMS - prevTrS};
}

void FeatureTracker::Track2(const int frames)
{
    for (curFrame = 0; curFrame < frames; curFrame++)
    {
        zedPtr->addKeyFrame = true;
        setLRImages(curFrame);

        TrackedKeys keysLeft;



        if ( curFrame == 0 )
        {
            computeStereoMatches(keysLeft, prevLeftPnts);
            // prevLeftPnts.clone(keysLeft);
            addMapPnts(keysLeft);
            setPreLImage();
            setPreRImage();
            continue;
        }

        // add the projected mappoint here that are in frame 
        // addStereoPnts();

        computeOpticalLeft(prevLeftPnts);

        // here check size of predKeysPoints and start from end of new keypoints - size (it is the first predKeyPoint)
        computeStereoMatches(keysLeft, prevLeftPnts);

        // getNewMatchedPoints(keysLeft, prevLeftPnts);
        // optimization goes here
        optimizePoseCeres(prevLeftPnts, keysLeft);

        addMapPnts(keysLeft);

        // prevLeftPnts.clear();
        const size_t s {prevLeftPnts.keyPoints.size()};
        const size_t s4 {prevLeftPnts.estimatedDepth.size()};
        const size_t s5 {prevLeftPnts.predKeyPoints.size()};
        const size_t s6 {prevLeftPnts.mapPointIdx.size()};
        const size_t s7 {prevLeftPnts.trackCnt.size()};
        const size_t s2 {keysLeft.keyPoints.size()};
        const size_t s3 {keysLeft.keyPoints.size()};


        setPreLImage();
        setPreRImage();

        // zedPtr->addKeyFrame = true;
    //     setLRImages(curFrame);

    // //     Eigen::Vector4d p(0,0,0,1);
    // // map->addMapPoint(p);

    // // Eigen::Vector4d p2(500,432,5234,3211);
    // // map->addMapPoint(p2);

    //     findStereoFeatures(lIm.im, rIm.im, pnts);

    //     triangulate3DPoints(pnts);

    //     if ( curFrame == 0 )
    //     {
            // setPreviousValuesIni();
    //         continue;
    //     }


    //     estimatePoseN();

    //     setPreviousValues();

        // calcOptical(pnts, true);
        // calcOptical(prePnts, false);

        // solvePnPIni(prePnts, Rvec, tvec, false);
        // solvePnPIni(pnts, Rvec, tvec, true);




        // if ( curFrame == 1 )
        //     opticalFlow();
        // else
        //     opticalFlowPredict();
        // getPoseCeresNew();

        // setPreTrial();

        // Logging("ustereo", uStereo,3);
        // Logging("umono", uMono,3);
    }
    datafile.close();
}

void FeatureTracker::updateMapPoints(TrackedKeys& prevLeftKeys)
{
    for (size_t i{0}, end {prevLeftKeys.keyPoints.size()}; i < end; i++)
    {

        if ( prevLeftKeys.mapPointIdx[i] >= 0)
        {
            if ( prevLeftKeys.estimatedDepth[i] > 0 )
            {
                const double zp = (double)prevLeftKeys.estimatedDepth[i];
                const double xp = (double)(((double)prevLeftKeys.keyPoints[i].pt.x-cx)*zp/fx);
                const double yp = (double)(((double)prevLeftKeys.keyPoints[i].pt.y-cy)*zp/fy);
                Eigen::Vector4d p(xp, yp, zp, 1);
                p = zedPtr->cameraPose.pose * p;

                Eigen::Vector4d mp = map->mapPoints[prevLeftKeys.mapPointIdx[i]]->getWordPose4d();
                p(0) = (p(0) + mp(0)) / 2;
                p(1) = (p(1) + mp(1)) / 2;
                p(2) = (p(2) + mp(2)) / 2;
                map->mapPoints[prevLeftKeys.mapPointIdx[i]]->updateMapPoint(p, prevLeftKeys.Desc.row(i), prevLeftKeys.keyPoints[i]);
            }
        }
        else if ( prevLeftKeys.close[i] )
        {
            // if ( curFrame == 0 || prevLeftKeys.trackCnt[i] > 2 )
            // {
                const double zp = (double)prevLeftKeys.estimatedDepth[i];
                const double xp = (double)(((double)prevLeftKeys.keyPoints[i].pt.x-cx)*zp/fx);
                const double yp = (double)(((double)prevLeftKeys.keyPoints[i].pt.y-cy)*zp/fy);
                Eigen::Vector4d p(xp, yp, zp, 1);
                p = zedPtr->cameraPose.pose * p;
                prevLeftKeys.mapPointIdx[i] = map->pIdx;
                map->addMapPoint(p, prevLeftKeys.Desc.row(i), prevLeftKeys.keyPoints[i], prevLeftKeys.close[i]);
            // }
        }
        // else if ( prevLeftKeys.close[i] )
        // {
        //     const double zp = (double)prevLeftKeys.estimatedDepth[i];
        //     const double xp = (double)(((double)prevLeftKeys.keyPoints[i].pt.x-cx)*zp/fx);
        //     const double yp = (double)(((double)prevLeftKeys.keyPoints[i].pt.y-cy)*zp/fy);
        //     Eigen::Vector4d p(xp, yp, zp, 1);
        //     p = zedPtr->cameraPose.pose * p;
        //     prevLeftPnts.mapPointIdx[i] = map->pIdx;
        //     map->addMapPoint(p, prevLeftKeys.Desc.row(i), prevLeftKeys.keyPoints[i], prevLeftKeys.close[i]);
        //     // Here check if another point is close to see if that point is 3d
            
        // }
    }
    map->addKeyFrame(zedPtr->cameraPose.pose);
}

void FeatureTracker::predictORBPoints(TrackedKeys& prevLeftKeys)
{
    const size_t prevS { prevLeftKeys.keyPoints.size()};
    prevLeftKeys.inliers = std::vector<bool>(prevS, true);
    prevLeftKeys.hasPrediction = std::vector<bool>(prevS, false);
    prevLeftKeys.predKeyPoints = prevLeftKeys.keyPoints;
    
    if ( curFrame == 1 )
        return;

    for ( size_t i{0}; i < prevS; i++)
    {
        if ( prevLeftKeys.mapPointIdx[i] >= 0 )
        {
            MapPoint* mp = map->mapPoints[prevLeftKeys.mapPointIdx[i]];
            cv::Point2f p2f;
            if ( getPredInFrame(predNPoseInv, mp, p2f))
            {
                prevLeftKeys.predKeyPoints[i].pt = p2f;
                prevLeftKeys.hasPrediction[i] = true;
            }
            else
                prevLeftKeys.inliers[i] = false;
        }
    }
}

void FeatureTracker::reduceTrackedKeys(TrackedKeys& leftKeys, std::vector<bool>& inliers)
{
    int j {0};
    for (int i = 0; i < int(leftKeys.keyPoints.size()); i++)
    {
        if (inliers[i])
        {
            leftKeys.keyPoints[j] = leftKeys.keyPoints[i];
            leftKeys.estimatedDepth[j] = leftKeys.estimatedDepth[i];
            leftKeys.mapPointIdx[j] = leftKeys.mapPointIdx[i];
            leftKeys.matchedIdxs[j] = leftKeys.matchedIdxs[i];
            leftKeys.close[j] = leftKeys.close[i];
            leftKeys.trackCnt[j] = leftKeys.trackCnt[i];
            j++;
        }

    }
    leftKeys.keyPoints.resize(j);
    leftKeys.estimatedDepth.resize(j);
    leftKeys.mapPointIdx.resize(j);
    leftKeys.matchedIdxs.resize(j);
    leftKeys.close.resize(j);
    leftKeys.trackCnt.resize(j);
}

void FeatureTracker::reduceTrackedKeysMatches(TrackedKeys& prevLeftKeys, TrackedKeys& leftKeys)
{
    const size_t end{prevLeftKeys.keyPoints.size()};
    const size_t endNew {leftKeys.keyPoints.size()};

    std::vector<bool>inliers(end,true);
    std::vector<bool>inliersNew(endNew,true);
    int descIdx {0};
    cv::Mat desc = cv::Mat(endNew, 32, CV_8U);

    for ( size_t i{0}; i < end; i++)
    {
        if ( prevLeftKeys.matchedIdxs[i] < 0)
        {
            inliers[i] = false;
        }
        else
        {
            prevLeftKeys.keyPoints[i] = leftKeys.keyPoints[prevLeftKeys.matchedIdxs[i]];
            prevLeftKeys.estimatedDepth[i] = leftKeys.estimatedDepth[prevLeftKeys.matchedIdxs[i]];
            prevLeftKeys.close[i] = leftKeys.close[prevLeftKeys.matchedIdxs[i]];
            leftKeys.Desc.row(prevLeftKeys.matchedIdxs[i]).copyTo(desc.row(descIdx));
                descIdx++;


            prevLeftKeys.trackCnt[i] ++;
            inliersNew[prevLeftKeys.matchedIdxs[i]] = false;
        }
    }
    reduceTrackedKeys(prevLeftKeys, inliers);
    // reduceTrackedKeys(leftKeys, inliersNew);

    for ( size_t i {0}; i < endNew; i++)
    {
        if ( inliersNew[i])
        {
            prevLeftKeys.keyPoints.emplace_back(leftKeys.keyPoints[i]);
            prevLeftKeys.estimatedDepth.emplace_back(leftKeys.estimatedDepth[i]);
            prevLeftKeys.close.emplace_back(leftKeys.close[i]);
            leftKeys.Desc.row(i).copyTo(desc.row(descIdx));
                descIdx++;
        }
    }
    prevLeftKeys.Desc = desc.clone();
    prevLeftKeys.mapPointIdx.resize(endNew, -1);
    prevLeftKeys.trackCnt.resize(endNew, 0);

}

void  FeatureTracker::cloneTrackedKeys(TrackedKeys& prevLeftKeys, TrackedKeys& leftKeys)
{
    prevLeftKeys.keyPoints = leftKeys.keyPoints;
    prevLeftKeys.Desc = leftKeys.Desc.clone();
    // prevLeftKeys.mapPointIdx = leftKeys.mapPointIdx;
    prevLeftKeys.estimatedDepth = leftKeys.estimatedDepth;
    prevLeftKeys.close = leftKeys.close;
    // prevLeftKeys.trackCnt = leftKeys.trackCnt;
    const size_t prevE{prevLeftKeys.mapPointIdx.size()};
    std::vector<int>newMapPointIdx(prevE, -1);
    std::vector<int>newTrackCnt(prevE, 0);
    for (size_t i{0}; i < prevE; i++)
    {
        if (prevLeftKeys.matchedIdxs[i] >= 0)
        {
            newMapPointIdx[prevLeftKeys.matchedIdxs[i]] = prevLeftKeys.mapPointIdx[i];
            newTrackCnt[prevLeftKeys.matchedIdxs[i]] = prevLeftKeys.trackCnt[i] + 1;

        }
    }
    prevLeftKeys.mapPointIdx = newMapPointIdx;
    prevLeftKeys.trackCnt = newTrackCnt;
}

void FeatureTracker::Track3(const int frames)
{
    for (curFrame = 0; curFrame < frames; curFrame++)
    {
        zedPtr->addKeyFrame = true;
        setLRImages(curFrame);

        TrackedKeys keysLeft;



        if ( curFrame == 0 )
        {
            extractORBStereoMatch(lIm.im, rIm.im, keysLeft);

            prevLeftPnts.clone(keysLeft);
            updateMapPoints(prevLeftPnts);
            // addMapPnts(keysLeft);
            setPreLImage();
            setPreRImage();
            continue;
        }
        extractORBStereoMatch(lIm.im, rIm.im, keysLeft);
        predictORBPoints(prevLeftPnts);
        fm.matchORBPoints(prevLeftPnts, keysLeft);
        // std::vector<cv::Point2f> ppnts, pntsn;
        // cv::KeyPoint::convert(prevLeftPnts.keyPoints, ppnts);
        // cv::KeyPoint::convert(keysLeft.keyPoints, pntsn);
        // prevLeftPnts.inliers.clear();
        // cv::findFundamentalMat(ppnts, pntsn,prevLeftPnts.inliers2);
        drawLeftMatches("left Matches", pLIm.rIm, prevLeftPnts, keysLeft);
        cv::waitKey(1);
        optimizePoseCeres(prevLeftPnts, keysLeft);

        cloneTrackedKeys(prevLeftPnts, keysLeft);
        updateMapPoints(prevLeftPnts);


        setPreLImage();
        setPreRImage();

    }
    datafile.close();
}

void FeatureTracker::initializeMap(TrackedKeys& keysLeft)
{
    activeMapPoints.reserve(keysLeft.keyPoints.size());
    mPPerKeyFrame.reserve(1000);
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        if ( keysLeft.estimatedDepth[i] > 0 )
        {
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = zedPtr->cameraPose.pose * p;
            keysLeft.mapPointIdx[i] = map->pIdx;
            MapPoint* mp = new MapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], keysLeft.close[i], map->kIdx, map->pIdx);
            map->addMapPoint(mp);
            activeMapPoints.emplace_back(mp);
        }
    }
    map->addKeyFrame(zedPtr->cameraPose.pose);
    mPPerKeyFrame.push_back(activeMapPoints.size());
}

void FeatureTracker::worldToImg(std::vector<MapPoint*>& MapPointsVec, std::vector<cv::KeyPoint>& projectedPoints)
{
    projectedPoints.resize(MapPointsVec.size());
    for ( size_t i {0}, end{MapPointsVec.size()}; i < end; i++)
    {
        Eigen::Vector4d p4d =  MapPointsVec[i]->getWordPose4d();

        if (p4d(2) <= 0.0f )
        {
            MapPointsVec[i]->SetInFrame(false);
        }

        const double invfx = 1.0f/fx;
        const double invfy = 1.0f/fy;


        double u {fx*p4d(0)/p4d(2) + cx};
        double v {fy*p4d(1)/p4d(2) + cy};

        const int h = zedPtr->mHeight;
        const int w = zedPtr->mWidth;

        if ( u < 0 || u > w  || v < 0 || v > h )
            MapPointsVec[i]->SetInFrame(false);
        else
            projectedPoints[i].pt = cv::Point2f((float)u, (float)v);
    }
}

void FeatureTracker::worldToImg(std::vector<MapPoint*>& MapPointsVec, std::vector<cv::KeyPoint>& projectedPoints, const Eigen::Matrix4d& currPoseInv)
{
    projectedPoints.resize(MapPointsVec.size());
    for ( size_t i {0}, end{MapPointsVec.size()}; i < end; i++)
    {
        Eigen::Vector4d p4d = currPoseInv * MapPointsVec[i]->getWordPose4d();

        if (p4d(2) <= 0.0f )
        {
            MapPointsVec[i]->SetInFrame(false);
            continue;
        }

        const double invfx = 1.0f/fx;
        const double invfy = 1.0f/fy;


        double u {fx*p4d(0)/p4d(2) + cx};
        double v {fy*p4d(1)/p4d(2) + cy};

        const int h = zedPtr->mHeight;
        const int w = zedPtr->mWidth;

        if ( u < 0 || u > w  || v < 0 || v > h )
            MapPointsVec[i]->SetInFrame(false);
        else
        {
            projectedPoints[i].pt = cv::Point2f((float)u, (float)v);
            MapPointsVec[i]->SetInFrame(true);
        }
    }
}

void FeatureTracker::getPoints3dFromMapPoints(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<cv::Point3d>& points3d, std::vector<cv::Point2d>& points2d, std::vector<int>& matchedIdxsN)
{
    points2d.reserve(matchedIdxsN.size());
    points3d.reserve(matchedIdxsN.size());
    for ( size_t i{0}, end{matchedIdxsN.size()}; i < end; i++)
    {
        if ( matchedIdxsN[i] >= 0 )
        {
            points2d.emplace_back((double)keysLeft.keyPoints[i].pt.x, (double)keysLeft.keyPoints[i].pt.y);
            Eigen::Vector3d p3d = activeMapPoints[matchedIdxsN[i]]->getWordPose3d();
            points3d.emplace_back(p3d(0), p3d(1), p3d(2));
        }
    }
}

void FeatureTracker::removePnPOut(std::vector<int>& idxs, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB)
{
    int idxIdx {0};
    int count {0};
    for ( size_t i{0}, end{matchedIdxsN.size()}; i < end; i++)
    {
        if ( matchedIdxsN[i] >= 0 )
        {
            if (idxs[idxIdx] == count )
                idxIdx++;
            else
            {
                matchedIdxsB[matchedIdxsN[i]] = -1;
                matchedIdxsN[i] = -1;
            }
            count ++;
        }
    }
}

bool FeatureTracker::check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const float weight)
{
    const double invZ = 1.0f/p4d(2);

    const double u {fx*p4d(0)*invZ + cx};
    const double v {fy*p4d(1)*invZ + cy};

    const double errorU = weight * ((double)obs.x - u);
    const double errorV = weight * ((double)obs.y - v);

    const double error = (errorU * errorU + errorV * errorV);
    if (error > thres)
        return true;
    else
        return false;
}

bool FeatureTracker::check3dError(const Eigen::Vector4d& p4d, const Eigen::Vector4d& obs, const double thres, const float weight)
{
    const double errorX = weight * ((double)obs(0) - p4d(0));
    const double errorY = weight * ((double)obs(1) - p4d(1));
    const double errorZ = weight * ((double)obs(2) - p4d(2));

    const double error = (errorX * errorX + errorY * errorY + errorZ * errorZ);
    if (error > thres)
        return true;
    else
        return false;
}

int FeatureTracker::OutliersReprojErr(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsB, const double thres, const std::vector<float>& weights, int& nInliers)
{
    // std::vector<cv::Point2f>out2d;
    int nStereo = 0;
    for (size_t i {0}, end{matchedIdxsB.size()}; i < end; i++)
    {
        if ( matchedIdxsB[i] < 0 )
            continue;
        MapPoint* mp = activeMapPoints[i];
        if ( !mp->GetInFrame() )
        {
            mp->SetIsOutlier(true);
            continue;
        }
        cv::Point2f out;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        p4d = estimatedP * p4d;
        const int nIdx {matchedIdxsB[i]};
        if (activeMapPoints[i]->close && keysLeft.estimatedDepth[nIdx] > 0)
        {
            Eigen::Vector4d obs;
            get3dFromKey(obs, keysLeft.keyPoints[nIdx], keysLeft.estimatedDepth[nIdx]);
            bool outlier = check3dError(p4d, obs, thres, 1.0);
            mp->SetIsOutlier(outlier);
            if ( outlier )
            {
                nInliers--;
            }
            else
            {

                // Logging("obs", obs,3);
                // Logging("point", p4d,3);
                nStereo++;
            }

        }
        else
        {
            bool outlier = check2dError(p4d, keysLeft.keyPoints[nIdx].pt, thres, 1.0);
            mp->SetIsOutlier(outlier);
            if ( outlier )
                nInliers--;
            else
            {
                if ( mp->close )
                    nStereo++;
            }

        }
    }

// #if PROJECTIM
//     // drawOptical("new", pLIm.rIm,pnts.left, pnts.newPnts);
//     drawOptical("reproj", pLIm.rIm,obs, out2d);
//     cv::waitKey(waitTrials);
// #endif
    return nStereo;
}

std::pair<int,int> FeatureTracker::refinePose(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsB, Eigen::Matrix4d& estimPose)
{


    const size_t prevS { activeMapPoints.size()};
    const size_t newS { keysLeft.keyPoints.size()};
    const size_t startPrev {newS - prevS};
    std::vector<bool> inliers(prevS,true);
    std::vector<bool> prevInliers(prevS,true);

    std::vector<float> weights;
    // calcWeights(prePnts, weights);
    weights.resize(prevS, 1.0f);
    std::vector<double>thresholds = {15.6f,9.8f,7.815f,7.815f};
    double thresh = 7.815f;
    int nIn {0};

    ceres::Problem problem;
    Eigen::Vector3d frame_tcw;
    Eigen::Quaterniond frame_qcw;
    // OutliersReprojErr(estimPose, activeMapPoints, keysLeft, matchedIdxsB, thresh, weights, nIn);
    Eigen::Matrix4d frame_pose = estimPose;
    Eigen::Matrix3d frame_R;
    frame_R = frame_pose.block<3, 3>(0, 0);
    frame_tcw = frame_pose.block<3, 1>(0, 3);
    frame_qcw = Eigen::Quaterniond(frame_R);
    std::vector<cv::Point2f> found, observed;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(7.815f));
    for (size_t i{0}, end{matchedIdxsB.size()}; i < end; i++)
    {
        if ( matchedIdxsB[i] < 0 )
            continue;
        MapPoint* mp = activeMapPoints[i];
        if  ( !mp->GetInFrame() || mp->GetIsOutlier() )
            continue;
        nIn ++;
        const int nIdx {matchedIdxsB[i]};
        if (mp->close && keysLeft.estimatedDepth[nIdx] > 0)
        {
            Eigen::Vector4d np4d;
            get3dFromKey(np4d, keysLeft.keyPoints[nIdx], keysLeft.estimatedDepth[nIdx]);
            Eigen::Vector3d obs(np4d(0), np4d(1),np4d(2));
            Eigen::Vector3d point = mp->getWordPose3d();
            Eigen::Vector4d point4d = mp->getWordPose4d();
            point4d = estimPose * point4d;
            const double u {fx*point4d(0)/point4d(2) + cx};
            const double v {fy*point4d(1)/point4d(2) + cy};
            found.emplace_back((float)u, (float)v);
            observed.emplace_back(keysLeft.keyPoints[nIdx].pt);
            if ( point4d(2) <= 0 || std::isnan(keysLeft.keyPoints[nIdx].pt.x) || std::isnan(keysLeft.keyPoints[nIdx].pt.y))
                Logging("out", point4d,3);
            if (u < 0 || v < 0 || v > zedPtr->mHeight || u > zedPtr->mWidth)
                Logging("out", point4d,3);
            // Logging("obs", obs,3);
            // Logging("point", point,3);
            ceres::CostFunction* costf = OptimizePoseICP::Create(K, point, obs, (double)weights[i]);
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }
        else
        {
            Eigen::Vector2d obs((double)keysLeft.keyPoints[nIdx].pt.x, (double)keysLeft.keyPoints[nIdx].pt.y);
            Eigen::Vector3d point = mp->getWordPose3d();
            Eigen::Vector4d point4d = mp->getWordPose4d();
            point4d = estimPose * point4d;
            const double u {fx*point4d(0)/point4d(2) + cx};
            const double v {fy*point4d(1)/point4d(2) + cy};
            found.emplace_back((float)u, (float)v);
            observed.emplace_back(keysLeft.keyPoints[nIdx].pt);
            if ( point4d(2) <= 0 || std::isnan(keysLeft.keyPoints[nIdx].pt.x) || std::isnan(keysLeft.keyPoints[nIdx].pt.y))
                Logging("out", point4d,3);
            if (u < 0 || v < 0 || v > zedPtr->mHeight || u > zedPtr->mWidth)
                Logging("out", point4d,3);
            
            ceres::CostFunction* costf = OptimizePose::Create(K, point, obs, (double)weights[i]);
            // Logging("obs", obs,3);
            // Logging("point", point,3);
            problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

            problem.SetManifold(frame_qcw.coeffs().data(),
                                        quaternion_local_parameterization);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.use_explicit_schur_complement = true;
    options.max_num_iterations = 100;
    // options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // Logging("ceres report", summary.FullReport(),3);
    Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
    estimPose.block<3, 3>(0, 0) = R;
    estimPose.block<3, 1>(0, 3) = frame_tcw;

    drawPointsTemp<cv::Point2f, cv::Point2f>("ceres", pLIm.rIm, found, observed);
    cv::waitKey(1);
    int nStereo = OutliersReprojErr(estimPose, activeMapPoints, keysLeft, matchedIdxsB, thresh, weights, nIn);
    // Logging("pose", estimPose,3);
    return std::pair<int,int>(nStereo, nIn);
}

void FeatureTracker::addKeyFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN)
{
    activeMapPoints.reserve(activeMapPoints.size() + keysLeft.keyPoints.size());
    mPPerKeyFrame.reserve(1000);
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        // if ( keysLeft.close[i] >  )
        if ( matchedIdxsN[i] >= 0 )
            continue;
        if ( keysLeft.estimatedDepth[i] > 0 )
        {
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = zedPtr->cameraPose.pose * p;
            MapPoint* mp = new MapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], keysLeft.close[i], map->kIdx, map->pIdx);
            map->addMapPoint(mp);
            activeMapPoints.emplace_back(mp);
        }
    }
    mPPerKeyFrame.push_back(activeMapPoints.size());
    map->addKeyFrame(zedPtr->cameraPose.pose);

}

void FeatureTracker::removeMapPointOut(std::vector<MapPoint*>& activeMapPoints, const Eigen::Matrix4d& estimPose)
{
    const size_t end{activeMapPoints.size()};
    std::vector<int> inliers(end, true);
    int j {0};
    for ( size_t i {0}; i < end; i++)
    {
        if ( activeMapPoints[i]->GetIsOutlier() || !activeMapPoints[i]->GetInFrame())
            continue;
        Eigen::Vector4d point = activeMapPoints[i]->getWordPose4d();
        point = estimPose * point;

        if ( point(2) <= 0.0 )
        {
            activeMapPoints[i]->SetInFrame(false);
            continue;
        }
        const double invZ = 1.0f/point(2);

        const double u {fx*point(0)*invZ + cx};
        const double v {fy*point(1)*invZ + cy};

        const int h {zedPtr->mHeight};
        const int w {zedPtr->mWidth};

        if ( u < 0 || v < 0 || u >= w || v >= h)
        {
            activeMapPoints[i]->SetInFrame(false);
            continue;
        }
        activeMapPoints[i]->addTCnt();
        activeMapPoints[j++] = activeMapPoints[i];
    }
    activeMapPoints.resize(j);
}

bool FeatureTracker::checkDisplacement(const Eigen::Matrix4d& currPose, Eigen::Matrix4d& estimPose, const double threshold)
{
    const double errorX = currPose(0,3) - estimPose(0,3);
    const double errorY = currPose(1,3) - estimPose(1,3);
    const double errorZ = currPose(2,3) - estimPose(2,3);

    const double error = errorX * errorX + errorY * errorY + errorZ * errorZ;

    if (error > threshold)
    {
        estimPose = currPose;
        return true;
    }
    else
        return false;
}

void FeatureTracker::Track4(const int frames)
{
    for (curFrame = 0; curFrame < frames; curFrame++)
    {
        zedPtr->addKeyFrame = true;
        setLRImages(curFrame);

        TrackedKeys keysLeft;



        if ( curFrame == 0 )
        {
            extractORBStereoMatch(lIm.im, rIm.im, keysLeft);

            initializeMap(keysLeft);
            // addMapPnts(keysLeft);
            setPreLImage();
            setPreRImage();
            continue;
        }
        extractORBStereoMatch(lIm.im, rIm.im, keysLeft);

        Eigen::Matrix4d currPose = predNPoseInv;
        Eigen::Matrix4d prevPose = zedPtr->cameraPose.poseInverse;
        std::vector<int> matchedIdxsN(keysLeft.keyPoints.size(), -1);
        std::vector<int> matchedIdxsB(activeMapPoints.size(), -1);

        
        if ( curFrame == 1 )
            int nMatches = fm.matchByProjection(activeMapPoints, keysLeft, matchedIdxsN, matchedIdxsB);
        else
        {
            std::vector<cv::KeyPoint> ConVelPoints;
            worldToImg(activeMapPoints, ConVelPoints, predNPoseInv);
            int nNewMatches = fm.matchByProjectionConVel(activeMapPoints, ConVelPoints, keysLeft, matchedIdxsN, matchedIdxsB, 2);
        }

        std::vector<cv::Point3d> points3d;
        std::vector<cv::Point2d> points2d;
        getPoints3dFromMapPoints(activeMapPoints,keysLeft, points3d, points2d, matchedIdxsN);
        cv::Mat Rvec, tvec;
        Converter::convertEigenPoseToMat(currPose, Rvec, tvec);
        std::vector<int>idxs;
        cv::solvePnPRansac(points3d, points2d, zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F), Rvec, tvec,true,100, 4.0f, 0.99, idxs);
        Eigen::Matrix4d estimPose = Converter::convertRTtoPose(Rvec, tvec);

        if (!checkDisplacement(currPose, estimPose, 10.0))
            removePnPOut(idxs, matchedIdxsN, matchedIdxsB);
        

        Logging("prednpose ", predNPoseInv,3);


        Logging("ransac pose", estimPose,3);

        refinePose(activeMapPoints, keysLeft, matchedIdxsB, estimPose);

        Logging("first refine pose", estimPose,3);

        // set outliers

        // after last refine check all matches, change outliers to inliers if they are no more, and in the end remove all outliers from vector. they are already saved on mappoints.

        // the outliers from first refine are not used on the next refines.

        // check for big displacement, if there is, use constant velocity model


        // std::cout << estimPose << std::endl;


        // std::vector<cv::Point2f> mpnts, pnts2f;
        // for ( size_t i {0}, end{activeMapPoints.size()}; i < end; i++)
        // {
        //     if ( matchedIdxsB[i] >= 0)
        //     {
        //         mpnts.emplace_back(activeMapPoints[i]->obs[0].pt);
        //         pnts2f.emplace_back(keysLeft.keyPoints[matchedIdxsB[i]].pt);
        //     }
        // }
        // drawPointsTemp<cv::Point2f>("matches left Pleft", pLIm.rIm, mpnts, pnts2f);
        // cv::waitKey(1);


        std::vector<cv::KeyPoint> projectedPoints;
        worldToImg(activeMapPoints, projectedPoints, estimPose);
        int nNewMatches = fm.matchByProjectionPred(activeMapPoints, projectedPoints, keysLeft, matchedIdxsN, matchedIdxsB, 3);

        std::vector<cv::Point2f> mpnts, pnts2f;
        for ( size_t i {0}, end{activeMapPoints.size()}; i < end; i++)
        {
            if ( projectedPoints[i].pt.x > 0)
            {
                if ( matchedIdxsB[i] >= 0)
                {
                    mpnts.emplace_back(keysLeft.keyPoints[matchedIdxsB[i]].pt);
                    pnts2f.emplace_back(projectedPoints[i].pt);
                }
            }
        }

        drawPointsTemp<cv::Point2f>("Projected", lIm.rIm, mpnts, pnts2f);
        cv::waitKey(1);

        std::vector<cv::KeyPoint> prevProjectedPoints;
        worldToImg(activeMapPoints, prevProjectedPoints, prevPose);
        std::vector<cv::Point2f> Nmpnts, Npnts2f;
        for ( size_t i {0}, end{activeMapPoints.size()}; i < end; i++)
        {
            if ( matchedIdxsB[i] >= 0)
            {
                if ( prevProjectedPoints[i].pt.x > 0)
                {
                    Nmpnts.emplace_back(prevProjectedPoints[i].pt);
                    Npnts2f.emplace_back(keysLeft.keyPoints[matchedIdxsB[i]].pt);
                }
            }
        }
        drawPointsTemp<cv::Point2f>("NEWW matches left Pleft", lIm.rIm, Nmpnts, Npnts2f);
        cv::waitKey(1);

        std::pair<int,int> nStIn = refinePose(activeMapPoints, keysLeft, matchedIdxsB, estimPose);

        Logging("second refine pose", estimPose,3);

        Logging("nIn", nStIn.second,3);
        Logging("nStereo", nStIn.first,3);

        // cv::Rodrigues(Rvec, Rvec);
        // estimPose = Converter::convertRTtoPose(Rvec, tvec);

        removeMapPointOut(activeMapPoints, estimPose);


        poseEst = estimPose.inverse();
        // Logging("estimPose after inv", poseEst,3);

        publishPoseNew();
        // if ( nStIn.first < 200 )
        // {
        addKeyFrame(keysLeft, matchedIdxsN);
        Logging("I AM IIIIIIIIIIN", activeMapPoints.size(),3);
        // }

        setPreLImage();
        setPreRImage();

        // Remove only out of frame mappoints

        // maybe after last refine triangulate with the new refined pose the far stereo keypoints.

        // ICP only ON CLOSE KEYPOINTS


        // cv::Mat R,t;
        // std::cout << R << std::endl << t << std::endl;

        // projected points ( only those that are not matched or are outliers )

        // fm.matchByProjection(currPose) // overloaded function that does not calculate vector of idxs

        // motion only ba only if keysleft.close = true you do icp if not then do 3d-2d





        // predictORBPoints(prevLeftPnts);
        // fm.matchORBPoints(prevLeftPnts, keysLeft);
        // // std::vector<cv::Point2f> ppnts, pntsn;
        // // cv::KeyPoint::convert(prevLeftPnts.keyPoints, ppnts);
        // // cv::KeyPoint::convert(keysLeft.keyPoints, pntsn);
        // // prevLeftPnts.inliers.clear();
        // // cv::findFundamentalMat(ppnts, pntsn,prevLeftPnts.inliers2);
        // drawLeftMatches("left Matches", pLIm.rIm, prevLeftPnts, keysLeft);
        // cv::waitKey(1);
        // optimizePoseCeres(prevLeftPnts, keysLeft);

        // cloneTrackedKeys(prevLeftPnts, keysLeft);
        // updateMapPoints(prevLeftPnts);


        

    }
    datafile.close();
}

void FeatureTracker::beginTrackingGoodFeatures(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        curFrame = frame;
        setLRImages(frame);
        // fm.checkDepthChange(pLIm.im,pRIm.im,prePnts);
        if ( (addFeatures || uStereo < mnSize || cv::norm(pTvec)*zedPtr->mFps > highSpeed) && ( uStereo < mxSize) )
        {
            // Logging("ptvec",pTvec,3);
            // Logging("cv::norm(pTvec)",cv::norm(pTvec),3);

            zedPtr->addKeyFrame = true;
            // if ( uMono > mxMonoSize )
            //     updateKeysClose(frame);
            // else
            updateKeysGoodFeatures(frame);
            fd.compute3DPoints(prePnts, keyNumb);
            keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
            keyNumb ++;
            
        }
        
        // opticalFlow();
        if ( curFrame == 1 )
            opticalFlow();
        else
            opticalFlowPredict();

        // Logging("addf", addFeatures,3);

        // getSolvePnPPoseWithEss();

        // getPoseCeres();
        getPoseCeresNew();

        setPreTrial();

        addFeatures = checkFeaturesArea(prePnts);
        // addFeatures = checkFeaturesAreaCont(prePnts);
        Logging("ustereo", uStereo,3);
        Logging("umono", uMono,3);
    }
    datafile.close();
}

void FeatureTracker::getWeights(std::vector<float>& weights, std::vector<cv::Point2d>& p2Dclose)
{
    const size_t end {prePnts.left.size()};
    weights.reserve(end);
    p2Dclose.reserve(end);
    const float vd {zedPtr->mBaseline * 40};
    const float sig {vd};
    uStereo = 0;
    for (size_t i {0}; i < end; i++)
    {
        p2Dclose.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
        if ( prePnts.depth[i] < vd)
        {
            uStereo ++;
            weights.emplace_back(1.0f);
        }
        else
        {
            float prob = norm_pdf(prePnts.depth[i], vd, sig);
            weights.emplace_back(2 * prob * vd);
        }
    }
}

float FeatureTracker::norm_pdf(float x, float mu, float sigma)
{
	return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-(pow((x - mu)/sigma, 2)/2.0));
}

bool FeatureTracker::checkFeaturesArea(const SubPixelPoints& prePnts)
{
    const size_t end{prePnts.left.size()};
    const int sep {3};
    std::vector<int> gridCount;
    gridCount.resize(sep * sep);
    const int wid {(int)zedPtr->mWidth/sep + 1};
    const int hig {(int)zedPtr->mHeight/sep + 1};
    for (size_t i{0};i < end; i++)
    {
        const int w {(int)prePnts.left[i].x/wid};
        const int h {(int)prePnts.left[i].y/hig};
        gridCount[(int)(h + sep*w)] += 1;
    }
    const int mnK {fe.numberPerCell/2};
    const int mnG {7};
    const size_t endgr {gridCount.size()};
    int count {0};
    for (size_t i{0}; i < endgr; i ++ )
    {
        if ( gridCount[i] > mnK)
            count ++;
    }
    if ( count < mnG)
        return true;
    else
        return false;
}

bool FeatureTracker::checkFeaturesAreaCont(const SubPixelPoints& prePnts)
{
    static int skip = 0;
    const size_t end{prePnts.left.size()};
    const int sep {3};
    std::vector<int> gridCount;
    gridCount.resize(sep * sep);
    const int wid {(int)zedPtr->mWidth/sep + 1};
    const int hig {(int)zedPtr->mHeight/sep + 1};
    for (size_t i{0};i < end; i++)
    {
        const int w {(int)prePnts.left[i].x/wid};
        const int h {(int)prePnts.left[i].y/hig};
        gridCount[(int)(h + sep*w)] += 1;
    }
    const int mnK {10};
    const int mnmxG {7};
    const int mnG {3};
    const size_t endgr {gridCount.size()};
    int count {0};
    for (size_t i{0}; i < endgr; i ++ )
    {
        if ( gridCount[i] > mnK)
            count ++;
    }
    if ( count < mnmxG)
        skip++;
    else if (count < mnG)
        return true;
    else
        skip = 0;
    Logging("skip", skip,3);
    Logging("count", count,3);
    if ( skip > 2 || skip == 0)
        return false;
    else
        return true;
}

void FeatureTracker::getEssentialPose()
{
    cv::Mat Rvec(3,3,CV_64F), tvec(3,1,CV_64F);
    std::vector <uchar> inliers;
    std::vector<cv::Point2f> p, pp;
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    
    
    cv::undistortPoints(pnts.left,p,zedPtr->cameraLeft.cameraMatrix, dist);
    cv::undistortPoints(prePnts.left,pp,zedPtr->cameraLeft.cameraMatrix, dist);
    cv::Mat E = cv::findEssentialMat(prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC,0.99,0.1, inliers);
    if (!inliers.empty())
    {
        prePnts.reduce<uchar>(inliers);
        pnts.reduce<uchar>(inliers);
        reduceVectorTemp<cv::Point2f,uchar>(p,inliers);
        reduceVectorTemp<cv::Point2f,uchar>(pp,inliers);
    }
    uStereo = prePnts.left.size();
    if (uStereo > 10)
    {
        cv::Mat R1,R2,t;
        cv::decomposeEssentialMat(E, R1, R2,t);
        if (cv::norm(prevR,R1) > cv::norm(prevR,R2))
            Rvec = R2;
        else
            Rvec = R1;
        prevR = Rvec.clone();
        tvec = -t/10;
        convertToEigen(Rvec,tvec,poseEstFrame);
        Logging("R1",R1,3);
        Logging("R2",R2,3);
        publishPose();

    }
    
}

void FeatureTracker::getSolvePnPPose()
{

    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;
    std::vector<cv::Point2d> outp2D;
    p3D.reserve(end);
    p2D.reserve(end);
    outp2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            if (prePnts.useable[i])
            {
                inliers[i] = true;
                p3D.emplace_back(point);
                p2D.emplace_back(pnts.points2D[i]);
                outp2D.emplace_back(p2dtemp);
            }
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
    // cv::projectPoints(p3D,cv::Mat::eye(3,3, CV_64F),cv::Mat::zeros(3,1, CV_64F),zedPtr->cameraLeft.cameraMatrix,cv::Mat::zeros(5,1, CV_64F),outp2D);
    inliers.clear();
    const size_t endproj{p3D.size()};
    inliers.resize(endproj);
    const int wid {zedPtr->mWidth - 1};
    const int hig {zedPtr->mHeight - 1};
    for (size_t i{0};i < endproj; i++)
    {
        if (!(outp2D[i].x > wid || outp2D[i].x < 0 || outp2D[i].y > hig || outp2D[i].y < 0))
            inliers[i] = true;
    }

    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
    reduceVectorTemp<cv::Point2d,bool>(outp2D,inliers);
    reduceVectorTemp<cv::Point2d,bool>(p2D,inliers);
    reduceVectorTemp<cv::Point3d,bool>(p3D,inliers);

    std::vector<uchar> check;
    cv::findFundamentalMat(outp2D, p2D, check, cv::FM_RANSAC, 1, 0.99);

    prePnts.reduce<uchar>(check);
    pnts.reduce<uchar>(check);
    reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    reduceVectorTemp<cv::Point3d,uchar>(p3D,check);

    uStereo = p3D.size();
    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);
    if (uStereo > 10)
    {
        //  cv::solvePnP(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true);
        check.clear();
        cv::solvePnPRansac(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true,100,2.0f, 0.999, check);

    }

    // prePnts.reduce<uchar>(check);
    // pnts.reduce<uchar>(check);
    // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point3d,uchar>(p3D,check);
    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    Logging("norm",cv::norm(tvec,pTvec),3);
    Logging("normr",cv::norm(Rvec,pRvec),3);
    if (cv::norm(tvec,pTvec) + cv::norm(Rvec,pRvec) > 1)
    {
        tvec = pTvec;
        Rvec = pRvec;
    }

    if (p3D.size() > mnInKal)
    {
        lkal.fillMeasurements(measurements, tvec, Rvec);
    }
    else
    {
        Logging("less than 50","",3);
    }

    pTvec = tvec;
    pRvec = Rvec;

    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    Logging("measurements",measurements,3);
    Logging("rot",rotation_estimated,3);
    Logging("tra",translation_estimated,3);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();
#if PROJECTIM
    draw2D3D(pLIm.rIm, outp2D, p2D);
#endif
}

void FeatureTracker::getSolvePnPPoseWithEss()
{

    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;
    std::vector<cv::Point2d> pp2Dess;
    std::vector<cv::Point2d> p3Dp2D;
    std::vector<cv::Point2d> outp2D;
    p3D.reserve(end);
    p3Dp2D.reserve(end);
    p2D.reserve(end);
    outp2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            outp2D.emplace_back(pnts.left[i]);
            pp2Dess.emplace_back(prePnts.left[i]);
            if (prePnts.useable[i])
            {

                p3D.emplace_back(point);
                p3Dp2D.emplace_back(p2dtemp);
                p2D.emplace_back(pnts.left[i]);
            }
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);


    // inliers.clear();
    // const size_t endproj{p3D.size()};
    // inliers.resize(endproj);
    // const int wid {zedPtr->mWidth - 1};
    // const int hig {zedPtr->mHeight - 1};
    // for (size_t i{0};i < endproj; i++)
    // {
    //     if (!(outp2D[i].x > wid || outp2D[i].x < 0 || outp2D[i].y > hig || outp2D[i].y < 0))
    //         inliers[i] = true;
    // }

    // prePnts.reduce<bool>(inliers);
    // pnts.reduce<bool>(inliers);
    // reduceVectorTemp<cv::Point2d,bool>(outp2D,inliers);
    // reduceVectorTemp<cv::Point2d,bool>(p2D,inliers);
    // reduceVectorTemp<cv::Point2d,bool>(pp2Dess,inliers);
    // reduceVectorTemp<cv::Point3d,bool>(p3D,inliers);

    // std::vector<uchar> check;
    // cv::findFundamentalMat(p3Dp2D, p2D, check, cv::FM_RANSAC, 1, 0.999);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p3Dp2D,check);
    // reduceVectorTemp<cv::Point3d,uchar>(p3D,check);
    // prePnts.reduce<uchar>(check);
    // pnts.reduce<uchar>(check);
    // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(pp2Dess,check);
    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);

    // cv::Mat E = cv::findEssentialMat(pp2Dess,outp2D,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC, 0.999,1.0f);
    // cv::Mat R1es,R2es,tes;
    // cv::decomposeEssentialMat(E,R1es, R2es, tes);
    // cv::Rodrigues(R1es,R1es);
    // cv::Rodrigues(R2es,R2es);
    // const double norm1 {cv::norm(Rvec,R1es)};
    // const double norm2 {cv::norm(Rvec,R2es)};

    // if (norm1 > norm2)
    //     Rvec = R2es;
    // else
    //     Rvec = R1es;

    uStereo = p3D.size();
    if (uStereo > 10)
    {
        //  cv::solvePnP(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true);
        std::vector<int>idxs;
        cv::solvePnPRansac(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true,100,2.0f, 0.99, idxs);

        reduceVectorInliersTemp<cv::Point2d,int>(p2D,idxs);
        reduceVectorInliersTemp<cv::Point2d,int>(p3Dp2D,idxs);
        reduceVectorInliersTemp<cv::Point3d,int>(p3D,idxs);
        // cv::solvePnPRefineLM(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec);
        // prePnts.reduce<uchar>(check);
        // pnts.reduce<uchar>(check);
        // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
        // reduceVectorTemp<cv::Point2d,uchar>(pp2Dess,check);

    }

    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    if (cv::norm(tvec,pTvec) + cv::norm(Rvec,pRvec) > 1)
    {
        tvec = pTvec;
        Rvec = pRvec;
    }

    if (p3D.size() > mnInKal)
    {
        lkal.fillMeasurements(measurements, tvec, Rvec);
    }
    else
    {
        Logging("less than ",mnInKal,3);
    }

    pTvec = tvec;
    pRvec = Rvec;

    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();
#if PROJECTIM
    draw2D3D(pLIm.rIm,p3Dp2D, p2D);
#endif
}

void FeatureTracker::getPoseCeres()
{

    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;

    get3dPointsforPoseAll(p3D, p2D);

    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);

    essForMonoPose(Rvec, tvec,p3D);

    if (p3D.size() > 10)
    {
        pnpRansac(Rvec, tvec, p3D, p2D);
    }
    // uStereo = p3D.size();
    poseEstKal(Rvec, tvec, p3D.size());

}

void FeatureTracker::getPoseCeresNew()
{

    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;

    get3dPointsforPoseAll(p3D, p2D);
    // std::vector<uchar>err;
    // cv::findFundamentalMat(p2D,pnts.left,err,cv::FM_RANSAC,3,0.99);
    // reduceVectorTemp<cv::Point3d,uchar>(p3D, err);
    // pnts.reduce<uchar>(err);
    // prePnts.reduce<uchar>(err);

    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = pTvec.clone();


    // essForMonoPose(Rvec, tvec, p3D);

    if (p3D.size() > 10)
    {
        pnpRansac(Rvec, tvec, p3D, p2D);
    }
    // optimizePoseMO(p3D, Rvec, tvec);
    // if (abs(Rvec.at<double>(1)) > 0.04)
    //     bigRot = true;
    // else
    //     bigRot = false;
    // uStereo = p3D.size();
    poseEstKal(Rvec, tvec, uStereo);

}

void FeatureTracker::estimatePose(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2f>& curPnts)
{
    cv::Mat Rvec = pRvec.clone();
    cv::Mat tvec = pTvec.clone();
    std::vector<cv::Point3d> close3D;
    std::vector<cv::Point2d> curPntsd;
    if (p3D.size() > 10)
    {
        curPntsd.reserve(curPnts.size());
        std::vector<cv::Point2f>::const_iterator it, end(curPnts.end());
        for (it = curPnts.begin(); it != end; it++)
            curPntsd.emplace_back((double)it->x, (double)it->y);


        std::vector<int>idxs;
        cv::solvePnPRansac(p3D, curPntsd,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true,100, 8.0f, 0.99, idxs);
        reduceVectorInliersTemp<cv::Point3d,int>(p3D, idxs);
        reduceVectorInliersTemp<cv::Point2d,int>(curPntsd, idxs);
        reduceVectorInliersTemp<int,int>(reprojIdxs, idxs);

        // ceresClose(p3D, curPntsd,Rvec,tvec);

        std::vector<cv::Point3d>::const_iterator p, pend(p3D.end());
        uStereo = 0;
        for (p = p3D.begin(); p != pend; p++)
            if ( p->z < zedPtr->mBaseline * 40)
                uStereo ++;
        uMono = p3D.size() - uStereo;
#if PROJECTIM
        std::vector<cv::Point2d> p2D, pn2D;

        compute2Dfrom3D(p3D, p2D, pn2D);

        drawPointsTemp<cv::Point2d, cv::Point2d>("solvepnp",pLIm.rIm,p2D,curPntsd);
#endif
    }
    poseEstKal(Rvec, tvec, uStereo);

    removeOutliers(curPntsd);
    

}

void FeatureTracker::removeOutliers(const std::vector<cv::Point2d>& curPntsd)
{
    std::vector<bool>in;
    const size_t endre {prePnts.left.size()};
    const int w {zedPtr->mWidth};
    const int h {zedPtr->mHeight};
    in.resize(endre,true);
    std::vector<cv::Point2d>rep, repnew;

    int count {0};
    for ( size_t i {0}; i < endre; i++)
    {
        cv::Point2d p2calc;
        cv::Point3d p3cam;
        bool inFr {wPntToCamPose(prePnts.points3D[i],p2calc, p3cam)};
        if ( i == reprojIdxs[count] )
        {
            if ( inFr ) 
            {
                if ( pointsDistTemp<cv::Point2d>(p2calc, curPntsd[count]) > 64 )
                    in[i] = false;
                else
                {
                    if ( !(curPntsd[count].x > w || curPntsd[count].x < 0 || curPntsd[count].y > h || curPntsd[count].y < 0) )
                    {
                        prePnts.left[i] = cv::Point2f((float)curPntsd[count].x,(float)curPntsd[count].y);
                        rep.emplace_back(p2calc);
                        repnew.emplace_back(curPntsd[count]);
                    }
                    else
                        in[i] = false;
                }
            }
            else
                in[i] = false;
            count ++;
        }
        else
        {
            if ( inFr )
                prePnts.left[i] = cv::Point2f((float)p2calc.x,(float)p2calc.y);
            else
                in[i] = false;
        }
    }

    // for ( auto& idx:reprojIdxs )
    // {
    //     cv::Point2d p2calc;
    //     wPntToCamPose(prePnts.points3D[idx],p2calc);
    //     // rep.emplace_back(p2calc);
    //     if ( pointsDistTemp<cv::Point2d>(p2calc, curPntsd[count]) > 64 || p2calc.x > zedPtr->mWidth || p2calc.x < 0 || p2calc.y > zedPtr->mHeight || p2calc.y < 0)
    //         in[idx] = false;
    //     else
    //     {
    //         prePnts.left[idx] = cv::Point2f((float)curPntsd[count].x,(float)curPntsd[count].y);
    //         rep.emplace_back(p2calc);
    //         repnew.emplace_back(curPntsd[count]);
    //     }
    //     count ++;
    // }
#if PROJECTIM
    drawPointsTemp<cv::Point2d, cv::Point2d>("reproj error",lIm.rIm,rep,repnew);
#endif
    prePnts.reduce<bool>(in);

}

void FeatureTracker::setTrackedLeft(std::vector<cv::Point2d>& curPntsd)
{
    const size_t endre {prePnts.points3D.size()};

    int count {0};
    for ( auto& idx:reprojIdxs )
    {
        prePnts.left[idx] = cv::Point2f((float)curPntsd[count].x,(float)curPntsd[count].y);
        count ++;
    }

}

int FeatureTracker::calcNumberOfStereo()
{
    int count {0};
    const size_t end {prePnts.left.size()};
    for (size_t i{0}; i < end; i++)
    {
        if ( prePnts.useable[i] )
            count ++;
    }
    return count;
}

void FeatureTracker::optimizePoseMotionOnly(std::vector<cv::Point3d>& p3D, cv::Mat& Rvec, cv::Mat& tvec)
{
    std::vector<cv::Point3d>p3Dclose;
    std::vector<cv::Point2d>p2Dclose;
    get3DClose(p3D,p3Dclose, p2Dclose);
    uStereo = p3Dclose.size();
    uMono = p3D.size() - uStereo;
    
    // ceresRansac(p3Dclose, p2Dclose, Rvec, tvec);
    ceresClose(p3Dclose, p2Dclose, Rvec, tvec);
    // ceresMO(p3Dclose, p2Dclose, Rvec, tvec);

    checkKeyDestrib(p2Dclose);
}

void FeatureTracker::optimizePoseMO(std::vector<cv::Point3d>& p3D, cv::Mat& Rvec, cv::Mat& tvec)
{
    std::vector<cv::Point2d>p2Dclose;
    std::vector<float> weights;
    getWeights(weights, p2Dclose);

    uMono = p3D.size() - uStereo;
    
    // ceresRansac(p3Dclose, p2Dclose, Rvec, tvec);
    ceresWeights(p3D, p2Dclose, Rvec, tvec, weights);
    // ceresMO(p3Dclose, p2Dclose, Rvec, tvec);

    // checkKeyDestrib(p2Dclose);
}

void FeatureTracker::ceresWeights(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec, std::vector<float>& weights)
{
    ceres::Problem problem;
    // make initial guess
    double cameraR[3] {Rvec.at<double>(0), Rvec.at<double>(1), Rvec.at<double>(2)};
    double cameraT[3] {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
    size_t end {p3Dclose.size()};
    problem.AddParameterBlock(cameraR,3);
    problem.AddParameterBlock(cameraT,3);
    for (size_t i{0}; i < end; i++)
    {
        ceres::CostFunction* costf = ReprojectionErrorWeighted::Create(p3Dclose[i],p2Dclose[i], (double)weights[i]);
        
        problem.AddResidualBlock(costf, new ceres::HuberLoss(10.0) /* squared loss */, cameraR, cameraT);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    Rvec.at<double>(0) = cameraR[0];
    Rvec.at<double>(1) = cameraR[1];
    Rvec.at<double>(2) = cameraR[2];
    tvec.at<double>(0) = cameraT[0];
    tvec.at<double>(1) = cameraT[1];
    tvec.at<double>(2) = cameraT[2];
}

void FeatureTracker::checkKeyDestrib(std::vector<cv::Point2d>& p2Dclose)
{
    const int sep {2};
    const int w {zedPtr->mWidth/sep};
    const int h {zedPtr->mHeight/sep};
    std::vector<int> grids;
    grids.resize(sep * sep);
    const size_t end {prePnts.left.size()};

    for (size_t i{0}; i < end; i++)
    {
        int x {(int)prePnts.left[i].x/w};
        int y {(int)prePnts.left[i].y/h};
        grids[(int)(x + sep * y)] += 1;
    }
    for (size_t i {0}; i < sep * sep; i++)
    {
        Logging("grid",i,3);
        Logging("",grids[i],3);
    }
    grids.clear();
    grids.resize(sep * sep);
    const size_t end2 {p2Dclose.size()};
    for (size_t i{0}; i < end2; i++)
    {
        int x {(int)p2Dclose[i].x/w};
        int y {(int)p2Dclose[i].y/h};
        grids[(int)(x + sep * y)] += 1;
    }
    
    for (size_t i {0}; i < sep * sep; i++)
    {
        Logging("grid3d",i,3);
        Logging("",grids[i],3);
    }

}

void FeatureTracker::ceresRansac(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec)
{
    std::vector<int>idxVec;
    getIdxVec(idxVec, p3Dclose.size());

    float mnError {INFINITY};
    int earlyTerm {0};

    double outCamera[6];

    for (size_t i{0}; i < mxIter ; i++)
    {
        ceres::Problem problem;
        std::set<int> idxs;
        getSamples(idxVec, idxs);
        // make initial guess

        double camera[6] {Rvec.at<double>(0), Rvec.at<double>(1), Rvec.at<double>(2), tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
        std::set<int>::iterator it;
        for (it=idxs.begin(); it!=idxs.end(); ++it)
        {
            ceres::CostFunction* costf = ReprojectionErrorMono::Create(p3Dclose[*it],p2Dclose[*it]);
            problem.AddResidualBlock(costf, nullptr /* squared loss */, camera);
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 25;
        // options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // double cost {0.0};
        // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
        // Logging("cost ", summary.final_cost,3);
        if ( mnError > summary.final_cost )
        {
            earlyTerm = 0;
            mnError = summary.final_cost;
            outCamera[0] = camera[0];
            outCamera[1] = camera[1];
            outCamera[2] = camera[2];
            outCamera[3] = camera[3];
            outCamera[4] = camera[4];
            outCamera[5] = camera[5];
        }
        else
            earlyTerm ++;
        if ( earlyTerm > 5 )
            break;
    }
    Rvec.at<double>(0) = outCamera[0];
    Rvec.at<double>(1) = outCamera[1];
    Rvec.at<double>(2) = outCamera[2];
    tvec.at<double>(0) = outCamera[3];
    tvec.at<double>(1) = outCamera[4];
    tvec.at<double>(2) = outCamera[5];
}

void FeatureTracker::ceresClose(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec)
{
    ceres::Problem problem;
    // make initial guess


    // double camera[6] {Rvec.at<double>(0), Rvec.at<double>(1), Rvec.at<double>(2), tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
    double cameraR[3] {Rvec.at<double>(0), Rvec.at<double>(1), Rvec.at<double>(2)};
    double cameraT[3] {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
    size_t end {p3Dclose.size()};
    // Logging("R", Rvec.at<double>(0),3);
    // Logging("cam", camera[0],3);
    problem.AddParameterBlock(cameraR,3);
    problem.AddParameterBlock(cameraT,3);
    for (size_t i{0}; i < end; i++)
    {
        ceres::CostFunction* costf = ReprojectionErrorMono::Create(p3Dclose[i],p2Dclose[i]);
        
        problem.AddResidualBlock(costf, new ceres::HuberLoss(2.44765) /* squared loss */, cameraR, cameraT);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    
    options.max_num_iterations = 100;
    options.max_solver_time_in_seconds = 0.05;

    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // options.gradient_tolerance = 1e-16;
    // options.function_tolerance = 1e-16;
    // options.parameter_tolerance = 1e-16;
    // double cost {0.0};
    // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
    // Logging("cost ", summary.final_cost,3);
    // Logging("R bef", Rvec,3);
    // Logging("T bef", tvec,3);
    Rvec.at<double>(0) = cameraR[0];
    Rvec.at<double>(1) = cameraR[1];
    Rvec.at<double>(2) = cameraR[2];
    tvec.at<double>(0) = cameraT[0];
    tvec.at<double>(1) = cameraT[1];
    tvec.at<double>(2) = cameraT[2];
    // Logging("R after", Rvec,3);
    // Logging("T after", tvec,3);
}

void FeatureTracker::ceresMO(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec)
{
    ceres::Problem problem;
    // make initial guess

    Eigen::Quaterniond q;
    q = Eigen::AngleAxisd(Rvec.at<double>(0), Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(Rvec.at<double>(1), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(Rvec.at<double>(3), Eigen::Vector3d::UnitZ());
    Logging("q", q.x(),3);
    double camera[7] {q.w(), q.x(), q.y(), q.z(), tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
    size_t end {p3Dclose.size()};
    // Logging("R", Rvec.at<double>(0),3);
    // Logging("cam", camera[0],3);
    // problem.AddParameterBlock(camera);
    for (size_t i{0}; i < end; i++)
    {
        ceres::CostFunction* costf = ReprojectionErrorMO::Create(p3Dclose[i],p2Dclose[i]);
        problem.AddResidualBlock(costf, new ceres::HuberLoss(1.0) /* squared loss */, camera);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    
    options.max_num_iterations = 100;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_solver_time_in_seconds = 0.1;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // options.gradient_tolerance = 1e-16;
    // options.function_tolerance = 1e-16;
    // options.parameter_tolerance = 1e-16;
    // double cost {0.0};
    // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
    // Logging("cost ", summary.final_cost,3);
    // Logging("R bef", Rvec,3);
    // Logging("T bef", tvec,3);
    Eigen::Quaterniond d(camera[0],camera[1],camera[2],camera[3]);
    auto euler = d.toRotationMatrix().eulerAngles(0, 1, 2);
    Logging("euler", euler[0],3);
    Logging("Rvec.at<double>(0)", Rvec.at<double>(0),3);
    Rvec.at<double>(0) = euler[0];
    Rvec.at<double>(1) = euler[1];
    Rvec.at<double>(2) = euler[2];
    tvec.at<double>(0) = camera[4];
    tvec.at<double>(1) = camera[5];
    tvec.at<double>(2) = camera[6];
    // Logging("R after", Rvec,3);
    // Logging("T after", tvec,3);
}

void FeatureTracker::getSamples(std::vector<int>& idxVec,std::set<int>& idxs)
{
    const size_t mxSize {idxVec.size()};
    while (idxs.size() < sampleSize)
    {
        std::random_device rd;
        std::mt19937 gen(rd());std::uniform_int_distribution<> distr(0, mxSize);
        idxs.insert(distr(gen));
    }

}

void FeatureTracker::get3DClose(std::vector<cv::Point3d>& p3D, std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose)
{
    const size_t end{prePnts.left.size()};
    p3Dclose.reserve(end);
    p2Dclose.reserve(end);
    for (size_t i{0}; i < end ; i++)
    {
        if ( prePnts.useable[i] )
        {
            p3Dclose.emplace_back(p3D[i]);
            p2Dclose.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
        }
    }
}

void FeatureTracker::getIdxVec(std::vector<int>& idxVec, const size_t size)
{
    idxVec.reserve(size);
    for (size_t i{0}; i < size ; i++)
    {
        idxVec.emplace_back(i);
    }
}

void FeatureTracker::compute2Dfrom3D(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D, std::vector<cv::Point2d>& pn2D)
{
    const size_t end {p3D.size()};

    p2D.reserve(end);
    pn2D.reserve(end);

    for (size_t i{0}; i < end ; i ++)
    {
        const double px {p3D[i].x};
        const double py {p3D[i].y};
        const double pz {p3D[i].z};

        const double invZ = 1.0f/pz;
        const double fx = zedPtr->cameraLeft.fx;
        const double fy = zedPtr->cameraLeft.fy;
        const double cx = zedPtr->cameraLeft.cx;
        const double cy = zedPtr->cameraLeft.cy;

        p2D.emplace_back(fx*px*invZ + cx, fy*py*invZ + cy);
        pn2D.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
    }

}

void FeatureTracker::essForMonoPose(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D)
{
    std::vector<uchar> inliers;
    cv::Mat E = cv::findEssentialMat(prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC, 0.99,1.0f, inliers);
    cv::Mat R1es,R2es,tes;
    cv::decomposeEssentialMat(E,R1es, R2es, tes);
    cv::Rodrigues(R1es,R1es);
    cv::Rodrigues(R2es,R2es);
    const double norm1 {cv::norm(Rvec,R1es)};
    const double norm2 {cv::norm(Rvec,R2es)};

    if (norm1 > norm2)
        Rvec = R2es;
    else
        Rvec = R1es;

    pnts.reduce<uchar>(inliers);
    prePnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::Point3d,uchar>(p3D, inliers);

#if POINTSIM
    drawPoints(lIm.rIm,prePnts.left, pnts.left, "essential");
#endif

}

void FeatureTracker::pnpRansac(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D)
{

    std::vector<int>idxs;
    // Logging("Rvecbef", Rvec,3);
    // Logging("Tvecbef", tvec,3);


    cv::solvePnPRansac(p3D, p2D ,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true,100, 8.0f, 0.99, idxs);
    // Logging("Rvecaft", Rvec,3);
    // Logging("Tvecaft", tvec,3);
    // prePnts.reduceWithInliers<int>(idxs);
    // pnts.reduceWithInliers<int>(idxs);
    // reduceStereoKeysIdx<int,cv::Point2f>(stereoKeys,idxs, pnts.left,pnts.left);
    reduceVectorInliersTemp<cv::Point3d,int>(p3D,idxs);
    reduceVectorInliersTemp<cv::Point3d,int>(prePnts.points3D,idxs);
    reduceVectorInliersTemp<bool,int>(prePnts.useable,idxs);
    reduceVectorInliersTemp<cv::Point2f,int>(prePnts.left,idxs);
    reduceVectorInliersTemp<float,int>(prePnts.depth,idxs);
    reduceVectorInliersTemp<cv::Point2f,int>(pnts.left,idxs);

    // std::vector<cv::Point3d> p3dclose;
    // std::vector<cv::Point2d> p2dclose;
    // p2dclose.reserve(pnts.left.size());
    // p3dclose.reserve(pnts.left.size());
    // for (size_t i {0}; i < pnts.left.size(); i++)
    // {
    //     if ( prePnts.useable[i] )
    //     {
    //         p3dclose.emplace_back(p3D[i]);
    //         p2dclose.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
    //     }
    // }

    // ceresClose(p3dclose,p2dclose,Rvec, tvec);

    // reduceVectorInliersTemp<cv::Point2d,int>(p2Ddepth,idxs);
    // reduceVectorInliersTemp<cv::Point3d,int>(p3Ddepth,idxs);
    // cv::solvePnP(p3D, pnts.left,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true);

    // uStereo = p3D.size();


#if PROJECTIM
    std::vector<cv::Point2d> p2Dtr, pn2D;

    compute2Dfrom3D(p3D, p2Dtr, pn2D);

    draw2D3D(pLIm.rIm, p2Dtr, pn2D);
#endif

}

void FeatureTracker::poseEstKal(cv::Mat& Rvec, cv::Mat& tvec, const size_t p3dsize)
{
    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    // Logging("tvec", cv::norm(tvec,pTvec), 3);
    // Logging("Rvec", cv::norm(Rvec,pRvec), 3);

    // // Logging("tvec", tvec, 3);
    // // Logging("pTvec", pTvec, 3);

    // // Logging("Rvec", Rvec, 3);
    // // Logging("pRvec", pRvec, 3);

    if ((cv::norm(tvec,pTvec) > 1.0f || cv::norm(Rvec,pRvec) > 0.5f) && curFrame != 1)
    {
        tvec = pTvec.clone();
        Rvec = pRvec.clone();
    }
    else
    {
        pTvec = tvec.clone();
        pRvec = Rvec.clone();
    }
    lkal.fillMeasurements(measurements, tvec, Rvec);


    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    // translation_estimated = tvec.clone();
    // cv::Rodrigues(Rvec, rotation_estimated);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();

    // publishPoseTrial();

}

void FeatureTracker::refine3DPnts()
{
    // Eigen::Matrix4d temp = poseEstFrame.inverse();
    // std::vector<cv::Point3d>::iterator it;
    // std::vector<cv::Point3d>::const_iterator end(prePnts.points3D.end());
    // for (it = prePnts.points3D.begin(); it != end; it ++)
    // {
    //     Eigen::Vector4d p4d(it->x, it->y, it->z,1);
    //     p4d = temp * p4d;
    //     *it = cv::Point3d(p4d(0), p4d(1), p4d(2));
    // }

}

void FeatureTracker::get3dPointsforPose(std::vector<cv::Point3d>& p3D)
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    p3D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            if (prePnts.useable[i])
                p3D.emplace_back(point);
            else
                p3D.emplace_back(cv::Point3d(prePnts.left[i].x, prePnts.left[i].y, 0.0f));
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
}

void FeatureTracker::get3dPointsforPoseAll(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D)
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    p3D.reserve(end);
    p2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            p3D.emplace_back(point);
            p2D.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
}

void FeatureTracker::pointsInFrame(std::vector<cv::Point3d>& p3D)
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    p3D.reserve(end);
    reprojIdxs.clear();
    reprojIdxs.reserve(end);
    uStereo = 0;
    int countIdx {0};
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            // if (p2dtemp.x > zedPtr->mWidth || p2dtemp.x < 0 || p2dtemp.y > zedPtr->mHeight || p2dtemp.y < 0)
            //     continue;
            reprojIdxs.emplace_back(countIdx++);
            inliers[i] = true;
            p3D.emplace_back(point);
            // prePnts.left[i] = cv::Point2f((float)p2dtemp.x, (float)p2dtemp.y);
            // if ( prePnts.useable[i] )
            //     uStereo ++;
        }
    }
    prePnts.reduce<bool>(inliers);
    // uMono = prePnts.points3D.size() - uStereo;
}

bool FeatureTracker::wPntToCamPose(const cv::Point3d& p3, cv::Point2d& p2, cv::Point3d& p3cam)
{
    Eigen::Vector4d point(p3.x, p3.y, p3.z, 1);
    // Logging("point",point,3);
    point = zedPtr->cameraPose.poseInverse * point;
    // point = poseEstFrameInv * point;

    const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;
    
    const double invZ = 1.0f/pointZ;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};

    const int min {0};
    const int maxW {zedPtr->mWidth};
    const int maxH {zedPtr->mHeight};

    if (u < min || u > maxW)
        return false;
    if (v < min || v > maxH)
        return false;

    p2 = cv::Point2d(u,v);
    
    p3cam = cv::Point3d(pointX, pointY, pointZ);

    return true;

}

void FeatureTracker::reprojError()
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    double err {0.0};
    int errC {0};
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            cv::Point2f ptr = cv::Point2f((float)p2dtemp.x, (float)p2dtemp.y);
            err += sqrt(pointsDist(ptr, prePnts.left[i]));
            errC ++;
        }
    }
    if (errC > 0)
    {
        const double avErr = err/errC;
        for (size_t i {0};i < end;i++)
        {
            // if ( )
        }
    }

    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
}

void FeatureTracker::calcGridVel()
{
    const int gRows {gridVelNumb};
    const int gCols {gridVelNumb};

    std::vector<float> gridx;
    std::vector<float> gridy;
    std::vector<int> counts;
    const int gridsq {gridVelNumb * gridVelNumb};
    gridx.resize(gridsq);
    gridy.resize(gridsq);
    counts.resize(gridsq);
    const int wid {(int)zedPtr->mWidth/gCols + 1};
    const int hig {(int)zedPtr->mHeight/gRows + 1};
    int ic {0};
    std::vector<cv::Point2f>::const_iterator it, end(prePnts.left.end());
    for (it = prePnts.left.begin(); it != end; it ++, ic++)
    {
        const int w {(int)it->x/wid};
        const int h {(int)it->y/hig};
        counts[(int)(w + h*gCols)] += 1;
        gridx[(int)(w + h*gCols)] += (it->x - pnts.left[ic].x);
        gridy[(int)(w + h*gCols)] += (it->y - pnts.left[ic].y);
    }

    for (size_t i {0}; i < gRows * gCols; i++)
    {
        if ( counts[i] != 0 )
        {
            gridTraX[i] = gridx[i]/counts[i];
            gridTraY[i] = gridy[i]/counts[i];
        }
        else
        {
            gridTraX[i] = gridTraX[i]/2;
            gridTraY[i] = gridTraY[i]/2;
        }
    }
}

void FeatureTracker::calculateNextPnts()
{
    const size_t end {prePnts.points3D.size()};
    pnts.left.reserve(end);
    std::vector<bool> in;
    in.resize(end,true);
    for (size_t i{0}; i < end; i++)
    {
        cv::Point2d pd((double)prePnts.left[i].x, (double)prePnts.left[i].y);
        if ( predictProjection3D(prePnts.points3D[i],pd) )
            pnts.left.emplace_back((float)pd.x, (float)pd.y);
        else
            in[i] = false;

    }
    prePnts.reduce<bool>(in);
}

void FeatureTracker::predictPts(std::vector<cv::Point2f>& curPnts)
{
    const size_t end {prePnts.points3D.size()};
    curPnts.reserve(end);
    for (size_t i{0}; i < end; i++)
    {
        cv::Point2d pd((double)prePnts.left[i].x, (double)prePnts.left[i].y);
        predictProjection3D(prePnts.points3D[i],pd);
        curPnts.emplace_back((float)pd.x, (float)pd.y);
    }
}

void FeatureTracker::calculateNextPntsGrids()
{
    const size_t end {prePnts.points3D.size()};
    const int gRows {gridVelNumb};
    const int gCols {gridVelNumb};
    const int wid {(int)zedPtr->mWidth/gCols + 1};
    const int hig {(int)zedPtr->mHeight/gRows + 1};
    pnts.left.reserve(end);
    for (size_t i{0}; i < end; i++)
    {
        cv::Point2f pf = prePnts.left[i];
        const int w  {(int)(prePnts.left[i].x/wid)};
        const int h  {(int)(prePnts.left[i].y/hig)};
        pf.x = pf.x - gridTraX[w + gCols*h];
        pf.y = pf.y - gridTraY[w + gCols*h];
        pnts.left.emplace_back(pf);
    }
}

void FeatureTracker::opticalFlow()
{
    std::vector<float> err, err1;
    std::vector <uchar>  inliers;
    cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, prePnts.left, pnts.left, inliers, err,cv::Size(21,21),3, criteria);

    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);
    // reduceStereoKeys<uchar>(stereoKeys, inliers, inliers);
    // reduceVectorTemp<float,uchar>(err,inliers);

    // const float minErrValue {20.0f};

    // prePnts.reduceWithValue<float>(err, minErrValue);
    // pnts.reduceWithValue<float>(err, minErrValue);

    // cv::cornerSubPix(lIm.im,pnts.left,cv::Size(5,5),cv::Size(-1,-1),criteria);

    // cv::findFundamentalMat(prePnts.left, pnts.left, inliers, cv::FM_RANSAC, 3, 0.99);


    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);

    // const size_t end{pnts.left.size()};
    // std::vector<bool> check;
    // check.resize(end);
    // for (size_t i{0};i < end;i++)
    // {
    //     if (!(pnts.left[i].x > zedPtr->mWidth || pnts.left[i].x < 0 || pnts.left[i].y > zedPtr->mHeight || pnts.left[i].y < 0))
    //         check[i] = true;
    // }

    // prePnts.reduce<bool>(check);
    // pnts.reduce<bool>(check);

#if OPTICALIM
    drawOptical("Optical", lIm.rIm,prePnts.left, pnts.left);
#endif
}

void FeatureTracker::changeUndef(std::vector<float>& err, std::vector <uchar>& inliers, std::vector<cv::Point2f>& temp)
{
    const float minErrValue {20.0f};
    const size_t end{pnts.left.size()};
    for (size_t i{0}; i < end; i++)
        if ( !inliers[i] || err[i] > minErrValue )
            pnts.left[i] = temp[i];
}

void FeatureTracker::opticalFlowPredict()
{
    Timer optical("optical");
    std::vector<float> err, err1;
    std::vector <uchar>  inliers, inliers2;
    std::vector<cv::Point3d> p3D;

    // reprojError();

    calculateNextPnts();
#if OPTICALIM
    drawOptical("before", pLIm.rIm,prePnts.left, pnts.left);
#endif
    std::vector<cv::Point2f> predPnts = pnts.left;
    cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, prePnts.left, pnts.left, inliers, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);
    // inliers.clear();
    std::vector<cv::Point2f> temp = prePnts.left;
    cv::calcOpticalFlowPyrLK(lIm.im, pLIm.im, pnts.left, temp, inliers2, err1,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);
    // reduceVectorTemp<cv::Point2f,uchar>(temp,inliers);

    // inliers.clear();
    for (size_t i {0}; i < pnts.left.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],prePnts.left[i]) <= 0.25)
            inliers[i] = true;
        else
            inliers[i] = false;
    }
    // changeOptRes(inliers, predPnts, pnts.left);

    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);
    // reduceStereoKeys<uchar>(stereoKeys, inliers, inliers);
    // cv::cornerSubPix(pLIm.im, prePnts.left, cv::Size(5,5),cv::Size(-1,-1),criteria);
    // cv::cornerSubPix(lIm.im, pnts.left, cv::Size(5,5),cv::Size(-1,-1),criteria);
    // reduceVectorTemp<cv::Point2f,bool>(predPnts, check);


// #if OPTICALIM
//     drawOptical("before", pLIm.rIm,prePnts.left, pnts.left);
// #endif
    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);

    // matcherTrial();

    // cv::findFundamentalMat(prePnts.left, pnts.left, inliers, cv::FM_RANSAC, 3, 0.99);


    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);
    // cv::imshow("prev left", pLIm.im);
    // cv::imshow("after left", lIm.im);
#if OPTICALIM
    drawOptical("after", pLIm.rIm,prePnts.left, pnts.left);
#endif

}

void FeatureTracker::optFlow(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2f>& pPnts, std::vector<cv::Point2f>& curPnts)
{
    std::vector<float> err, err1;
    std::vector <uchar>  inliers, inliers2;
    pPnts = prePnts.left;
    std::vector<cv::Point2f> temp = pPnts;
    // reprojError();
    if (curFrame == 1)
    {
        cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, pPnts, curPnts, inliers, err,cv::Size(21,21),3, criteria);

        cv::calcOpticalFlowPyrLK(lIm.im, pLIm.im, curPnts, temp, inliers2, err1,cv::Size(21,21),3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    }
    else
    {
        predictPts(curPnts);
        cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, pPnts, curPnts, inliers, err,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        cv::calcOpticalFlowPyrLK(lIm.im, pLIm.im, curPnts, temp, inliers2, err1,cv::Size(21,21),1, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    for (size_t i {0}; i < curPnts.size(); i ++)
    {
        if ( inliers[i] && inliers2[i] && pointsDist(temp[i],pPnts[i]) <= 0.25)
            inliers[i] = true;
        else
            inliers[i] = false;
    }

    reduceVectorTemp<cv::Point2f,uchar>(pPnts,inliers);
    reduceVectorTemp<cv::Point2f,uchar>(curPnts,inliers);
    reduceVectorTemp<cv::Point3d,uchar>(p3D,inliers);
    reduceVectorTemp<int,uchar>(reprojIdxs,inliers);

#if OPTICALIM
    drawOptical("after", pLIm.rIm,pPnts, curPnts);
#endif

}

void FeatureTracker::changeOptRes(std::vector <uchar>&  inliers, std::vector<cv::Point2f>& pnts1, std::vector<cv::Point2f>& pnts2)
{
    const size_t end {pnts1.size()};
    const int off{0};
    // const float mxPointsDist {400.0f};
    const int w {zedPtr->mWidth + off};
    const int h {zedPtr->mHeight + off};
    for ( size_t i {0}; i < end; i ++)
    {
        // if ( pnts1[i].x > w || pnts1[i].x < -off || pnts1[i].y > h || pnts1[i].y < -off )
        //     inliers[i] = false;
        // if (pnts1[i].x > w || pnts1[i].x < -off || pnts1[i].y > h || pnts1[i].y < -off )
        //     pnts2[i] = pnts1[i];
        if ( !inliers[i] && (pnts1[i].x > w || pnts1[i].x < -off || pnts1[i].y > h || pnts1[i].y < -off) )
        {
            pnts2[i] = pnts1[i];
            inliers[i] = true;
        }
            
    }
}

float FeatureTracker::pointsDist(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return pow(p1.x - p2.x,2) + pow(p1.y - p2.y,2);
}

void FeatureTracker::matcherTrial()
{
    StereoKeypoints ke;
    StereoDescriptors de;
    fe.extractORBGrids(pLIm.im, lIm.im, de, ke);
    std::vector<cv::DMatch> matr;
    std::vector<cv::DMatch> newmatr;
    cv::Ptr<cv::BFMatcher> matcherer = cv::BFMatcher::create(cv::NORM_HAMMING,true);
    std::vector<uchar> in;
    matcherer->match(de.left,de.right,matr);

    for (int i = 0; i < matr.size(); i++)
    {
        tr.left.push_back(ke.left[matr[i].queryIdx].pt);
        tr.right.push_back(ke.right[matr[i].trainIdx].pt);
        newmatr.push_back(cv::DMatch(i,i, matr[i].distance));
    }

    cv::findFundamentalMat(tr.left,tr.right,in, cv::FM_RANSAC,3,0.99);
    reduceVectorTemp<cv::Point2f,uchar>(tr.left,in);
    reduceVectorTemp<cv::Point2f,uchar>(tr.right,in);
    reduceVectorTemp<cv::DMatch,uchar>(newmatr,in);
    
    drawMatchesKeys(pLIm.rIm, tr.left, tr.right, newmatr);

    tr.left.clear();
    tr.right.clear();
    matr.clear();
    newmatr.clear();
    ke.left.clear();
    ke.right.clear();


}

void FeatureTracker::drawMatchesKeys(const cv::Mat& lIm, const std::vector<cv::Point2f>& keys1, const std::vector<cv::Point2f>& keys2, const std::vector<cv::DMatch> matches)
{
    cv::Mat outIm = lIm.clone();
    for (auto m:matches)
    {
        cv::circle(outIm, keys1[m.queryIdx],2,cv::Scalar(0,255,0));
        cv::line(outIm, keys1[m.queryIdx], keys2[m.queryIdx],cv::Scalar(0,0,255));
        cv::circle(outIm, keys2[m.queryIdx],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Matches matcherrr", outIm);
    cv::waitKey(waitImMat);
}

void FeatureTracker::opticalFlowGood()
{
    std::vector<float> err;
    std::vector<cv::Point2f> pbef,pnex;
    std::vector <uchar>  inliers;
    cv::goodFeaturesToTrack(pLIm.im, pbef,300, 0.01, 30);
    cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, pbef, pnex, inliers, err,cv::Size(21,21),1, criteria);


#if OPTICALIM
    drawOptical("before fund good", pLIm.rIm,pbef, pnex);
#endif

    cv::findFundamentalMat(pbef, pnex, inliers, cv::FM_RANSAC, 3, 0.99);


    reduceVectorTemp<cv::Point2f, uchar>(pbef ,inliers);
    reduceVectorTemp<cv::Point2f, uchar>(pnex ,inliers);

#if OPTICALIM
    drawOptical("after fund good", pLIm.rIm,pbef, pnex);
#endif


}

void FeatureTracker::updateKeysGoodFeatures(const int frame)
{
    std::vector<cv::DMatch> matches;
    // stereoFeaturesPop(pLIm.im, pRIm.im, matches,pnts, prePnts);
    stereoFeaturesGoodFeatures(pLIm.im, pRIm.im,pnts, prePnts);
    prePnts.addLeft(pnts);
    pnts.clear();
}

void FeatureTracker::updateKeys(const int frame)
{
    std::vector<cv::DMatch> matches;
    // stereoFeaturesPop(pLIm.im, pRIm.im, matches,pnts, prePnts);
    stereoFeaturesMask(pLIm.im, pRIm.im, matches,pnts, prePnts);
    prePnts.addLeft(pnts);
    pnts.clear();
}

void FeatureTracker::updateKeysClose(const int frame)
{
    std::vector<cv::DMatch> matches;
    // stereoFeaturesPop(pLIm.im, pRIm.im, matches,pnts, prePnts);
    stereoFeaturesClose(pLIm.im, pRIm.im, matches,pnts);
    prePnts.addLeft(pnts);
    pnts.clear();
}

float FeatureTracker::calcDt()
{
    endTime =  std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    startTime = std::chrono::high_resolution_clock::now();
    return duration.count();
}

void FeatureTracker::setLRImages(const int frameNumber)
{
    lIm.setImage(frameNumber,"left", zedPtr->seq);
    rIm.setImage(frameNumber,"right", zedPtr->seq);
    if (!zedPtr->rectified)
        rectifyLRImages();
}

void FeatureTracker::setLImage(const int frameNumber)
{
    lIm.setImage(frameNumber,"left", zedPtr->seq);
    if (!zedPtr->rectified)
        rectifyLImage();
    
}

void FeatureTracker::setPreLImage()
{
    pLIm.im = lIm.im.clone();
    pLIm.rIm = lIm.rIm.clone();
}

void FeatureTracker::setPreRImage()
{
    pRIm.im = rIm.im.clone();
    pRIm.rIm = rIm.rIm.clone();
}

void FeatureTracker::setPre()
{
    setPreLImage();
    setPreRImage();
    // calcGridVel();
    prePnts.left = pnts.left;
    clearPre();
}

void FeatureTracker::setPreTrial()
{
    setPreLImage();
    setPreRImage();
    // calcGridVel();
    checkBoundsLeft();
    clearPre();
}

void FeatureTracker::checkBoundsLeft()
{
    const int w {zedPtr->mWidth};
    const int h {zedPtr->mHeight};
    // prePnts.left = pnts.left;
    std::vector<bool> check;
    check.resize(pnts.left.size());
    int count {0};
    uStereo = 0;
    std::vector<cv::Point2f>::const_iterator it, end(pnts.left.end());
    for (it = pnts.left.begin(); it != end; it ++, count ++)
    {
        prePnts.left[count] = *it;
        if (!(it->x > w || it->x < 0 || it->y > h || it->y < 0))
        {
            check[count] = true;
            if ( prePnts.useable[count] )
                uStereo ++;
        }
    }
    prePnts.reduce<bool>(check);
    uMono = prePnts.left.size() - uStereo;
}

void FeatureTracker::setPreInit()
{
    setPreLImage();
    setPreRImage();
    prePnts.clone(pnts);
    clearPre();
}

void FeatureTracker::clearPre()
{
    pnts.clear();
}

cv::Mat FeatureTracker::getLImage()
{
    return lIm.im;
}

cv::Mat FeatureTracker::getRImage()
{
    return rIm.im;
}

cv::Mat FeatureTracker::getPLImage()
{
    return pLIm.im;
}

cv::Mat FeatureTracker::getPRImage()
{
    return pRIm.im;
}

void FeatureTracker::rectifyLRImages()
{
    lIm.rectifyImage(lIm.im, rmap[0][0], rmap[0][1]);
    lIm.rectifyImage(lIm.rIm, rmap[0][0], rmap[0][1]);
    rIm.rectifyImage(rIm.im, rmap[1][0], rmap[1][1]);
    rIm.rectifyImage(rIm.rIm, rmap[1][0], rmap[1][1]);
}

void FeatureTracker::rectifyLImage()
{
    lIm.rectifyImage(lIm.im, rmap[0][0], rmap[0][1]);
    lIm.rectifyImage(lIm.rIm, rmap[0][0], rmap[0][1]);
}

void FeatureTracker::drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys)
{
    cv::Mat outIm = im.clone();
    for (auto& key:keys)
    {
        cv::circle(outIm, key.pt,2,cv::Scalar(0,255,0));

    }
    cv::imshow(com, outIm);
    cv::waitKey(waitImKey);
}

void FeatureTracker::drawMatches(const cv::Mat& lIm, const SubPixelPoints& pnts, const std::vector<cv::DMatch> matches)
{
    cv::Mat outIm = lIm.clone();
    for (auto m:matches)
    {
        cv::circle(outIm, pnts.left[m.queryIdx],2,cv::Scalar(0,255,0));
        cv::line(outIm, pnts.left[m.queryIdx], pnts.right[m.trainIdx],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts.right[m.trainIdx],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Matches", outIm);
    cv::waitKey(waitImMat);
}

void FeatureTracker::drawMatchesGoodFeatures(const cv::Mat& lIm, const SubPixelPoints& pnts)
{
    cv::Mat outIm = lIm.clone();
    const size_t size {pnts.left.size()};
    for (size_t i{0}; i < size; i ++)
    {
        cv::circle(outIm, pnts.left[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, pnts.left[i], pnts.right[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts.right[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Matches", outIm);
    cv::waitKey(waitImMat);
}

void FeatureTracker::drawOptical(const char* com,const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts)
{
    cv::Mat outIm = im.clone();
    const size_t end {prePnts.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, prePnts[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, prePnts[i], pnts[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow(com, outIm);
     cv::waitKey(waitImOpt);
}

void FeatureTracker::drawPoints(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts, const char* str)
{
    cv::Mat outIm = im.clone();
    const size_t end {prePnts.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, prePnts[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, prePnts[i], pnts[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow(str, outIm);
    cv::waitKey(waitImOpt);
}

void FeatureTracker::draw2D3D(const cv::Mat& im, const std::vector<cv::Point2d>& p2Dfp3D, const std::vector<cv::Point2d>& p2D)
{
    cv::Mat outIm = im.clone();
    const size_t end {p2Dfp3D.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, p2Dfp3D[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, p2Dfp3D[i], p2D[i],cv::Scalar(0,0,255));
        cv::circle(outIm, p2D[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Project", outIm);
    cv::waitKey(waitImPro);

}

bool FeatureTracker::checkProjection3D(cv::Point3d& point3D, cv::Point2d& point2d)
{
    
    // Logging("key",keyFrameNumb,3);
    Eigen::Vector4d point(point3D.x, point3D.y, point3D.z,1);
    // Logging("point",point,3);
    point = zedPtr->cameraPose.poseInverse * point;
    // Logging("point",point,3);
    // Logging("zedPtr",zedPtr->cameraPose.poseInverse,3);
    // Logging("getPose",keyframes[keyFrameNumb].getPose(),3);
    point3D.x = point(0);
    point3D.y = point(1);
    point3D.z = point(2);
    const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;

    const double invZ = 1.0f/pointZ;
    const double fx = zedPtr->cameraLeft.fx;
    const double fy = zedPtr->cameraLeft.fy;
    const double cx = zedPtr->cameraLeft.cx;
    const double cy = zedPtr->cameraLeft.cy;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};


    const int min {0};
    const int maxW {zedPtr->mWidth};
    const int maxH {zedPtr->mHeight};

    if (u < min || u > maxW)
        return false;
    if (v < min || v > maxH)
        return false;

    // const double k1 = zedptr->cameraLeft.distCoeffs.at<double>(0,0);
    // const double k2 = zedptr->cameraLeft.distCoeffs.at<double>(0,1);
    // const double p1 = zedptr->cameraLeft.distCoeffs.at<double>(0,2);
    // const double p2 = zedptr->cameraLeft.distCoeffs.at<double>(0,3);
    // const double k3 = zedptr->cameraLeft.distCoeffs.at<double>(0,4);

    const double k1 {0};
    const double k2 {0};
    const double p1 {0};
    const double p2 {0};
    const double k3 {0};

    double u_distort, v_distort;

    double x = (u - cx) * invfx;
    double y = (v - cy) * invfy;
    
    double r2 = x * x + y * y;

    // Radial distorsion
    double x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    double y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    // u = u_distort;
    // v = v_distort;

    point2d = cv::Point2d(u,v);
    

    if (u > maxW || u < 0 || v > maxH || v < 0)
        return false;

    return true;

}

bool FeatureTracker::predictProjection3D(const cv::Point3d& point3D, cv::Point2d& point2d)
{
    
    // Logging("key",keyFrameNumb,3);
    Eigen::Vector4d point(point3D.x, point3D.y, point3D.z,1);
    // Logging("point",point,3);
    // point = zedPtr->cameraPose.poseInverse * point;

    point = predNPose.inverse() * point;
    // Logging("point",point,3);
    // Logging("zedPtr",zedPtr->cameraPose.poseInverse,3);
    // Logging("getPose",keyframes[keyFrameNumb].getPose(),3);
     const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;

    const double invZ = 1.0f/pointZ;
    const double fx = zedPtr->cameraLeft.fx;
    const double fy = zedPtr->cameraLeft.fy;
    const double cx = zedPtr->cameraLeft.cx;
    const double cy = zedPtr->cameraLeft.cy;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};


    const int min {0};
    const int maxW {zedPtr->mWidth};
    const int maxH {zedPtr->mHeight};

    // if ( u < min )
    //     u = min;
    // else if ( u > maxW )
    //     u = maxW ;

    // if ( v < min )
    //     v = min;
    // else if ( v > maxH )
    //     v = maxH ;


    point2d = cv::Point2d(u,v);
    return true;

}

void FeatureTracker::convertToEigen(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& tr)
{
    Eigen::Matrix3d Reig;
    Eigen::Vector3d teig;
    cv::cv2eigen(Rvec.t(),Reig);
    cv::cv2eigen(-tvec,teig);

    tr.setIdentity();
    tr.block<3,3>(0,0) = Reig;
    tr.block<3,1>(0,3) = teig;
}

void FeatureTracker::publishPoseNew()
{
    prevWPose = zedPtr->cameraPose.pose;
    prevWPoseInv = zedPtr->cameraPose.poseInverse;
    zedPtr->cameraPose.setPose(poseEst);
    zedPtr->cameraPose.setInvPose(poseEst.inverse());
    predNPose = poseEst * (prevWPoseInv * poseEst);
    predNPoseInv = predNPose.inverse();
#if SAVEODOMETRYDATA
    saveData();
#endif
    // Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
    // Logging("predNPose", predNPose,3);
}

void FeatureTracker::publishPoseCeres()
{
    poseEst = poseEst * poseEstFrame;
    poseEstFrameInv = poseEstFrame.inverse();
    prevWPose = zedPtr->cameraPose.pose;
    prevWPoseInv = zedPtr->cameraPose.poseInverse;
    zedPtr->cameraPose.setPose(poseEst);
    zedPtr->cameraPose.setInvPose(poseEst.inverse());
    predNPose = poseEst * (prevWPoseInv * poseEst);
    predNPoseInv = predNPose.inverse();
#if SAVEODOMETRYDATA
    saveData();
#endif
    // Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
    // Logging("predNPose", predNPose,3);
}

void FeatureTracker::publishPose()
{
    poseEst = poseEst * poseEstFrame;
    poseEstFrameInv = poseEstFrame.inverse();
    prevWPose = zedPtr->cameraPose.pose;
    prevWPoseInv = zedPtr->cameraPose.poseInverse;
    zedPtr->cameraPose.setPose(poseEst);
    zedPtr->cameraPose.setInvPose(poseEst.inverse());
    predNPose = poseEst * (prevWPoseInv * poseEst);
    predNPoseInv = predNPose.inverse();
#if SAVEODOMETRYDATA
    saveData();
#endif
    // Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
    // Logging("predNPose", predNPose,3);
}

void FeatureTracker::publishPoseTrial()
{
    const float velThresh {0.07f};
    static double pvx {}, pvy {}, pvz {};
    double px {},py {},pz {};
    px = poseEst(0,3);
    py = poseEst(1,3);
    pz = poseEst(2,3);
    Eigen::Matrix4d prePoseEst = poseEst;
    poseEst = poseEst * poseEstFrame;
    double vx {},vy {},vz {};
    vx = (poseEst(0,3) - px)/dt;
    vy = (poseEst(1,3) - py)/dt;
    vz = (poseEst(2,3) - pz)/dt;
    poseEstFrameInv = poseEstFrame.inverse();
    prevWPose = zedPtr->cameraPose.pose;
    prevWPoseInv = zedPtr->cameraPose.poseInverse;
    if ( curFrame != 1 )
    {

        if ( abs(vx - pvx) > velThresh || abs(vy - pvy) > velThresh || abs(vz - pvz) > velThresh )
        {
            poseEst = prePoseEst * prevposeEstFrame;
        }
        else
        {
            setVel(pvx, pvy, pvz, vx, vy, vz);
            prevposeEstFrame = poseEstFrame;
        }

    }
    else
    {
        setVel(pvx, pvy, pvz, vx, vy, vz);
        prevposeEstFrame = poseEstFrame;
    }
    predNPose = poseEst * (prevWPoseInv * poseEst);
    zedPtr->cameraPose.setPose(poseEst);
    zedPtr->cameraPose.setInvPose(poseEst.inverse());
#if SAVEODOMETRYDATA
    saveData();
#endif
    Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
}

inline void FeatureTracker::setVel(double& pvx, double& pvy, double& pvz, double vx, double vy, double vz)
{
    pvx = vx;
    pvy = vy;
    pvz = vz;
}

void FeatureTracker::saveData()
{
    Eigen::Matrix4d mat = zedPtr->cameraPose.pose.transpose();
    for (int32_t i{0}; i < 12; i ++)
    {
        if ( i == 0 )
            datafile << mat(i);
        else
            datafile << " " << mat(i);
    }
    datafile << '\n';
}

} // namespace vio_slam