#include "FeatureTracker.h"

namespace vio_slam
{

void ImageData::setImage(const int frameNumber, const char* whichImage)
{
    std::string imagePath;
    std::string first;
    std::string second, format;
#if KITTI_DATASET
    first = "/home/christos/Downloads/data_odometry_gray/dataset/sequences/00/";
    second = "/00";
    format = ".png";
#else
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/";
    second = "/frame";
    format = ".jpg";
#endif

    if (frameNumber > 999)
    {
        imagePath = first + whichImage + second + std::to_string(frameNumber/1000) + std::to_string((frameNumber%1000 - frameNumber%100)/100) + std::to_string((frameNumber%100 - frameNumber%10)/10) + std::to_string(frameNumber%10) + format;
    }
    if (frameNumber > 99)
    {
        imagePath = first + whichImage + second + "0" + std::to_string(frameNumber/100) + std::to_string((frameNumber%100 - frameNumber%10)/10) + std::to_string(frameNumber%10) + format;
    }
    else if (frameNumber > 9)
    {
        imagePath = first + whichImage + second + "00" + std::to_string(frameNumber/10) + std::to_string(frameNumber%10) + format;
    }
    else
    {
        imagePath = first + whichImage + second + "000" + std::to_string(frameNumber) + format;
    }
    im = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    rIm = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void ImageData::rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2)
{
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
}

FeatureData::FeatureData(Zed_Camera* zedPtr) : fx(zedPtr->cameraLeft.fx), fy(zedPtr->cameraLeft.fy), cx(zedPtr->cameraLeft.cx), cy(zedPtr->cameraLeft.cy)
{

}

void FeatureData::compute3DPoints(SubPixelPoints& prePnts, SubPixelPoints& pnts)
{
    const size_t end{pnts.left.size()};

    prePnts3DStereo.reserve(end);
    pnts2DStereo.reserve(end);

    prePnts2DMono.reserve(end);
    pnts2DMono.reserve(end);

    for (size_t i = 0; i < end; i++)
    {   

        prePnts2DMono.emplace_back(cv::Point2d((double)prePnts.left[i].x,(double)prePnts.left[i].y));
        pnts2DMono.emplace_back(cv::Point2d((double)pnts.left[i].x,(double)pnts.left[i].y));

        if (!pnts.useable[i])
            continue;

        const double zp = (double)prePnts.depth[i];
        const double xp = (double)(((double)prePnts.left[i].x-cx)*zp/fx);
        const double yp = (double)(((double)prePnts.left[i].y-cy)*zp/fy);

        prePnts3DStereo.emplace_back(cv::Point3d(xp,yp,zp));
        pnts2DStereo.emplace_back(cv::Point2d((double)pnts.left[i].x,(double)pnts.left[i].y));
        
    }
}

FeatureTracker::FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr) : zedPtr(_zedPtr), fm(zedPtr, zedPtr->mHeight, fe.getGridRows(), fe.getGridCols()), pE(zedPtr), fd(zedPtr)
{
    rmap[0][0] = _rmap[0][0];
    rmap[0][1] = _rmap[0][1];
    rmap[1][0] = _rmap[1][0];
    rmap[1][1] = _rmap[1][1];
}

void FeatureTracker::stereoFeatures(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    fe.extractFeatures(leftIm, rightIm, desc, keys);
    fm.computeStereoMatches(leftIm, rightIm, desc, matches, pnts, keys);
    std::vector<uchar> inliers;
    cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);

    pnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
    Logging("matches size", matches.size(),1);
#if MATCHESIM
    drawMatches(lIm.rIm, pnts, matches);
#endif
}

void FeatureTracker::initializeTracking()
{
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
    setPre();
}

void FeatureTracker::beginTracking(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        float dt {calcDt()};
        setLRImages(frame);
        if (uStereo < mnSize)
            updateKeys(frame);
        opticalFlow();



        getEssentialPose();

        setPre();
        
    }
}

void FeatureTracker::getEssentialPose()
{
    cv::Mat R,t;
    std::vector <uchar> inliers;
    cv::Mat E = cv::findEssentialMat(prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC,0.99,0.5, inliers);
    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);
    cv::recoverPose(E, prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix, R, t);
    pE.convertToEigenMat(R,t,poseEstFrame);

    publishPose();
    
}

void FeatureTracker::opticalFlow()
{
    std::vector<float> err, err1;
    std::vector <uchar>  inliers, inliers1;
    cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, prePnts.left, pnts.left, inliers, err,cv::Size(21,21),3, criteria);

    // Reverse Flow
    cv::calcOpticalFlowPyrLK(lIm.im, pLIm.im, pnts.left, prePnts.left, inliers1, err1,cv::Size(21,21),3, criteria);

    for (int i{0};i < inliers.size();i ++)
    {
        if (!(inliers1[i] && inliers[i]))
            inliers[i] = 0U;
        else
            err[i] += err1[i];
    }

    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);
    reduceVectorTemp<float,uchar>(err,inliers);

    const float minErrValue {20.0f};

    prePnts.reduceWithValue<float>(err, minErrValue);
    pnts.reduceWithValue<float>(err, minErrValue);

    inliers.clear();
    inliers1.clear();
    cv::findFundamentalMat(pnts.left, prePnts.left, inliers, cv::FM_RANSAC, 3, 0.99);
    // cv::findFundamentalMat(prePnts.left, pnts.left, inliers1, cv::FM_RANSAC, 3, 0.99);

    // for (int i{0};i < inliers.size();i ++)
    // {
    //     if (!(inliers1[i] && inliers[i]))
    //         inliers[i] = 0U;

    // }


    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);

#if OPTICALIM
    drawOptical(lIm.rIm,prePnts.left, pnts.left);
#endif

    uStereo = prePnts.left.size();
}

void FeatureTracker::updateKeys(const int frame)
{
    std::vector<cv::DMatch> matches;
    stereoFeatures(pLIm.im, pRIm.im, matches,pnts);
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
    lIm.setImage(frameNumber,"left");
    rIm.setImage(frameNumber,"right");
    if (!zedPtr->rectified)
        rectifyLRImages();
}

void FeatureTracker::setLImage(const int frameNumber)
{
    lIm.setImage(frameNumber,"left");
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
    cv::waitKey(waitIm);
}

void FeatureTracker::drawOptical(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts)
{
    cv::Mat outIm = im.clone();
    const size_t end {prePnts.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, prePnts[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, prePnts[i], pnts[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Optical", outIm);
    cv::waitKey(waitIm);
}

void FeatureTracker::publishPose()
{
poseEst = poseEst * poseEstFrame;
zedPtr->cameraPose.setPose(poseEst);
zedPtr->cameraPose.setInvPose(poseEst.inverse());

Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
}

} // namespace vio_slam