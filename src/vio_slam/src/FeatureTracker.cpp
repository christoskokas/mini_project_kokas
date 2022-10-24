#include "FeatureTracker.h"

namespace vio_slam
{

void ImageData::setImage(int frameNumber, const char* whichImage)
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

void ImageData::rectifyImage(cv::Mat& image, cv::Mat& map1, cv::Mat& map2)
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

void FeatureTracker::initializeTracking()
{
    startTime = std::chrono::high_resolution_clock::now();
    setLRImages(0);
    StereoDescriptors desc;
    StereoKeypoints keys;
    fe.extractFeatures(lIm.im, rIm.im, desc, keys);
    std::vector<cv::DMatch> matches;
    fm.computeStereoMatches(lIm.im, rIm.im, desc, matches, pnts, keys);
    pE.setPrevR(zedPtr->sensorsRotate);
    cv::Mat tr = (cv::Mat_<double>(3,1) << 0.0,0.0,0.0);
    pE.setPrevT(tr);
    setPre();
}

void FeatureTracker::setLRImages(int frameNumber)
{
    lIm.setImage(frameNumber,"left");
    rIm.setImage(frameNumber,"right");
    if (!zedPtr->rectified)
        rectifyLRImages();
}

void FeatureTracker::setLImage(int frameNumber)
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
    prepnts.clone(pnts);
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

} // namespace vio_slam