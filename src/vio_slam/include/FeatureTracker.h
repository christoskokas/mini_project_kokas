#pragma once

#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Camera.h"
#include "KeyFrame.h"
#include "PoseEstimator.h"
#include "FeatureManager.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Settings.h"
#include "Optimizer.h"

namespace vio_slam
{

class ImageData
{
    private:

    public:
        cv::Mat im, rIm;

        void setImage(int frameNumber, const char* whichImage);
        void rectifyImage(cv::Mat& image, cv::Mat& map1, cv::Mat& map2);
};

struct FeatureData
{
    const double fx,fy,cx,cy;

    std::vector<cv::Point3d> prePnts3DStereo;
    std::vector<cv::Point2d> pnts2DStereo, prePnts2DMono, pnts2DMono;
    Eigen::Matrix4Xd hPnts3D;

    FeatureData(Zed_Camera* zedPtr);

    void compute3DPoints(SubPixelPoints& prePnts, SubPixelPoints& pnts);

};

class FeatureTracker
{
    private :
        std::chrono::_V2::system_clock::time_point startTime, endTime;
        std::chrono::duration<float> duration;

        ImageData pLIm, pRIm, lIm, rIm;
        cv::Mat rmap[2][2];
        Zed_Camera* zedPtr;
        FeatureExtractor fe;
        FeatureMatcher fm;
        SubPixelPoints pnts,prepnts;
        PoseEstimator pE;
        FeatureData fd;


    public :

        FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr);

        void initializeTracking();

        void setLRImages(int frameNumber);
        void setLImage(int frameNumber);
        void setPreLImage();
        void setPreRImage();
        void setPre();

        cv::Mat getLImage();
        cv::Mat getRImage();
        cv::Mat getPLImage();
        cv::Mat getPRImage();

        void rectifyLRImages();
        void rectifyLImage();

};



} // namespace vio_slam


#endif // FEATURETRACKER_H