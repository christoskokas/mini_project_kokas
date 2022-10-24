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

#define MATCHESIM false
#define OPTICALIM true

namespace vio_slam
{

class ImageData
{
    private:

    public:
        cv::Mat im, rIm;

        void setImage(const int frameNumber, const char* whichImage);
        void rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2);
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

        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEstFrame = Eigen::Matrix4d::Identity();

        const int waitIm {1};

        const int mnSize {200};
        int uStereo {0};
        int uMono {0};

        ImageData pLIm, pRIm, lIm, rIm;
        cv::Mat rmap[2][2];
        Zed_Camera* zedPtr;
        FeatureExtractor fe;
        FeatureMatcher fm;
        SubPixelPoints pnts,prePnts;
        PoseEstimator pE;
        FeatureData fd;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 60, (0.0001000000000000000021))};


    public :

        FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr);

        void initializeTracking();
        void beginTracking(const int frames);

        void updateKeys(const int frame);

        void stereoFeatures(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts);
        void opticalFlow();
        void getEssentialPose();

        void setLRImages(const int frameNumber);
        void setLImage(const int frameNumber);
        void setPreLImage();
        void setPreRImage();
        void setPre();
        void clearPre();

        float calcDt();

        cv::Mat getLImage();
        cv::Mat getRImage();
        cv::Mat getPLImage();
        cv::Mat getPRImage();

        void rectifyLRImages();
        void rectifyLImage();

        void drawMatches(const cv::Mat& lIm, const SubPixelPoints& pnts, const std::vector<cv::DMatch> matches);
        void drawOptical(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts);

        void publishPose();

};



} // namespace vio_slam


#endif // FEATURETRACKER_H