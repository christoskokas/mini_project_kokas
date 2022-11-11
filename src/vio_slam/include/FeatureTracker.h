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
#include <fstream>
#include <string>
#include <iostream>
#include <random>

#define KEYSIM true
#define MATCHESIM true
#define OPTICALIM true
#define PROJECTIM true
#define POINTSIM true

namespace vio_slam
{

class ImageData
{
    private:

    public:
        cv::Mat im, rIm;

        void setImage(const int frameNumber, const char* whichImage, const std::string& seq);
        void rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2);
};

struct FeatureData
{
    const double fx,fy,cx,cy;

    Zed_Camera* zedPtr;

    std::vector<cv::Point3d> prePnts3DStereo;
    std::vector<cv::Point2d> pnts2DStereo, prePnts2DMono, pnts2DMono;
    Eigen::Matrix4Xd hPnts3D;

    FeatureData(Zed_Camera* zedPtr);

    void compute3DPoints(SubPixelPoints& prePnts, const int keyNumb);

};

class FeatureTracker
{
    private :

#if SAVEODOMETRYDATA
#if KITTI_DATASET
        std::string filepath {KITTI_SEQ + std::string(".txt")};
#else
        std::string filepath {"zedPoses.txt"};
#endif
#else
        std::string filepath {"empty.txt"};
#endif

        std::ofstream datafile;
        std::chrono::_V2::system_clock::time_point startTime, endTime;
        std::chrono::duration<float> duration;

        std::vector<KeyFrame> keyframes;

        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d keyFramePose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEstFrame = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEstFrameInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevWPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevWPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevposeEstFrame = Eigen::Matrix4d::Identity();
        cv::Mat prevR = (cv::Mat_<double>(3,3) << 1,0,0,0, 1,0,0,0,1);
        cv::Mat pTvec = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat pRvec = cv::Mat::zeros(3,1, CV_64F);
        const int waitImKey {1};
        const int waitImMat {1};
        const int waitImOpt {1};
        const int waitImPro {1};
        const int mnSize {100};
        const int mxSize {400};
        const int mnInKal {30};
        const int sampleSize {15};
        const int gridVelNumb {10};

        // Optimization Parameters
        const size_t mxIter {25};
        const float mnErr {1.0f};
        
        int uStereo {0};
        int uMono {0};
        int keyNumb {0};
        int curFrame {0};

        bool addFeatures {false};
        bool bigRot {false};
        bool redo {false};



        SubPixelPoints tr;



        ImageData pLIm, pRIm, lIm, rIm;
        cv::Mat rmap[2][2];
        Zed_Camera* zedPtr;
        FeatureExtractor fe;
        FeatureMatcher fm;
        SubPixelPoints pnts,prePnts;
        PoseEstimator pE;
        FeatureData fd;

        const double dt;
        LKalmanFilter lkal;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01))};

        std::vector<float> gridTraX;
        std::vector<float> gridTraY;

        void saveData();

    public :

        FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr);

        void initializeTracking();
        void beginTracking(const int frames);
        void beginTrackingTrial(const int frames);
        void beginTrackingTrialClose(const int frames);

        void updateKeys(const int frame);
        void updateKeysClose(const int frame);

        void stereoFeatures(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts);
        void stereoFeaturesMask(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts);
        void stereoFeaturesClose(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts);
        void stereoFeaturesPop(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts);
        void calculateNextPnts();
        void calculateNextPntsGrids();
        void opticalFlow();
        void opticalFlowPredict();
        void opticalFlowGood();
        void matcherTrial();
        void getEssentialPose();
        void getSolvePnPPose();
        void getSolvePnPPoseWithEss();
        void getPoseCeres();
        void getPoseCeresNew();

        void optimizePoseMO(std::vector<cv::Point3d>& p3D, cv::Mat& Rvec, cv::Mat& tvec);
        void optimizePoseMotionOnly(std::vector<cv::Point3d>& p3D, cv::Mat& Rvec, cv::Mat& tvec);
        void get3DClose(std::vector<cv::Point3d>& p3D, std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose);
        void getIdxVec(std::vector<int>& idxVec, const size_t size);
        void getSamples(std::vector<int>& idxVec,std::set<int>& idxs);
        void ceresRansac(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec);
        void ceresClose(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec);
        void ceresWeights(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec, std::vector<float>& weights);
        void ceresMO(std::vector<cv::Point3d>& p3Dclose, std::vector<cv::Point2d>& p2Dclose, cv::Mat& Rvec, cv::Mat& tvec);
        void checkKeyDestrib(std::vector<cv::Point2d>& p2Dclose);

        void get3dPointsforPose(std::vector<cv::Point3d>& p3D);
        void get3dPointsforPoseAll(std::vector<cv::Point3d>& p3D);
        void poseEstKal(cv::Mat& Rvec, cv::Mat& tvec, const size_t p3dsize);
        void essForMonoPose(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D);
        void pnpRansac(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D);
        void compute2Dfrom3D(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D, std::vector<cv::Point2d>& pn2D);
        int calcNumberOfStereo();
        float norm_pdf(float x, float mu, float sigma);
        void getWeights(std::vector<float>& weights, std::vector<cv::Point2d>& p2Dclose);

        void setLRImages(const int frameNumber);
        void setLImage(const int frameNumber);
        void setPreLImage();
        void setPreRImage();
        void setPre();
        void setPreTrial();
        void checkBoundsLeft();
        void setPreInit();
        void clearPre();

        float calcDt();

        cv::Mat getLImage();
        cv::Mat getRImage();
        cv::Mat getPLImage();
        cv::Mat getPRImage();

        void rectifyLRImages();
        void rectifyLImage();

        void drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys);
        void drawMatches(const cv::Mat& lIm, const SubPixelPoints& pnts, const std::vector<cv::DMatch> matches);
        void drawMatchesKeys(const cv::Mat& lIm, const std::vector<cv::Point2f>& keys1, const std::vector<cv::Point2f>& keys2, const std::vector<cv::DMatch> matches);
        void drawOptical(const char* com, const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts);
        void drawPoints(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts, const char* str);
        void draw2D3D(const cv::Mat& im, const std::vector<cv::Point2d>& p2Dfp3D, const std::vector<cv::Point2d>& p2D);

        void calcGridVel();
        bool checkProjection3D(cv::Point3d& point3D, cv::Point2d& point2d);
        void predictProjection3D(const cv::Point3d& point3D, cv::Point2d& point2d);
        bool checkFeaturesArea(const SubPixelPoints& prePnts);
        bool checkFeaturesAreaCont(const SubPixelPoints& prePnts);
        void setMask(const SubPixelPoints& prePnts, cv::Mat& mask);
        void setPopVec(const SubPixelPoints& prePnts, std::vector<int>& pop);
        void changeUndef(std::vector<float>& err, std::vector <uchar>& inliers, std::vector<cv::Point2f>& temp);
        inline void setVel(double& pvx, double& pvy, double& pvz, double vx, double vy, double vz);
        void checkPointsDist(std::vector<cv::Point2f>& pnts1, std::vector<cv::Point2f>& pnts2);
        float pointsDist(const cv::Point2f& p1, const cv::Point2f& p2);
        void changeOptRes(std::vector <uchar>&  inliers, std::vector<cv::Point2f>& pnts1, std::vector<cv::Point2f>& pnts2);

        void convertToEigen(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& tr);
        void publishPose();
        void publishPoseTrial();

};



} // namespace vio_slam


#endif // FEATURETRACKER_H