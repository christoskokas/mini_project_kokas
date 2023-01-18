#pragma once

#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "PoseEstimator.h"
#include "FeatureManager.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
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
        std::string filepath {KITTI_SEQ + std::string("lel.txt")};
#else
        std::string filepath {"zedPoses.txt"};
#endif
#else
        std::string filepath {"empty.txt"};
#endif

#if KITTI_DATASET
        const int nFeatures {2000};
#else
        const int nFeatures {1000};
#endif

        std::ofstream datafile;
        std::chrono::_V2::system_clock::time_point startTime, endTime;
        std::chrono::duration<float> duration;

        std::vector<KeyFrame> keyframes;

        std::vector<MapPoint*>& activeMapPoints;
        std::vector<KeyFrame*>& activeKeyFrames;

        // std::vector<int> activeMapPointsCorr;
        std::vector<int> mPPerKeyFrame;

        std::vector<int> reprojIdxs;

        Eigen::Matrix4d referencePose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d lastKFPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d lastKFPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d keyFramePose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEstFrame = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEstFrameInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevWPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevWPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevposeEstFrame = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

        cv::Mat prevR = (cv::Mat_<double>(3,3) << 1,0,0,0, 1,0,0,0,1);
        cv::Mat pTvec = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat pRvec = cv::Mat::zeros(3,1, CV_64F);
        const int waitImKey {1};
        const int waitImMat {1};
        const int waitImOpt {1};
        const int waitImPro {1};
        const int waitImRep {1};
        const int waitImClo {1};
        const int waitTrials {1};
        const int mnSize {100};
        const int mxSize {400};
        const int highSpeed {15};
        const int mxMonoSize {300};
        const int mnInKal {30};
        const int sampleSize {15};
        const int gridVelNumb {10};
        const int maskRadius {15};
        const int keyFrameConThresh {5};
        const int maxKeyFrameDist {10};
        const int keyFrameInsertThresh {1};
        const int actvKFMaxSize {10};
        const int maxActvKFMaxSize {50};
        const int minNStereo {70};
        const int minNMono {20};
        const int maxActiveMPSize {800};

        int lastKFTrackedNumb {0};

        const double imageDifThres {0.93};
        const double noMovementCheck {0.96};

        const double fx,fy,cx,cy;

        const size_t gfmxCount {400};
        const double gfmnDist {30.0f};

        // Optimization Parameters
        const size_t mxIter {25};
        const float mnErr {1.0f};

        const int keyFrameCountEnd {5};
        int insertKeyFrameCount {0};
        int lastValidKF {0};
        int uStereo {0};
        int uMono {0};
        int keyNumb {0};
        int curFrame {0};
        int curFrameNumb {-1};

        float leftRight {0};

        bool addFeatures {false};
        bool bigRot {false};
        bool redo {false};
        std::vector<KeyFrame*> allFrames;
        cv::Mat prevF;

        SubPixelPoints tr;

        Sophus::Vector6d displacement = Sophus::Vector6d::Zero();

        ImageData pLIm, pRIm, lIm, rIm;
        cv::Mat rmap[2][2];
        cv::Mat lastKFImage;
        Zed_Camera* zedPtr;
        FeatureExtractor fe;
        FeatureExtractor feLeft;
        FeatureExtractor feRight;
        FeatureMatcher fm;
        SubPixelPoints pnts,prePnts;
        PoseEstimator pE;
        FeatureData fd;

        Map* map;

        TrackedKeys prevLeftPnts;

        const double dt;
        LKalmanFilter lkal;

        cv::TermCriteria criteria {cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01))};

        std::vector<float> gridTraX;
        std::vector<float> gridTraY;

        std::vector<cv::Point3d> activePoints;

        void saveData();

    public :

        

        FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr, Map* _map);
        FeatureTracker(Zed_Camera* _zedPtr, Map* _map);

        Eigen::Matrix4d TrackImage(const cv::Mat& leftRect, const cv::Mat& rightRect, const Eigen::Matrix4d& predPose, const int frameNumb);

        void checkPrevAngles(std::vector<float>& mapAngles, std::vector<cv::KeyPoint>& prevKeys, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB, const TrackedKeys& keysLeft);

        void calcPrevFramePos(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& prevKeyPos, const Eigen::Matrix4d& prevPose);

        void checkWithFund(const std::vector<cv::KeyPoint>& activeKeys, const std::vector<cv::KeyPoint>& newKeys, std::vector<int>& matchedIdxsB, std::vector<int>& matchedIdxsN);

        void calculatePrevKeyPos(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, const Eigen::Matrix4d& currPoseInv, const Eigen::Matrix4d& predPoseInv);

        void changePosesLBA();
        void publishPoseLBA();
        void insertKeyFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, const int nStereo, const int nMono, const Eigen::Matrix4d& estimPose);
        void insertFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, const int nStereo, const Eigen::Matrix4d& estimPose);
        void addFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, const int nStereo, const Eigen::Matrix4d& estimPose);
        bool worldToFrame(MapPoint* mp, const Eigen::Matrix4d& pose);
        bool worldToFrameKF(MapPoint* mp, const Eigen::Matrix4d& pose);

        void kalmanF(Eigen::Matrix4d& calcPoseDif);

        void calcAngles(std::vector<MapPoint*>& activeMapPoints, std::vector<cv::KeyPoint>& projectedPoints, std::vector<cv::KeyPoint>& prevKeyPos, std::vector<float>& mapAngles);

        void removeKeyFrame(std::vector<vio_slam::KeyFrame *>& activeKeyFrames);

        bool checkDisplacement(const Eigen::Matrix4d& currPose, Eigen::Matrix4d& estimPose);
        void removeMapPoints(std::vector<MapPoint*>& activeMapPoints, std::vector<bool>& toRemove);
        void removeMapPointOut(std::vector<MapPoint*>& activeMapPoints, const Eigen::Matrix4d& estimPose, std::vector<bool>& toRemove);
        void removeMapPointOutBackUp(std::vector<MapPoint*>& activeMapPoints, const Eigen::Matrix4d& estimPose);
        void addKeyFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsN, const int nStereo);
        bool check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const float weight);
        bool check3dError(const Eigen::Vector4d& p4d, const Eigen::Vector4d& obs, const double thres, const float weight);
        int OutliersReprojErr(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsB, const double thres, const std::vector<float>& weights, int& nInliers);
        std::pair<int,int> refinePose(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsB, Eigen::Matrix4d& estimPose);
        std::pair<int,int> estimatePoseCeres(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsB, Eigen::Matrix4d& estimPose, const bool first);
        void publishPoseNew();
        void removePnPOut(std::vector<int>& idxs, std::vector<int>& matchedIdxsN, std::vector<int>& matchedIdxsB);
        void worldToImg(std::vector<MapPoint*>& MapPointsVec, std::vector<cv::KeyPoint>& projectedPoints);
        void worldToImg(std::vector<MapPoint*>& MapPointsVec, std::vector<cv::KeyPoint>& projectedPoints, const Eigen::Matrix4d& currPoseInv);
        void worldToImgAng(std::vector<MapPoint*>& MapPointsVec, std::vector<float>& mapAngles, const Eigen::Matrix4d& currPoseInv, std::vector<cv::KeyPoint>& prevKeyPos, std::vector<cv::KeyPoint>& projectedPoints);
        void getPoints3dFromMapPoints(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<cv::Point3d>& points3d, std::vector<cv::Point2d>& points2d, std::vector<int>& matchedIdxs);

        void initializeTracking();
        void initializeTrackingGoodFeatures();
        void Track(const int frames);
        void Track2(const int frames);

        void get3dFromKey(Eigen::Vector4d& pnt4d, const cv::KeyPoint& pnt, const float depth);
        void computeStereoMatchesORB(TrackedKeys& keysLeft, TrackedKeys& prevLeftKeys);

        void initializeMap(TrackedKeys& keysLeft);
        void Track5(const int frames);
        void Track4(const int frames);
        void Track3(const int frames);
        void beginTracking(const int frames);
        void beginTrackingTrial(const int frames);
        void beginTrackingTrialClose(const int frames);
        void beginTrackingGoodFeatures(const int frames);

        void cloneTrackedKeys(TrackedKeys& prevLeftKeys, TrackedKeys& leftKeys);
        void reduceTrackedKeysMatches(TrackedKeys& prevLeftKeys, TrackedKeys& leftKeys);
        void reduceTrackedKeys(TrackedKeys& leftKeys, std::vector<bool>& inliers);
        void updateMapPoints(TrackedKeys& prevLeftKeys);
        void predictORBPoints(TrackedKeys& prevLeftKeys);

        void extractORB(cv::Mat& leftIm, cv::Mat& rightIm, StereoKeypoints& keys, StereoDescriptors& desc);
        void extractORBStereoMatch(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft);
        void extractFAST(const cv::Mat& leftIm, const cv::Mat& rightIm, StereoKeypoints& keys, StereoDescriptors& desc, const std::vector<cv::Point2f>& prevPnts);
        void updateKeys(const int frame);
        void updateKeysGoodFeatures(const int frame);
        void updateKeysClose(const int frame);
        bool wPntToCamPose(const cv::Point3d& p3, cv::Point2d& p2, cv::Point3d& p3cam);

        void removeOutliers(const std::vector<cv::Point2d>& curPntsd);
        void setTrackedLeft(std::vector<cv::Point2d>& curPntsd);

        void pointsInFrame(std::vector<cv::Point3d>& p3D);

        void computeOpticalLeft(TrackedKeys& keysLeft);
        void computeStereoMatches(TrackedKeys& keysLeft, TrackedKeys& prevLeftKeys);

        void findFeaturesTh(const cv::Mat& lim, const cv::Mat& rim, const cv::Mat& pLim, const cv::Mat& pRim, PointsWD& pnts);

        bool checkOutlierMap3d(const Eigen::Matrix4d& estimatedP, Eigen::Vector4d& p4d, const double thres, const float weight,  Eigen::Vector4d& obs);
        
        bool checkOutlierMap(const Eigen::Matrix4d& estimatedP, Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const float weight, cv::Point2f& out2d);
        int checkOutliersMap(const Eigen::Matrix4d& estimatedP, TrackedKeys& prevKeysLeft, TrackedKeys& newKeys, std::vector<bool>& inliers, const double thres, const std::vector<float>& weights);

        void optimizePoseCeres(TrackedKeys& prevKeys, TrackedKeys& newKeys);
        void optimizePoseORB(TrackedKeys& prevKeys, TrackedKeys& newKeys);
        void getNewMatchedPoints(TrackedKeys& keysMatched, TrackedKeys& newPnts);
        void addNewPoints(PointsWD& pntsWD);
        void addMapPnts(TrackedKeys& keysLeft);
        void addMonoPnts(TrackedKeys& keysLeft);
        void addStereoPnts();
        bool getPoseInFrame(const Eigen::Matrix4d& pose, const Eigen::Matrix4d& predPose, MapPoint* mp, cv::Point2f& pnt, cv::Point2f& predPnt);
        bool getPredInFrame(const Eigen::Matrix4d& predPose, MapPoint* mp, cv::Point2f& predPnt);
        void setMaskOfIdxs(cv::Mat& mask, const TrackedKeys& keysLeft);
        void predictPntsLeft(TrackedKeys& keysLeft);

        void calcOpticalFlowPrev(SubPixelPoints& pnts);

        void predictPntsWD(PointsWD& pnts);
        bool predictProjWD(const float depth, const cv::Point2f& p2f, cv::Point3d& p3d, cv::Point2f& p2fOut);

        void estimatePoseWD(const PointsWD& pnts);
        void opticalLeft(const cv::Mat& lim, const cv::Mat& pLim, const cv::Mat& pRim, PointsWD& leftP);
        void findFastF(const cv::Mat& im, std::vector<cv::Point2f>& pnts);
        void calcWeights(const SubPixelPoints& pnts, std::vector<float>& weights);
        bool checkOutlier(const Eigen::Matrix4d& estimatedP, const cv::Point3d& p3d, const cv::Point2f& obs, const double thres, const float weight, cv::Point2f& out2d);
        int checkOutliers(const Eigen::Matrix4d& estimatedP, const std::vector<cv::Point3d>& p3d, const std::vector<cv::Point2f>& obs, std::vector<bool>& inliers, const double thres, const std::vector<float>& weights);
        void optimizePose(SubPixelPoints& prePnts, SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec);
        void checkRotTra(cv::Mat& Rvec, cv::Mat& tvec,cv::Mat& RvecN, cv::Mat& tvecN);
        void estimatePoseN();
        void optWithSolve(SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec, const bool new3D);
        void solvePnPIni(SubPixelPoints& pnts, cv::Mat& Rvec, cv::Mat& tvec, const bool new3D);
        bool inBorder(cv::Point3d& p3d, cv::Point2d& p2d);
        void checkInBorder(SubPixelPoints& pnts);
        bool predProj(const cv::Point3d& p3d, cv::Point2d& p2d, const bool new3D);
        void predictNewPnts(SubPixelPoints& pnts, const bool new3D);
        void calcOptical(SubPixelPoints& pnts, const bool new3D);
        void calcOpticalFlow(SubPixelPoints& pnts);
        void setPre3DPnts(SubPixelPoints& prePnts, SubPixelPoints& pnts);
        void setPreviousValues();
        void setPreviousValuesIni();
        void triangulate3DPoints(SubPixelPoints& pnts);
        void findFeaturesWD(const cv::Mat& leftIm, const cv::Mat& rightIm, PointsWD& pnts);
        void findStereoFeatures(cv::Mat& leftIm, cv::Mat& rightIm, SubPixelPoints& pnts);
        void stereoFeatures(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts);
        void stereoFeaturesGoodFeatures(cv::Mat& lIm, cv::Mat& rIm, SubPixelPoints& pnts, const SubPixelPoints& prePnts);
        void stereoFeaturesMask(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts);
        void stereoFeaturesClose(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts);
        void stereoFeaturesPop(cv::Mat& lIm, cv::Mat& rIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts);
        void calculateNextPnts();
        void predictPts(std::vector<cv::Point2f>& curPnts);
        void calculateNextPntsDepth();
        void calculateNextPntsGrids();
        void opticalFlow();
        void opticalFlowPredict();
        void optFlow(std::vector<cv::Point3d>& p3D,std::vector<cv::Point2f>& pPnts, std::vector<cv::Point2f>& curPnts);
        void opticalFlowGood();
        void matcherTrial();
        void getEssentialPose();
        void getSolvePnPPose();
        void getSolvePnPPoseWithEss();
        void getPoseCeres();
        void getPoseCeresNew();
        void estimatePose(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2f>& curPnts);

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
        void get3dPointsforPoseAll(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D);
        void reprojError();
        void poseEstKal(cv::Mat& Rvec, cv::Mat& tvec, const size_t p3dsize);
        void essForMonoPose(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D);
        void pnpRansac(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D);
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
        void drawMatchesGoodFeatures(const cv::Mat& lIm, const SubPixelPoints& pnts);
        void drawMatchesKeys(const cv::Mat& lIm, const std::vector<cv::Point2f>& keys1, const std::vector<cv::Point2f>& keys2, const std::vector<cv::DMatch> matches);
        void drawOptical(const char* com, const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts);
        void drawPoints(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts, const char* str);
        void draw2D3D(const cv::Mat& im, const std::vector<cv::Point2d>& p2Dfp3D, const std::vector<cv::Point2d>& p2D);
        template <typename T, typename U>
        void drawPointsTemp(const char* com, const cv::Mat& im, const std::vector<T>& p2Dfp3D, const std::vector<U>& p2D)
        {
                cv::Mat outIm = im.clone();
                const size_t end {p2Dfp3D.size()};
                for (size_t i{0};i < end; i ++ )
                {
                        cv::circle(outIm, p2Dfp3D[i],2,cv::Scalar(0,255,0));
                        cv::line(outIm, p2Dfp3D[i], p2D[i],cv::Scalar(0,0,255));
                        cv::circle(outIm, p2D[i],2,cv::Scalar(255,0,0));
                }
                cv::imshow(com, outIm);
                cv::waitKey(waitImRep);

        }

        template <typename T, typename U, typename N>
        void draw3PointsTemp(const char* com, const cv::Mat& im, const std::vector<T>& prevpnts, const std::vector<U>& projpnts, const std::vector<N>& matchedpnts)
        {
                cv::Mat outIm = im.clone();
                const size_t end {prevpnts.size()};
                for (size_t i{0};i < end; i ++ )
                {
                        cv::circle(outIm, prevpnts[i],1,cv::Scalar(0,255,0));
                        cv::line(outIm, prevpnts[i], projpnts[i],cv::Scalar(0,0,255), 1);
                        cv::circle(outIm, projpnts[i],6,cv::Scalar(255,0,0));
                        cv::line(outIm, prevpnts[i], matchedpnts[i],cv::Scalar(0,255,255),2);
                        cv::circle(outIm, matchedpnts[i],4,cv::Scalar(255,255,0));
                }
                cv::imshow(com, outIm);
                cv::waitKey(waitImRep);

        }

        void drawKeyPointsCloseFar(const char* com, const cv::Mat& im, const TrackedKeys& keysLeft, const std::vector<cv::KeyPoint>& right);
        void drawLeftMatches(const char* com, const cv::Mat& im, const TrackedKeys& prevKeysLeft, const TrackedKeys& keysLeft);

        void calcGridVel();
        bool checkProjection3D(cv::Point3d& point3D, cv::Point2d& point2d);
        bool predictProjection3D(const cv::Point3d& point3D, cv::Point2d& point2d);
        void predictProjection3DNewDepth(const cv::Point2f& point2f, const float& depth, cv::Point2d& point2d);
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
        void publishPoseCeres();
        void publishPoseTrial();
        void refine3DPnts();

        template <typename T>
        double pointsDistTemp(const T& p1, const T& p2)
        {
                return pow(p1.x - p2.x,2) + pow(p1.y - p2.y,2);
        }

        template <typename T>
        void reduceStereoKeys(StereoKeypoints& keypoints, std::vector<T>& inliersL, std::vector<T>& inliersR)
        {
                const int grdC = feLeft.getGridCols();
                const int grdR = feLeft.getGridRows();
                const int wid = zedPtr->mWidth;
                const int hig = zedPtr->mHeight;

                const float multW = (float)grdC/(float)wid;
                const float multH = (float)grdR/(float)hig;

                int j {0};
                for (int i = 0; i < int(keypoints.left.size()); i++)
                {
                        const cv::KeyPoint& kp = keypoints.left[i];
                        if (inliersL[i])
                                keypoints.left[j++] = keypoints.left[i];
                        else
                                feLeft.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;

                }
                keypoints.left.resize(j);

                j = 0;
                for (int i = 0; i < int(keypoints.right.size()); i++)
                {
                        const cv::KeyPoint& kp = keypoints.right[i];
                        if (inliersR[i])
                                keypoints.right[j++] = keypoints.right[i];
                        else
                                feRight.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;
                }
                keypoints.right.resize(j);
        }

        template <typename T, typename U>
        void reduceStereoKeysIdx(StereoKeypoints& keypoints, std::vector<T>& idxs, std::vector<U>& nxtsPntsL, std::vector<U>& nxtsPntsR)
        {
                const int grdC = feLeft.getGridCols();
                const int grdR = feLeft.getGridRows();
                const int wid = zedPtr->mWidth;
                const int hig = zedPtr->mHeight;

                const float multW = (float)grdC/(float)wid;
                const float multH = (float)grdR/(float)hig;
                int count {0};
                int j {0};
                for (int i = 0; i < int(keypoints.left.size()); i++)
                {
                        const cv::KeyPoint& kp = keypoints.left[i];
                        if ( i == idxs[count])
                        {
                                feLeft.KeyDestrib[cvRound(nxtsPntsL[i].y * multH)][cvRound(nxtsPntsL[i].x * multW)] += 1;
                                feLeft.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;
                                count ++;
                                keypoints.left[j++] = keypoints.left[i];
                        }
                        feLeft.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;

                }
                keypoints.left.resize(j);
                count = 0;
                j = 0;
                for (int i = 0; i < int(keypoints.right.size()); i++)
                {
                        const cv::KeyPoint& kp = keypoints.right[i];
                        if ( i == idxs[count])
                        {
                                feRight.KeyDestrib[cvRound(nxtsPntsR[i].y * multH)][cvRound(nxtsPntsR[i].x * multW)] += 1;
                                feRight.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;
                                count ++;
                                keypoints.right[j++] = keypoints.right[i];
                        }
                        feRight.KeyDestrib[cvRound(kp.pt.y * multH)][cvRound(kp.pt.x * multW)] -= 1;

                }
                keypoints.right.resize(j);
        }

};



} // namespace vio_slam


#endif // FEATURETRACKER_H