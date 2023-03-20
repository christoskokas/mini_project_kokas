#pragma once

#ifndef LOCALBA_H
#define LOCALBA_H


#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Optimizer.h"
#include "Eigen/Dense"
#include <fstream>
#include <string>
#include <iostream>
#include <random>

namespace vio_slam
{


class LocalMapper
{
    private:

    public:

        bool stopRequested {false};

        const double fx,fy,cx,cy;

        Map* map;

        const float reprjThreshold {7.815f};

        const int actvKFMaxSize {10};
        const int minCount {3};

        Zed_Camera* zedPtr;
        Zed_Camera* zedPtrB;

        FeatureMatcher* fm;
        FeatureMatcher* fmB;

        LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm);

        LocalMapper(Map* _map, Zed_Camera* _zedPtr, Zed_Camera* _zedPtrB, FeatureMatcher* _fm);

        void loopClosureR(std::vector<vio_slam::KeyFrame *>& actKeyF);
        void loopClosureRB(std::vector<vio_slam::KeyFrame *>& actKeyF);
        void beginLoopClosure();
        void beginLoopClosureB();
        void insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const int lastActKF, int& blocks, const bool back);
        
        void triangulateNewPointsRB(const Zed_Camera* zedCam,std::vector<vio_slam::KeyFrame *>& activeKF, const bool back);
        void calcAllMpsOfKFROnlyEstB(const Zed_Camera* zedCam, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale, const bool back);
        void predictKeysPosRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);
        void predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, std::vector<std::pair<float, float>>& keysAngles, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);
        void localBARB(std::vector<vio_slam::KeyFrame *>& actKeyF);
        void insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const int lastActKF, int& blocks, const bool back);
        bool checkReprojErrNewRB(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec, const bool back);

        
        
        void calcProjMatricesRB(const Zed_Camera* zedCam,std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF,const bool back);
        void processMatchesRB(std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, const bool back);
        void addMultiViewMapPointsRB(const Eigen::Vector4d& posW, const std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos, const bool back);
        void addNewMapPointsB(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const bool back);

        

        void triangulateNewPointsR(std::vector<vio_slam::KeyFrame *>& activeKF);
        void calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale);
        void calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF);
        void processMatchesR(std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points);
        bool checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec);
        void addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos);
        void addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs);
        void localBAR(std::vector<vio_slam::KeyFrame *>& actKeyF);
        bool checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);

        void beginLocalMapping();
        void beginLocalMappingB();
        Eigen::Vector3d TriangulateMultiViewPoint(
                const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices,
                const std::vector<Eigen::Vector2d>& points);
        bool checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);
        Eigen::Vector3d get3d(const cv::KeyPoint& key, const float depth);
        void triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first);

};



} // namespace vio_slam


#endif // LOCALBA_H