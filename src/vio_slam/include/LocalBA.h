#pragma once

#ifndef LOCALBA_H
#define LOCALBA_H


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

        const double fx,fy,cx,cy;

        Map* map;

        const float reprjThreshold {7.815f};

        Zed_Camera* zedPtr;

        FeatureMatcher* fm;

        LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm);

        void predictKeysPos(TrackedKeys& keys, const Eigen::Matrix4d& curPose, const Eigen::Matrix4d& camPoseInv);
        void beginLocalMapping();
        void computeAllMapPoints();
        Eigen::Vector3d TriangulateMultiViewPoint(
                const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices,
                const std::vector<Eigen::Vector2d>& points);
        void triangulateCeres(Eigen::Vector4d& p4d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs);
        void calcProjMatrices(std::unordered_map<int, Eigen::Matrix<double,3,4>>& projMatrices);

        void processMatches(std::vector<std::pair<int, int>>& matchesOfPoint, std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points);
        bool checkReprojErr(Eigen::Vector4d& calcVec, std::vector<std::pair<int, int>>& matchesOfPoint, const std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices);
        void projectToPlane(Eigen::Vector4d& vec, cv::Point2f& p2f);

        void drawPred(KeyFrame* lastKF, std::vector<cv::KeyPoint>& keys,std::vector<cv::KeyPoint>& predKeys);
        void drawLBA(const char* com,std::vector<std::vector<std::pair<int, int>>>& matchedIdxs, const KeyFrame* lastKF, const KeyFrame* otherKF);

};



} // namespace vio_slam


#endif // LOCALBA_H