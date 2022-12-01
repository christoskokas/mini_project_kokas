#pragma once

#ifndef MAP_H
#define MAP_H

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
#include <unordered_map>

namespace vio_slam
{

struct Observation
{
    int frameId {};

    cv::KeyPoint obs;
};

class MapPoint
{
    private:

    public:

        Eigen::Vector4d wp;
        int trackCnt {0};
        std::vector<Observation> obs;
        cv::Mat desc;

        bool inFrame {true};
        bool isOutlier {false};

        int keyFrameNb {0};
        const int idx;
        const int kdx;

        bool getPosInFrame();
        void SetInFrame(bool infr);
        void SetIsOutlier(bool isOut);
        bool GetIsOutlier();
        bool GetInFrame();
        MapPoint(Eigen::Vector4d& p, const int _kdx, const int _iddx);


        void addTCnt();

        Eigen::Vector4d getWordPose();
};

class Map
{
    private:

    public:
        std::unordered_map<unsigned long, KeyFrame*> keyFrames;
        std::unordered_map<unsigned long, MapPoint*> mapPoints;
        unsigned long kIdx {0};
        unsigned long pIdx {0};
        Map(){};
        void addMapPoint(Eigen::Vector4d& p);
        void addKeyFrame(Eigen::Matrix4d _pose, const int _numb);
};

} // namespace vio_slam

#endif // MAP_H