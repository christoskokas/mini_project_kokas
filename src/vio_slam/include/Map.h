#pragma once

#ifndef MAP_H
#define MAP_H

#include "Camera.h"
#include "KeyFrame.h"
#include "PoseEstimator.h"
#include "FeatureManager.h"
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

class KeyFrame;

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
        Eigen::Vector3d wp3d;
        int unMCnt {0};
        int outLCnt {0};
        int trackCnt {0};
        int seenCnt {1};
        // std::vector<Observation> obs;
        std::vector<cv::KeyPoint> obs;
        cv::KeyPoint lastObsL;
        cv::KeyPoint lastObsR;
        KeyFrame* lastObsKF;
        cv::Mat desc;
        std::unordered_map<KeyFrame*, size_t> kFWithFIdx;
        std::unordered_map<KeyFrame*, std::pair<int,int>> kFMatches;
        std::unordered_map<KeyFrame*, std::pair<int,int>> kFMatchesB;

        int LBAID {-1};

        float maxScaleDist, minScaleDist;

        bool isActive {true};

        bool inFrame {true};
        bool inFrameR {true};
        bool isOutlier {false};
        bool close {true};
        bool added {false};

        cv::Point2f predL, predR;
        float predAngleL, predAngleR;

        int scaleLevel {0};
        int prdScaleLevel {0};
        int scaleLevelL {0};
        int scaleLevelR {0};

        int keyFrameNb {0};
        const unsigned long idx;
        const unsigned long kdx;

        void update(KeyFrame* kF);
        void update(KeyFrame* kF, const bool back);
        int predictScale(float dist);
        void addConnection(KeyFrame* kF, const std::pair<int,int>& keyPos);
        void addConnectionB(KeyFrame* kF, const std::pair<int,int>& keyPos);

        void eraseKFConnection(KeyFrame* kF);
        void setActive(bool act);
        void SetInFrame(bool infr);
        void SetIsOutlier(bool isOut);
        bool getActive() const;
        bool GetIsOutlier() const;
        bool GetInFrame() const;
        void calcDescriptor();
        MapPoint(const Eigen::Vector4d& p, const cv::Mat& _desc, const cv::KeyPoint& obsK, const bool _close, const unsigned long _kdx, const unsigned long _idx);
        MapPoint(const unsigned long _idx, const unsigned long _kdx);

        void copyMp(MapPoint* mp, const Zed_Camera* zedPtr);
        void changeMp(const MapPoint* mp);


        // MapPoint operator = (MapPoint const& obj)
        // {
        //     MapPoint res(obj.idx, obj.kdx);

        // }

        void addTCnt();

        Eigen::Vector4d getWordPose4d() const;
        Eigen::Vector3d getWordPose3d() const;
        void updatePos(const Eigen::Vector3d& newPos, const Zed_Camera* zedPtr);
        void setWordPose4d(const Eigen::Vector4d& p);
        void setWordPose3d(const Eigen::Vector3d& p);
        void updateMapPoint(Eigen::Vector4d& p, const cv::Mat& _desc, cv::KeyPoint& _obs);
};

class Map
{
    private:

    public:

        bool endOfFrames {false};

        std::unordered_map<unsigned long, KeyFrame*> keyFrames;
        std::unordered_map<unsigned long, MapPoint*> mapPoints;
        std::vector<MapPoint*> activeMapPoints;
        std::vector<MapPoint*> activeMapPointsB;
        std::vector<KeyFrame*> activeKeyFrames;
        unsigned long kIdx {0};
        unsigned long pIdx {0};
        
        bool keyFrameAdded {false};
        bool keyFrameAddedMain {false};
        bool frameAdded {false};
        bool LBADone {false};
        int endLBAIdx {0};

        Map(){};
        void addMapPoint(Eigen::Vector4d& p, const cv::Mat& _desc, cv::KeyPoint& obsK, bool _useable);
        void addMapPoint(MapPoint* mp);
        void addKeyFrame(Eigen::Matrix4d _pose);
        void addKeyFrame(KeyFrame* kF);
        void removeKeyFrame(int idx);
        mutable std::mutex mapMutex;

    protected:
};

} // namespace vio_slam

#endif // MAP_H