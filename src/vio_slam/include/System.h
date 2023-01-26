#pragma once

#ifndef SYSTEM_H
#define SYSTEM_H

#include "Settings.h"
#include "Camera.h"
#include "trial.h"
#include "Frame.h"
#include "Map.h"
#include "LocalBA.h"
#include <thread>
#include <string>



namespace vio_slam
{

class System
{

    private:

    public:

#if KITTI_DATASET
        const int nFeatures {2000};
#else
        const int nFeatures {1000};
#endif

        System(ConfigFile* _mConf);
        System(ConfigFile* _mConf, bool multi);

        void SLAM();
        void MultiSLAM();

        std::thread* Visual;
        std::thread* Tracking;
        std::thread* LocalMapping;

        std::thread* FeatTrack;
        std::thread* FeatTrackB;

        FeatureTracker* featTracker;
        FeatureTracker* featTrackerB;

        ViewFrame* mFrame;

        Zed_Camera* mZedCamera;
        Zed_Camera* mZedCameraB;

        ConfigFile* mConf;

        Map* map;
        Map* mapB;

        LocalMapper* localMap;

        FeatureExtractor* feLeft;
        FeatureExtractor* feLeftB;

        FeatureExtractor* feRight;
        FeatureExtractor* feRightB;
        
        FeatureMatcher* fm;
        FeatureMatcher* fmB;

        const int minNStereo {70};
        const int minNMono {20};
        const float maxPerc {0.5};

        Eigen::Vector3d prevV, prevVB, prevT, prevTB;


        bool checkDisplacement(Eigen::Matrix4d& estimPose, Eigen::Matrix4d& estimPoseB, Eigen::Matrix4d& predPose, Eigen::Matrix4d& transfC1C2Inv, Eigen::Matrix4d& transfC1C2);

        void setActiveOutliers(Map* map, std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<bool>& MPsMatches);
        void insertKF(Map* map, KeyFrame* kF, std::vector<int>& matchedIdxsN, const Eigen::Matrix4d& estimPose, const int nStereo, const int nMono, const bool front, const bool addKF);
        void drawTrackedKeys(KeyFrame* kF, std::vector<int> matched, const char* com, cv::Mat& im);
        Eigen::Matrix4d changePosesFromBoth(Eigen::Matrix4d& estimPose, Eigen::Matrix4d& estimPoseB);

};

} // namespace vio_slam



#endif // SYSTEM_H