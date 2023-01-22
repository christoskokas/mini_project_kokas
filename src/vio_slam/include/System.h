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

        System(std::string& confFile);

        void SLAM();

        std::thread* Visual;
        std::thread* Tracking;
        std::thread* LocalMapping;

        FeatureTracker* featTracker;

        Frame* mFrame;

        Zed_Camera* mZedCamera;

        ConfigFile* mConf;

        RobustMatcher2* mRb;

        Map* map;

        LocalMapper* localMap;

        FeatureExtractor* feLeft;

        FeatureExtractor* feRight;
        
        FeatureMatcher* fm;

        const int minNStereo {70};
        const int minNMono {20};

        void setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<bool>& MPsMatches);
        void insertKF(KeyFrame* kF, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsN, const Eigen::Matrix4d& estimPose, const int nStereo, const int nMono);

};

} // namespace vio_slam



#endif // SYSTEM_H