#pragma once

#ifndef SYSTEM_H
#define SYSTEM_H

#include "Settings.h"
#include "Camera.h"
#include "Frame.h"
#include "FeatureTracker.h"
#include "Map.h"
#include "LocalBA.h"
#include <thread>
#include <string>



namespace vio_slam
{

class System
{

        private:

        // image_transport::ImageTransport m_it;
        // message_filters::Subscriber<sensor_msgs::Image> subLeftIm;
        // message_filters::Subscriber<sensor_msgs::Image> subRightIm;
        // message_filters::Synchronizer<MySyncPolicy> img_sync;
                

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
        void MultiSLAM2();
        void trackNewImage(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb);
        void trackNewImageMutli(const cv::Mat& imLRect, const cv::Mat& imRRect, const cv::Mat& imLRectB, const cv::Mat& imRRectB, const int frameNumb);
        void saveTrajectory(const std::string& filepath);
        void saveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition);

        void exitSystem();

        // void ImagesCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);

        std::thread* Visual;
        std::thread* Tracking;
        std::thread* LocalMapping;
        std::thread* LoopClosure;

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
        LocalMapper* loopCl;

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