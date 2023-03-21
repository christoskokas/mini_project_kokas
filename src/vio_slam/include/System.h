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

        public:

#if KITTI_DATASET
        const int nFeatures {2000};
#else
        const int nFeatures {1000};
#endif

        System(ConfigFile* _mConf);
        System(ConfigFile* _mConf, bool multi);

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

        LocalMapper* localMap;
        LocalMapper* loopCl;

        FeatureExtractor* feLeft;
        FeatureExtractor* feLeftB;

        FeatureExtractor* feRight;
        FeatureExtractor* feRightB;
        
        FeatureMatcher* fm;

};

} // namespace vio_slam



#endif // SYSTEM_H