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

namespace vio_slam
{

class MapPoint
{
    private:

    public:

        Eigen::Vector4d point;
        int trackCnt {0};
        bool inFrame {false};
};

class Map
{
    private:

    public:

        std::vector<MapPoint> Points;
};

} // namespace vio_slam

#endif // MAP_H