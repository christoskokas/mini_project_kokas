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

        Map* map;

        LocalMapper(Map* _map);

        void beginLocalMapping();


};



} // namespace vio_slam


#endif // LOCALBA_H