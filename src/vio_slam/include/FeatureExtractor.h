#pragma once

#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include "Settings.h"
#include <ros/ros.h>
#include <iostream>

namespace vio_slam
{

class FeatureExtractor
{

    enum FeatureChoice
    {
        ORB,
        FAST
    };

    class FindFeatures
    {
        const int nFeatures;
        FindFeatures(const int _nfeatures = 1000);
    };
    
    public:
        FeatureExtractor();
    // FindFeatures orbs;


};

} // namespace vio_slam

#endif // FEATUREEXTRACTOR_H