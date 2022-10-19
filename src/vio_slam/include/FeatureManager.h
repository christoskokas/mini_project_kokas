#pragma once

#ifndef FEATUREMANAGER_H
#define FEATUREMANAGER_H

#include "FeatureExtractor.h"
#include "Camera.h"
#include "opencv2/core.hpp"



namespace vio_slam
{

class FeatureManager
{
    private:


    public:

        std::vector<cv::Point3d> prevPoints3DStereo, prevPoints3DMono;
        std::vector<cv::Point2d> points2DStereo, points2DMono;

        
        FeatureManager(){};
        void calculate3DPoints(SubPixelPoints& prevPoints, SubPixelPoints& points, const Zed_Camera* zedcamera);
        void clear();


};

} // namespace vio_slam

#endif // FEATUREMANAGER_H