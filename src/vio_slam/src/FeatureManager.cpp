#include "FeatureManager.h"

namespace vio_slam
{

void FeatureManager::calculate3DPoints(SubPixelPoints& prevPoints, SubPixelPoints& points, const Zed_Camera* zedcamera)
{
    const size_t end {prevPoints.left.size()};

    prevPoints3DStereo.reserve(end);
    points2DStereo.reserve(end);
    points2DMono.reserve(end);

    for (size_t i = 0; i < end; i++)
    {   
        const double cx {zedcamera->cameraLeft.cx};
        const double cy {zedcamera->cameraLeft.cy};
        const double fx {zedcamera->cameraLeft.fx};
        const double fy {zedcamera->cameraLeft.fy};

        // const double x = (double)(((double)points.left[i].x-cx)*(double)points.depth[i]/fx);
        // const double y = (double)(((double)points.left[i].y-cy)*(double)points.depth[i]/fy);
        // const double z = (double)points.depth[i];
        points2DMono.emplace_back(cv::Point2d((double)points.left[i].x,(double)points.left[i].y));
        // prevPoints3DMono.emplace_back(cv::Point3d(xp,yp,zp));


        if (!prevPoints.useable[i] || !points.useable[i])
            continue;


        const double xp = (double)(((double)prevPoints.left[i].x-cx)*(double)prevPoints.depth[i]/fx);
        const double yp = (double)(((double)prevPoints.left[i].y-cy)*(double)prevPoints.depth[i]/fy);
        const double zp = (double)prevPoints.depth[i];

        prevPoints3DStereo.emplace_back(cv::Point3d(xp,yp,zp));
        points2DStereo.emplace_back(cv::Point2d((double)points.left[i].x,(double)points.left[i].y));
        
    }
}

void FeatureManager::clear()
{
    prevPoints3DStereo.clear();
    points2DMono.clear();
    points2DStereo.clear();
}

}