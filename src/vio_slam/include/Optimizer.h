#pragma once

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Settings.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <sophus/se3.hpp>

namespace vio_slam
{

class Reprojection3dError
{
public:

    Reprojection3dError(const double observed_x, const double observed_y)
        : observed_x(observed_x), observed_y(observed_y)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T predicted_x = T(p[0]*K[0]/p[2] + K[2]);
        T predicted_y = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::AutoDiffCostFunction<Reprojection3dError, 2, 6,3>(
                        new Reprojection3dError(observed_x,observed_y)));
    }

private:
    double observed_x;
    double observed_y;
    // Camera intrinsics
    double K[4] = {265.795, 265.6975, 338.7225, 186.95575}; // fx,fy,cx,cy
};

class ReprojectionError
{
public:

    ReprojectionError(const double observed_x, const double observed_y)
        : observed_x(observed_x), observed_y(observed_y)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T predicted[2];
        predicted[0] = T(p[0]*K[0]/p[2] + K[2]);
        predicted[1] = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = predicted[0] - T(observed_x);
        residuals[1] = predicted[1] - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6,3>(
                        new ReprojectionError(observed_x,observed_y)));
    }

private:
    double observed_x;
    double observed_y;
    // Camera intrinsics
    double K[4] = {265.795, 265.6975, 338.7225, 186.95575}; // fx,fy,cx,cy
};

class ReprojectionErrorMO
{
public:

    ReprojectionErrorMO(const cv::Point3d& point, const cv::Point2d& obs)
        : point(point), obs(obs)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x);
        pt1[1] = T(point.y);
        pt1[2] = T(point.z);
        // Logging("camera0", camera[0],3);
        // Logging("camera1", camera[1],3);
        // Logging("camera2", camera[2],3);
        // Logging("camera3", camera[3],3);
        // Logging("camera4", camera[4],3);
        // Logging("camera5", camera[5],3);
        
        

        Sophus::SE3d se3;
        for(int i = 0; i < 7; ++i)
            se3.data()[i] = camera[i];

        Eigen::Vector4d vec {point.x, point.y, point.z, 1.0};
        Eigen::Vector4d pvec = se3.matrix() * vec;

        Logging("SE3", pvec,3);

        T p[3];


        ceres::AngleAxisRotatePoint(camera, pt1, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T predicted[2];
        predicted[0] = T(p[0]*K[0]/p[2] + K[2]);
        predicted[1] = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = predicted[0] - T(obs.x);
        residuals[1] = predicted[1] - T(obs.y);

        // Logging("pointx", point.x,3);
        // Logging("point", point,3);
        // Logging("pt1", pt1[0],3);
        // Logging("pt1", pt1[1],3);
        // Logging("pt1", pt1[2],3);
        // Logging("K0", K[0],3);
        // Logging("K2", K[2],3);
        // Logging("p0", p[0],3);
        // Logging("p2", p[2],3);
        // Logging("p[0]*K[0]/p[2] + K[2]", p[0]*K[0]/p[2] + K[2],3);
        // Logging("pred x", predicted[0],3);
        // Logging("pred y", predicted[1],3);
        // Logging("obs x", obs.x,3);
        // Logging("obs y", obs.y,3);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3d& point, const cv::Point2d& obs)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::NumericDiffCostFunction<ReprojectionErrorMO, ceres::CENTRAL, 2, 7>(
                        new ReprojectionErrorMO(point, obs)));
    }

private:
    cv::Point3d point;
    cv::Point2d obs;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy
};

class ReprojectionErrorMono
{
public:

    ReprojectionErrorMono(const cv::Point3d& point, const cv::Point2d& obs)
        : point(point), obs(obs)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x);
        pt1[1] = T(point.y);
        pt1[2] = T(point.z);
        // Logging("camera0", camera[0],3);
        // Logging("camera1", camera[1],3);
        // Logging("camera2", camera[2],3);
        // Logging("camera3", camera[3],3);
        // Logging("camera4", camera[4],3);
        // Logging("camera5", camera[5],3);

        T p[3];


        ceres::AngleAxisRotatePoint(camera, pt1, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T predicted[2];
        predicted[0] = T(p[0]*K[0]/p[2] + K[2]);
        predicted[1] = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = predicted[0] - T(obs.x);
        residuals[1] = predicted[1] - T(obs.y);

        // Logging("pointx", point.x,3);
        // Logging("point", point,3);
        // Logging("pt1", pt1[0],3);
        // Logging("pt1", pt1[1],3);
        // Logging("pt1", pt1[2],3);
        // Logging("K0", K[0],3);
        // Logging("K2", K[2],3);
        // Logging("p0", p[0],3);
        // Logging("p2", p[2],3);
        // Logging("p[0]*K[0]/p[2] + K[2]", p[0]*K[0]/p[2] + K[2],3);
        // Logging("pred x", predicted[0],3);
        // Logging("pred y", predicted[1],3);
        // Logging("obs x", obs.x,3);
        // Logging("obs y", obs.y,3);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3d& point, const cv::Point2d& obs)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::NumericDiffCostFunction<ReprojectionErrorMono, ceres::CENTRAL, 2, 6>(
                        new ReprojectionErrorMono(point, obs)));
    }

private:
    cv::Point3d point;
    cv::Point2d obs;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy
};

} // namespace vio_slam

#endif