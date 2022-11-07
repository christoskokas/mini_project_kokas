#pragma once

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

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

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3d& point, const cv::Point2d& obs)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorMono, 2, 6>(
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

#endif