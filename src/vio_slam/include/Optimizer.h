#pragma once

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class Reprojection3dError
{
public:

    Reprojection3dError(Eigen::Vector3d point_, Eigen::Vector3d observed_)
        : point(point_), observed(observed_)
    {
    }

    template<typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x());
        pt1[1] = T(point.y());
        pt1[2] = T(point.z());

        T pt2[3];
        ceres::AngleAxisRotatePoint(camera, pt1, pt2);

        pt2[0] = pt2[0] + camera[3];
        pt2[1] = pt2[1] + camera[4];
        pt2[2] = pt2[2] + camera[5];

        residuals[0] = observed[0] - pt2[0];
        residuals[1] = observed[1] - pt2[1];
        residuals[2] = observed[2] - pt2[2];

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Vector3d point, Eigen::Vector3d observed)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::AutoDiffCostFunction<Reprojection3dError, 3, 6>(
                        new Reprojection3dError(point, observed)));
    }

private:
    Eigen::Vector3d point;
    Eigen::Vector3d observed;
    // Camera intrinsics
    double K[4] = {265.795, 265.6975, 338.7225, 186.95575}; // fx,fy,cx,cy
};

class ReprojectionError
{
public:

    ReprojectionError(Eigen::Vector3d point_, Eigen::Vector2d observed_)
        : point(point_), observed(observed_)
    {
    }

    template<typename T>
    bool operator()(const T* const camera_r, const T* const camera_t, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x());
        pt1[1] = T(point.y());
        pt1[2] = T(point.z());

        T pt2[3];
        ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);

        pt2[0] = pt2[0] + camera_t[0];
        pt2[1] = pt2[1] + camera_t[1];
        pt2[2] = pt2[2] + camera_t[2];

        const T xp = T(K[0] * (pt2[0] / pt2[2]) + K[2]);
        const T yp = T(K[1] * (pt2[1] / pt2[2]) + K[3]);

        const T u = T(observed.x());
        const T v = T(observed.y());

        residuals[0] = u - xp;
        residuals[1] = v - yp;

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Vector3d points, Eigen::Vector2d observed)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                        new ReprojectionError(points, observed)));
    }

private:
    Eigen::Vector3d point;
    Eigen::Vector2d observed;
    // Camera intrinsics
    double K[4] = {520.9, 521.0, 325.1, 249.7}; // fx,fy,cx,cy
};


#endif