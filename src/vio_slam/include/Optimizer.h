#pragma once

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Settings.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <sophus/se3.hpp>
#include "opencv2/core.hpp"

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

        // Logging("SE3", pvec,3);
        // Logging("point", point,3);
        // Logging("pvec[0]", pvec[0],3);

        // T p[3];


        // ceres::AngleAxisRotatePoint(camera, pt1, p);

        // p[0] += camera[3];
        // p[1] += camera[4];
        // p[2] += camera[5];

        T predicted[2];
        predicted[0] = T(pvec[0]*K[0]/pvec[2] + K[2]);
        predicted[1] = T(pvec[1]*K[1]/pvec[2] + K[3]);

        // Logging("predicted[0]", predicted[0],3);
        // Logging("obs.x", obs.x,3);


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
    bool operator()(const T* const cameraR, const T* const cameraT, T* residuals) const
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


        ceres::AngleAxisRotatePoint(cameraR, pt1, p);

        p[0] += cameraT[0];
        p[1] += cameraT[1];
        p[2] += cameraT[2];

        T predicted_x;
        T predicted_y;
        predicted_x = T(p[0]*K[0]/p[2] + K[2]);
        predicted_y = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = predicted_x - T(obs.x);
        residuals[1] = predicted_y - T(obs.y);

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
        return (new ceres::NumericDiffCostFunction<ReprojectionErrorMono, ceres::CENTRAL, 2, 3, 3>(
                        new ReprojectionErrorMono(point, obs)));
    }

private:
    cv::Point3d point;
    cv::Point2d obs;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy
    // double K[4] = {265.795, 265.6975, 338.7225, 186.95575}; // fx,fy,cx,cy

};

class ReprojectionErrorWeighted
{
public:

    ReprojectionErrorWeighted(const cv::Point3d& point, const cv::Point2d& obs, const double& weight)
        : point(point), obs(obs), weight(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const cameraR, const T* const cameraT, T* residuals) const
    {
        T pt1[3];
        pt1[0] = T(point.x);
        pt1[1] = T(point.y);
        pt1[2] = T(point.z);

        T p[3];


        ceres::AngleAxisRotatePoint(cameraR, pt1, p);

        p[0] += cameraT[0];
        p[1] += cameraT[1];
        p[2] += cameraT[2];

        T predicted_x;
        T predicted_y;
        predicted_x = T(p[0]*K[0]/p[2] + K[2]);
        predicted_y = T(p[1]*K[1]/p[2] + K[3]);

        residuals[0] = weight * (predicted_x - T(obs.x));
        residuals[1] = weight * (predicted_y - T(obs.y));


        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3d& point, const cv::Point2d& obs, const double& weight)
    {
        // AutoDiffCostFunction<Reprojection3dError, 3, 6> 3 = dimensions of residuals, 6 = dimensions of camera(first input),if more then they go after camera accoring to bool operator()(const T* const camera, T* residuals).
        return (new ceres::NumericDiffCostFunction<ReprojectionErrorWeighted, ceres::CENTRAL, 2, 3, 3>(
                        new ReprojectionErrorWeighted(point, obs, weight)));
    }

private:
    cv::Point3d point;
    cv::Point2d obs;
    double observed_x;
    double observed_y;
    double weight;
    // Camera intrinsics
    double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy
    // double K[4] = {265.795, 265.6975, 338.7225, 186.95575}; // fx,fy,cx,cy

};

class OptimizePose
{
public:

    OptimizePose(const Eigen::Matrix3d& K, const Eigen::Vector3d& point, const Eigen::Vector2d& observation, const float weight)
        : K_(K), point_(point), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * point_.template cast<T>() + p_frame;
    // Compute the map point pose in pixel frame.
        Eigen::Matrix<T, 3, 1> projected = K_ * p_cp;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - projected[0] / projected[2];
        residuals[1] =
            observation_.template cast<T>()[1] - projected[1] / projected[2];


        residuals[0] = T(weight_) * residuals[0];
        residuals[1] = T(weight_) * residuals[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Vector3d& point, const Eigen::Vector2d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<OptimizePose, 2, 3, 4>(
                        new OptimizePose(K, point, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector2d observation_;
    const Eigen::Vector3d point_;
    const Eigen::Matrix3d& K_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

class OptimizePoseR
{
public:

    OptimizePoseR(const Eigen::Matrix3d& K, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Matrix3d& qc1c2, const Eigen::Vector3d& point, const Eigen::Vector2d& observation, const float weight)
        : K_(K), tc1c2_(tc1c2), qc1c2_(qc1c2), point_(point), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * point_ + p_frame;
    // Compute the map point pose in pixel frame.
        Eigen::Matrix<T,3,1> pointc1c2 = qc1c2_ * p_cp + tc1c2_;
    // Compute the map point pose in pixel frame.
        Eigen::Matrix<T, 3, 1> projected = K_ * pointc1c2;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - projected[0] / projected[2];
        residuals[1] =
            observation_.template cast<T>()[1] - projected[1] / projected[2];


        // residuals[0] = T(weight_) * residuals[0];
        // residuals[1] = T(weight_) * residuals[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Matrix3d& qc1c2, const Eigen::Vector3d& point, const Eigen::Vector2d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<OptimizePoseR, 2, 3, 4>(
                        new OptimizePoseR(K, tc1c2, qc1c2, point, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector2d observation_;
    const Eigen::Vector3d point_;
    const Eigen::Matrix3d& K_;
    const Eigen::Matrix3d& qc1c2_;
    const Eigen::Matrix<double,3,1>& tc1c2_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

class OptimizePoseICP
{
public:

    OptimizePoseICP(const Eigen::Matrix3d& K, const Eigen::Vector3d& point, const Eigen::Vector3d& observation, const float weight)
        : K_(K), point_(point), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * point_.template cast<T>() + p_frame;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - p_cp[0];
        residuals[1] =
            observation_.template cast<T>()[1] - p_cp[1];

        residuals[2] = observation_.template cast<T>()[2] - p_cp[2];


        residuals[0] = T(weight_) * residuals[0];
        residuals[1] = T(weight_) * residuals[1];
        residuals[2] = T(weight_) * residuals[2];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Vector3d& point, const Eigen::Vector3d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<OptimizePoseICP, 3, 3, 4>(
                        new OptimizePoseICP(K, point, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector3d observation_;
    const Eigen::Vector3d point_;
    const Eigen::Matrix3d& K_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

class MultiViewTriang
{
public:

    MultiViewTriang(const Eigen::Matrix4d& camPose, const Eigen::Matrix<double, 3, 4>& proj, const Eigen::Vector2d& observation)
        : camPose_(camPose), proj_(proj), observation_(observation)
    {
    }

    template<typename T>
    bool operator()(const T* const point, T* residuals_ptr) const
    {

        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        // Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> point3d(point);
        T p[4];
        p[0] = point[0];
        p[1] = point[1];
        p[2] = point[2];
        p[3] = T(1);
        // Logging("point", point[0], 3);
        // Logging("point1", point[1], 3);
        // Logging("point2", point[2], 3);
        Eigen::Map<const Eigen::Matrix<T,4,1>> point4d(p);
        const Eigen::Matrix<T,4,1> pointCam = camPose_ * point4d;
        // Logging("pointCam", pointCam, 3);
        // Logging("point4d", point4d, 3);
        const Eigen::Matrix<T, 3, 1> p_cp =
        proj_ * pointCam;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - p_cp[0] / p_cp[2];
        residuals[1] =
            observation_.template cast<T>()[1] - p_cp[1] / p_cp[2];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix4d& camPose, const Eigen::Matrix<double, 3, 4>& proj, const Eigen::Vector2d& observation)
    {
        return (new ceres::AutoDiffCostFunction<MultiViewTriang, 2, 3>(
                        new MultiViewTriang(camPose, proj, observation)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector2d observation_;
    const Eigen::Matrix<double,3,4> proj_;
    // const Eigen::Quaterniond Rot_;
    // const Eigen::Matrix<double,3,1> Tra_;
    const Eigen::Matrix4d& camPose_;

};

class LocalBundleAdjustment
{
public:

    LocalBundleAdjustment(const Eigen::Matrix3d& K, const Eigen::Vector2d& observation, const float weight)
        : K_(K), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const point, const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_point(point);

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * p_point + p_frame;
    // Compute the map point pose in pixel frame.
        Eigen::Matrix<T, 3, 1> projected = K_ * p_cp;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - projected[0] / projected[2];
        residuals[1] =
            observation_.template cast<T>()[1] - projected[1] / projected[2];


        // residuals[0] = T(weight_) * residuals[0];
        // residuals[1] = T(weight_) * residuals[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Vector2d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<LocalBundleAdjustment, 2, 3, 3, 4>(
                        new LocalBundleAdjustment(K, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector2d observation_;
    const Eigen::Matrix3d& K_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

class LocalBundleAdjustmentB
{
public:

    LocalBundleAdjustmentB(const Eigen::Matrix3d& K, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Matrix3d& qc1c2, const Eigen::Vector2d& observation, const float weight)
        : K_(K), tc1c2_(tc1c2), qc1c2_(qc1c2), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const point, const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_point(point);

        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frameB(tc1c2_);
        // Eigen::Map<const Eigen::Quaternion<T>> q_frameB(qc1c2_);

        // Eigen::Matrix<T,3,3> Rotc1c2 = c1c2_.block<3,3>(0,0);

        // Eigen::Quaternion<T> qc1c2(Rotc1c2);

        // qc1c2 = qc1c2* q_frame;
        // Rotc1c2 = Rotc1c2 *

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * p_point + p_frame;
        Eigen::Matrix<T,3,1> pointc1c2 = qc1c2_ * p_cp + tc1c2_;
    // Compute the map point pose in pixel frame.
        Eigen::Matrix<T, 3, 1> projected = K_ * pointc1c2;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - projected[0] / projected[2];
        residuals[1] =
            observation_.template cast<T>()[1] - projected[1] / projected[2];


        // residuals[0] = T(weight_) * residuals[0];
        // residuals[1] = T(weight_) * residuals[1];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Matrix3d& qc1c2, const Eigen::Vector2d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<LocalBundleAdjustmentB, 2, 3, 3, 4>(
                        new LocalBundleAdjustmentB(K, tc1c2, qc1c2, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector2d observation_;
    const Eigen::Matrix3d& K_;
    const Eigen::Matrix<double,3,1>& tc1c2_;
    const Eigen::Matrix3d& qc1c2_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

class LocalBundleAdjustmentICP
{
public:

    LocalBundleAdjustmentICP(const Eigen::Matrix3d& K, const Eigen::Vector3d& observation, const float weight)
        : K_(K), observation_(observation), weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const point, const T* const cameraT, const T* const cameraR, T* residuals_ptr) const
    {
        // T pt1[3];
        // pt1[0] = T(point.x);
        // pt1[1] = T(point.y);
        // pt1[2] = T(point.z);

        // T p[3];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
        Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_point(point);

        Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * p_point + p_frame;

        // Compute the residuals.
        // [ undistorted - projected ]
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals[0] =
            observation_.template cast<T>()[0] - p_cp[0];
        residuals[1] =
            observation_.template cast<T>()[1] - p_cp[1];

        residuals[2] = observation_.template cast<T>()[2] - p_cp[2];


        residuals[0] = T(weight_) * residuals[0];
        residuals[1] = T(weight_) * residuals[1];
        residuals[2] = T(weight_) * residuals[2];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Vector3d& observation, const float weight)
    {
        return (new ceres::AutoDiffCostFunction<LocalBundleAdjustmentICP, 2, 3, 3, 4>(
                        new LocalBundleAdjustmentICP(K, observation, weight)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    const Eigen::Vector3d observation_;
    const Eigen::Matrix3d& K_;
    const float weight_;
    double observed_x;
    double observed_y;
    // Camera intrinsics
    // double K[4] = {718.856, 718.856, 607.1928, 185.2157}; // fx,fy,cx,cy

};

// class MultiViewTriangProj
// {
// public:

//     MultiViewTriangProj(const Eigen::Matrix3d& K, const Eigen::Quaterniond& Rot, const Eigen::Matrix<double,3,1>& Tra, const Eigen::Vector2d& observation)
//         : K_(K), Rot_(Rot), Tra_(Tra), observation_(observation)
//     {
//     }

//     template<typename T>
//     bool operator()(const T* const point, T* residuals_ptr) const
//     {

//         // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_frame(cameraT);
//         // Eigen::Map<const Eigen::Quaternion<T>> q_frame(cameraR);
//         Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(point_);

//         Eigen::Matrix<T, 3, 1> p_cp =
//         Rot_.template cast<T>() * point + Tra_.template cast<T>();

//         Eigen::Matrix<T, 3, 1> projected = K_ * p_cp;

//         // Compute the residuals.
//         // [ undistorted - projected ]
//         Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
//         residuals[0] =
//             observation_.template cast<T>()[0] - projected[0] / projected[2];
//         residuals[1] =
//             observation_.template cast<T>()[1] - projected[1] / projected[2];

//         return true;
//     }

//     static ceres::CostFunction* Create(const Eigen::Matrix3d& K, const Eigen::Quaterniond& Rot, const Eigen::Matrix<double,3,1>& Tra, const Eigen::Vector2d& observation)
//     {
//         return (new ceres::AutoDiffCostFunction<MultiViewTriangProj, 2, 3>(
//                         new MultiViewTriangProj(K, Rot, Tra, observation)));
//     }
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// private:
//     const Eigen::Vector2d observation_;
//     const Eigen::Matrix<double,3,4> projMat_;
//     const Eigen::Quaterniond Rot_;
//     const Eigen::Matrix<double,3,1> Tra_;
//     const Eigen::Matrix3d& K_;

// };


} // namespace vio_slam

#endif