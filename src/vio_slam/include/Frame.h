#pragma once

#ifndef FRAME_H
#define FRAME_H

#include "Settings.h"
#include "Camera.h"
#include "Map.h"
#include "pangolin/pangolin.h"
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/scene/renderable.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/geometry/glgeometry.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/foreach.hpp>
#include <tf/tf.h>
#include <Eigen/Core>

// typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

namespace vio_slam
{
/**
 * @brief Contains The Transformation Matrix and the PointCloud for each KeyFrame.
 */
struct KeyFrameVars
{
    std::vector < pangolin::GLprecision > mT;                //Column Major
        // pointcloud;                          //TO BE DONE
    std::vector <std::vector < pcl::PointXYZ> > pointCloud;
    // std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
    void clear()
    {
        mT.clear();
        pointCloud.clear();
    }
};

class Frame
{
    private:

    public:
        Frame();
        std::list< KeyFrameVars > keyFrames;
        void pangoQuit(Zed_Camera* zedPtr, const Map* _map);                    


};

struct Lines : public pangolin::Renderable
{
    pangolin::GLprecision m[6];
    void Render(const pangolin::RenderParams& params) override;
    void getValues(std::vector < pangolin::GLprecision >& mKeyFrame, pangolin::GLprecision mCamera[16]);
};

struct Points : public pangolin::Renderable
{
    const std::vector<pcl::PointXYZ>* points;
    Points(const std::vector<pcl::PointXYZ>* point);
    void Render(const pangolin::RenderParams& params) override;
};

struct CameraFrame : public pangolin::Renderable
{
    std::string mGroundTruthPath, mPointCloudPath;
    const char *color;

    int mapPntsDrawn {0};

    const float cameraWidth = 0.0575f;

    Zed_Camera* zedCamera;

    const Map* map;

    std::vector < pcl::PointXYZ > mPointCloud;
    ros::Subscriber groundSub;
    Eigen::Matrix4f Trial = Eigen::Matrix4f::Identity();
    // ros::Subscriber pointSub;
    // void Subscribers(ros::NodeHandle *nh);
    // void groundCallback(const nav_msgs::Odometry& msg);
    // void pointCallback(const PointCloud::ConstPtr& msg);
    void lineFromKeyFrameToCamera();
    void Render(const pangolin::RenderParams&) override;
    void drawCamera();
    void drawPoints();
    void drawKeyFrames();
    void getOpenGLMatrix(pangolin::OpenGlMatrix &MOw);
};


} //namespace vio_slam

#endif // FRAME_H