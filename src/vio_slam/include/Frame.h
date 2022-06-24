#pragma once

#ifndef FRAME_H
#define FRAME_H

#include "pangolin/pangolin.h"
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/scene/renderable.h>
#include <pangolin/scene/interactive_index.h>
#include <pangolin/gl/opengl_render_state.h>
#include <pangolin/gl/viewport.h>
#include <pangolin/gl/gldraw.h>
#include <Eigen/Geometry>
#include <pangolin/scene/tree.h>
#include <pangolin/geometry/glgeometry.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/foreach.hpp>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

namespace vio_slam
{
/**
 * @brief Contain The Transformation Matrix and the PointCloud for each KeyFrame.
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
        void pangoQuit(ros::NodeHandle *nh);                    
        std::list< KeyFrameVars > keyFrames;
        void printList(std::list< KeyFrameVars > keyFrames);


};

struct Lines : public pangolin::Renderable
{
    pangolin::GLprecision m[6];
    void Render(const pangolin::RenderParams& params) override;
    void getValues(std::vector < pangolin::GLprecision > mKeyFrame, pangolin::GLprecision mCamera[16]);
};

struct CameraFrame : public pangolin::Interactive, public pangolin::Renderable
{
    std::string mGroundTruthPath, mPointCloudPath;
    const char *color;
    bool buttonPressed;
    std::vector < pcl::PointXYZ > mPointCloud;
    ros::Subscriber groundSub;
    ros::Subscriber pointSub;
    CameraFrame()
    {
    };
    void Render(const pangolin::RenderParams&) override;
    bool Mouse(
        int button,
        const pangolin::GLprecision /*win*/[3], const pangolin::GLprecision /*obj*/[3], const pangolin::GLprecision /*normal*/[3],
        bool /*pressed*/, int button_state, int pickId
    ) override;
    virtual bool MouseMotion(
        const pangolin::GLprecision /*win*/[3], const pangolin::GLprecision /*obj*/[3], const pangolin::GLprecision /*normal*/[3],
        int /*button_state*/, int /*pickId*/
    ) override;
    void Subscribers(ros::NodeHandle *nh);
    void groundCallback(const nav_msgs::Odometry& msg);
    void pointCallback(const PointCloud::ConstPtr& msg);
    void lineFromKeyFrameToCamera(std::vector < pangolin::GLprecision > mT);
    
};


} //namespace vio_slam

#endif // FRAME_H