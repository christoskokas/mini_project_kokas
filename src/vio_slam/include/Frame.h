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

namespace vio_slam
{

class Frame
{
    private:

    public:
        Frame();
        void pangoQuit(ros::NodeHandle *nh);
        
};

struct CameraFrame : public pangolin::Interactive, public pangolin::Renderable
{
    std::string mGroundTruth;
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
    void groundSubscriber(ros::NodeHandle *nh);
    void groundCallback(const nav_msgs::Odometry& msg);
    float axis_length;
    const pangolin::InteractiveIndex::Token label_x;
    const pangolin::InteractiveIndex::Token label_y;
    const pangolin::InteractiveIndex::Token label_z;
    ros::Subscriber groundSub;
};


} //namespace vio_slam

#endif // FRAME_H