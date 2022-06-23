#include "Frame.h"
#include "pangolin/pangolin.h"
#include <ros/ros.h>
#include <tf/tf.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <vector>

namespace vio_slam
{
Frame::Frame()
{
    
}

void Frame::printList(std::list< KeyFrameVars > keyFrames)
{
    


    for (auto vect : keyFrames)
    {
        {
        std::cout << "OpenGLMatrix : [ ";
        std::vector < pangolin::GLprecision > curvect = vect.mT;

        for (auto element : curvect)
        {
            std::cout << element << ' ';
        }
        std::cout << ']';
        std::cout << '\n';
        }
        std::cout << "ints : [ ";
        std::vector < int > curvect = vect.trial;

        for (auto element : curvect)
        {
            std::cout << element << ' ';
        }
        std::cout << ']';
        std::cout << '\n';
    }
}

void Frame::pangoQuit(ros::NodeHandle *nh)
{
    const int UI_WIDTH = 180;
    
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);

        // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    pangolin::Renderable tree;
    auto axis_i = std::make_shared<pangolin::Axis>();
    tree.Add(axis_i);
    // std::shared_ptr<CameraFrame> camera(new CameraFrame());
    auto camera = std::make_shared<CameraFrame>();
    camera->color = "G";
    tree.Add(camera);
    camera->groundSubscriber(nh);
    // std::vector<pangolin::OpenGlMatrix> lel;
    // lel.at(0) = camera->T_pc;
    KeyFrameVars temp;
    for (size_t i = 0; i < 16; i++)
    {
        temp.trial.push_back(i);
        temp.mT.push_back(camera->T_pc.m[i]);
    }
    keyFrames.push_back(temp);
    
    // Create Interactive View in window
    pangolin::SceneHandler handler(tree, s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    d_cam.SetDrawFunction([&](pangolin::View& view){
        view.Activate(s_cam);
        tree.Render();
    });
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> a_button("ui.Button", false, false);
    while( ros::ok() && !pangolin::ShouldQuit() )
    {
        
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        if (pangolin::Pushed(a_button))
        {
            ROS_INFO("Keyframe Added \n");
            auto keyframe = std::make_shared<CameraFrame>();
            keyframe->T_pc = camera->T_pc;
            keyframe->color = "B";
            tree.Add(keyframe);
            temp.mT.clear();
            temp.trial.clear();
            for (size_t i = 0; i < 16; i++)
            {
                temp.trial.push_back(i+1);
                temp.mT.push_back(keyframe->T_pc.m[i]);
            }
            keyFrames.push_back(temp);
            
            // pangolin::SceneHandler handler(tree, s_cam);
            // pangolin::View& d_cam = pangolin::CreateDisplay()
            //         .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            //         .SetHandler(&handler);

            d_cam.SetDrawFunction([&](pangolin::View& view){
                view.Activate(s_cam);
                tree.Render();
            });
            printList(keyFrames);
            
        }
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

void CameraFrame::groundSubscriber(ros::NodeHandle *nh)
{
    nh->getParam("ground_truth_path", mGroundTruth);
    groundSub = nh->subscribe(mGroundTruth, 1, &CameraFrame::groundCallback, this);

}

void CameraFrame::groundCallback(const nav_msgs::Odometry& msg)
{
    // T_pc.m is column major
    tf::Quaternion quatRot;
    quatRot.setW(msg.pose.pose.orientation.w);
    quatRot.setX(msg.pose.pose.orientation.y);              // Y on Gazebo is X on Pangolin
    quatRot.setY(msg.pose.pose.orientation.z);              // Z on Gazebo is Y on Pangolin
    quatRot.setZ(msg.pose.pose.orientation.x);              // X on Gazebo is Z on Pangolin
    tf::Matrix3x3 rotMat(quatRot);
    for (size_t i = 0; i < 3; i++)
    {
        this->T_pc.m[4*i] = rotMat[0][i];
        this->T_pc.m[4*i+1] = rotMat[1][i];
        this->T_pc.m[4*i+2] = rotMat[2][i];
    }
    
    this->T_pc.m[12] = msg.pose.pose.position.y;            // Y on Gazebo is X on Pangolin
    this->T_pc.m[13] = msg.pose.pose.position.z;            // Z on Gazebo is Y on Pangolin
    this->T_pc.m[14] = msg.pose.pose.position.x;            // X on Gazebo is Z on Pangolin

    // TODO add pointcloud to pangolin, change camera shape, add transformation from base link to camera
}

void CameraFrame::Render(const pangolin::RenderParams&)
{

    const float &w = 1;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    if (color == "G")
    {
        glColor3f(0.0f,1.0f,0.0f);
    }
    if (color == "B")
    {
        glColor3f(0.0f,0.0f,1.0f);
    }
    glLineWidth(1);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

bool CameraFrame::Mouse(int button,
        const pangolin::GLprecision /*win*/[3], const pangolin::GLprecision /*obj*/[3], const pangolin::GLprecision /*normal*/[3],
        bool /*pressed*/, int button_state, int pickId
    )
    {
        PANGOLIN_UNUSED(button);
        PANGOLIN_UNUSED(button_state);
        PANGOLIN_UNUSED(pickId);
        return false;
    }

bool CameraFrame::MouseMotion(
        const pangolin::GLprecision /*win*/[3], const pangolin::GLprecision /*obj*/[3], const pangolin::GLprecision /*normal*/[3],
        int /*button_state*/, int /*pickId*/
    ) 
    {
        return false;
    }

}