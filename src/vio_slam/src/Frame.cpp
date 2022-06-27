#include "Frame.h"
#include <tf/tf.h>


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

        // THIS IS FOR THE FEATURES NOT POINTCLOUDS

        // std::cout << " All Pointclouds : [ ";
        // std::vector <std::vector < pcl::PointXYZ> > curvect = vect.pointCloud;

        // for (auto element : curvect)
        // {
        //     std::vector < pcl::PointXYZ> curvect2 = element;
        //     std::cout << " each Pointclouds : [ ";
        //     for (auto elementxyz : curvect2)
        //     {
        //         std::cout << "( "  <<elementxyz.x << ' ' << elementxyz.y << ' ' << elementxyz.z << ")";
        //     }
        //     std::cout << "]";
        //     std::cout << '\n';

        // }
        // std::cout << ']';
        // std::cout << '\n';
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

    pangolin::Renderable renders;
    renders.Add(std::make_shared<pangolin::Axis>());
    auto camera = std::make_shared<CameraFrame>();
    camera->color = "G";
    camera->Subscribers(nh);
    renders.Add(camera);
    KeyFrameVars temp;
    for (size_t i = 0; i < 16; i++)
    {
        temp.mT.push_back(camera->T_pc.m[i]);
    }
    keyFrames.push_back(temp);
    
    // Create Interactive View in window
    pangolin::SceneHandler handler(renders, s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    d_cam.SetDrawFunction([&](pangolin::View& view){
        view.Activate(s_cam);
        renders.Render();
    });
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> a_button("ui.Button", false, false);
    while( ros::ok() && !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        auto lines = std::make_shared<Lines>();    
        lines->getValues(temp.mT,camera->T_pc.m);
        if (pangolin::Pushed(a_button))
        {
            ROS_INFO("Keyframe Added \n");
            {
                auto keyframe = std::make_shared<CameraFrame>();
                keyframe->T_pc = camera->T_pc;
                // keyframe->mPointCloud = camera->mPointCloud;
                keyframe->color = "B";
                renders.Add(keyframe);
            }
            renders.Add(lines);
            temp.clear();
            for (size_t i = 0; i < 16; i++)
            {
                temp.mT.push_back(camera->T_pc.m[i]);
            }
            // temp.pointCloud.push_back(keyframe->mPointCloud);
            keyFrames.push_back(temp);
            
            
            d_cam.SetDrawFunction([&](pangolin::View& view){
                view.Activate(s_cam);
                renders.Render();
                camera->lineFromKeyFrameToCamera(temp.mT);
            });
            // printList(keyFrames);
            
        }
        
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

void CameraFrame::Subscribers(ros::NodeHandle *nh)
{
    nh->getParam("ground_truth_path", mGroundTruthPath);
    nh->getParam("pointcloud_path", mPointCloudPath);
    groundSub = nh->subscribe(mGroundTruthPath, 1, &CameraFrame::groundCallback, this);
    pointSub = nh->subscribe<PointCloud>(mPointCloudPath, 1, &CameraFrame::pointCallback, this);

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
}

void CameraFrame::pointCallback(const PointCloud::ConstPtr& msg)
{
    BOOST_FOREACH (const pcl::PointXYZ& pt, msg->points)
    if (!(pt.x != pt.x || pt.y != pt.y || pt.z != pt.z))
    {
        mPointCloud.push_back(pt);
    }
}

void CameraFrame::Render(const pangolin::RenderParams&)
{

    const float w = 1;
    const float h = w*0.75;
    const float z = w*0.3;

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
    glVertex3f(w/2,h/2,0);
    glVertex3f(w,h,z);
    glVertex3f(w/2,-h/2,0);
    glVertex3f(w,-h,z);
    glVertex3f(-w/2,-h/2,0);
    glVertex3f(-w,-h,z);
    glVertex3f(-w/2,h/2,0);
    glVertex3f(-w,h,z);

    glVertex3f(w/2,h/2,0);
    glVertex3f(w/2,-h/2,0);

    glVertex3f(-w/2,h/2,0);
    glVertex3f(-w/2,-h/2,0);

    glVertex3f(-w/2,h/2,0);
    glVertex3f(w/2,h/2,0);

    glVertex3f(-w/2,-h/2,0);
    glVertex3f(w/2,-h/2,0);

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

void Lines::getValues(std::vector < pangolin::GLprecision > mKeyFrame, pangolin::GLprecision mCamera[16])
{

    m[0] = mKeyFrame[12];
    m[1] = mKeyFrame[13];
    m[2] = mKeyFrame[14];
    m[3] = mCamera[12];
    m[4] = mCamera[13];
    m[5] = mCamera[14];
}

void Lines::Render(const pangolin::RenderParams& params)
{
    glPushMatrix();
    glColor3f(1.0f,0.0f,0.0f);
    glLineWidth(1);
    glBegin(GL_LINES);
    glVertex3f(m[0],m[1],m[2]);
    glVertex3f(m[3],m[4],m[5]);
    glEnd();

    glPopMatrix();
}

void CameraFrame::lineFromKeyFrameToCamera(std::vector < pangolin::GLprecision > mT)
{
    glPushMatrix();
    glColor3f(1.0f,0.0f,0.0f);
    glLineWidth(1);
    glBegin(GL_LINES);
    glVertex3f(mT[12],mT[13],mT[14]);
    glVertex3f(T_pc.m[12], T_pc.m[13], T_pc.m[14]);
    glEnd();
    glPopMatrix();
}


}