#include "Frame.h"


namespace vio_slam
{

Frame::Frame()
{
    
}

void Frame::pangoQuit(Zed_Camera* zedPtr, const Map* _map)
{
    using namespace std::literals::chrono_literals;

    const int UI_WIDTH = 180;
    
    pangolin::CreateWindowAndBind("Main", 1024,768);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768,500.0,500.0,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-1,-2, 0,0,0, pangolin::AxisNegY)
    );

    pangolin::Renderable renders;
    renders.Add(std::make_shared<pangolin::Axis>());
    auto camera = std::make_shared<CameraFrame>();
    camera->color = "G";
    // camera->Subscribers(nh);
    camera->zedCamera = zedPtr;
    camera->map = _map;
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
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    camera->getOpenGLMatrix(Ow);
    camera->drawCamera();
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> a_button("ui.Button", false, false);
    while(!pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        auto lines = std::make_shared<Lines>();    
        lines->getValues(temp.mT,camera->T_pc.m);
        if (zedPtr->addKeyFrame)
        {
            {
                zedPtr->addKeyFrame = false;
                auto keyframe = std::make_shared<CameraFrame>();
                keyframe->T_pc = camera->T_pc;
                // keyframe->mPointCloud = camera->mPointCloud;
                keyframe->color = "B";
                // renders.Add(keyframe);
            }
            renders.Add(lines);
            temp.clear();
            for (size_t i = 0; i < 16; i++)
            {
                temp.mT.push_back(camera->T_pc.m[i]);
            }
            // temp.pointCloud.push_back(keyframe->mPointCloud);
            keyFrames.push_back(temp);
            
            d_cam.Activate(s_cam);
            d_cam.SetDrawFunction([&](pangolin::View& view){
                view.Activate(s_cam);
                renders.Render();
                camera->lineFromKeyFrameToCamera();
                camera->drawPoints();
            });

            
        }
        d_cam.Activate(s_cam);
        camera->getOpenGLMatrix(Ow);
        camera->drawCamera();
        camera->drawKeyFrames();
        s_cam.Follow(Ow);


        // Swap frames and Process Events
        pangolin::FinishFrame();

        std::this_thread::sleep_for(50ms);
    }
}

void CameraFrame::Render(const pangolin::RenderParams&)
{

    const float w = cameraWidth;
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

void CameraFrame::drawPoints()
{
    glColor3f(1.0f,1.0f,1.0f);
    // std::lock_guard<std::mutex> lock(map->mapMutex);
    glBegin(GL_POINTS);


    for ( size_t i {0}, end{map->mapPoints.size()}; i < end; i ++)
    {
        const MapPoint* mp = map->mapPoints.at(i);
        if ( !mp->GetIsOutlier() && !mp->getActive())
            glVertex3d(mp->wp3d(0),mp->wp3d(1),mp->wp3d(2));
    }

    glColor3f(0.0f,1.0f,0.0f);

    std::vector<MapPoint*>::const_iterator it, end(map->activeMapPoints.end());
    for ( it = map->activeMapPoints.begin(); it != end; it++)
    {
        if ( !(*it)->GetIsOutlier() && (*it)->getActive())
            glVertex3d((*it)->wp3d(0),(*it)->wp3d(1),(*it)->wp3d(2));
    }
    glEnd();

}

void CameraFrame::drawKeyFrames()
{

    const float w = cameraWidth;
    const float h = w*0.75;
    const float z = w*0.3;

    glColor3f(0.0f,0.0f,1.0f);
    std::unordered_map<unsigned long,KeyFrame*>::const_iterator it, end(map->keyFrames.end());
    for ( it = map->keyFrames.begin(); it != end; it ++)
    {
        if (!(*it).second->visualize)
            continue;
        glPushMatrix();
        Eigen::Matrix4d keyPose = (*it).second->getPose();
        glMultMatrixd((GLdouble*)keyPose.data());
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

}

void CameraFrame::getOpenGLMatrix(pangolin::OpenGlMatrix &MOw)
{
    for (int i = 0; i<4; i++) {
        MOw.m[4*i] = zedCamera->cameraPose.pose(4*i);
        MOw.m[4*i+1] = zedCamera->cameraPose.pose(4*i + 1);
        MOw.m[4*i+2] = zedCamera->cameraPose.pose(4*i + 2);
        MOw.m[4*i+3] = zedCamera->cameraPose.pose(4*i + 3);
    }
    // MOw.SetIdentity();
    // MOw.m[12] = T_pc(0,3);
    // MOw.m[13] = T_pc(1,3);
    // MOw.m[14] = T_pc(2,3);
}

void CameraFrame::drawCamera()
{

    const float w = cameraWidth;
    const float h = w*0.75;
    const float z = w*0.3;

    glPushMatrix();
    glColor3f(0.0f,1.0f,0.0f);


    glMultMatrixd((GLdouble*)zedCamera->cameraPose.pose.data());
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

void Lines::getValues(std::vector < pangolin::GLprecision >& mKeyFrame, pangolin::GLprecision mCamera[16])
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

void CameraFrame::lineFromKeyFrameToCamera()
{
    const int lastKeyFrameIdx {map->kIdx - 1};
    if (lastKeyFrameIdx < 0)
        return;
    glPushMatrix();
    glColor3f(1.0f,0.0f,0.0f);
    glLineWidth(1);
    std::unordered_map<unsigned long, KeyFrame*>::const_iterator it, end(map->keyFrames.end());
    glBegin(GL_LINES);
    for ( it = map->keyFrames.begin(); it != end; it ++)
    {
        for (const auto& key:(*it).second->connections)
        {
            if (!(*it).second->visualize)
                continue;
            glVertex3f((GLfloat)map->keyFrames.at(key)->pose.pose(0,3),(GLfloat)map->keyFrames.at(key)->pose.pose(1,3),(GLfloat)map->keyFrames.at(key)->pose.pose(2,3));
            glVertex3f((GLfloat)(*it).second->pose.pose(0,3), (GLfloat)(*it).second->pose.pose(1,3), (GLfloat)(*it).second->pose.pose(2,3));
        }
    }
    glEnd();
    Eigen::Matrix4d keyPose = map->keyFrames.at(lastKeyFrameIdx)->getPose();
    Eigen::Matrix4d camPose = zedCamera->cameraPose.pose;
    glBegin(GL_LINES);
    glVertex3f((GLfloat)keyPose(0,3),(GLfloat)keyPose(1,3),(GLfloat)keyPose(2,3));
    glVertex3f((GLfloat)camPose(0,3), (GLfloat)camPose(1,3), (GLfloat)camPose(2,3));
    glEnd();
    glPopMatrix();
}

Points::Points(const std::vector<pcl::PointXYZ>* point)
{
    if(!point->empty())
    {
        points = point;
    }
}

void Points::Render(const pangolin::RenderParams& params)
{

}


}