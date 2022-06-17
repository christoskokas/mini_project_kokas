#include "Viewer.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";

namespace vio_slam
{

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh) : m_it(*nh)
{
    nh->getParam("Camera_path",m_camera_path);
    m_image_sub = m_it.subscribe(m_camera_path, 1, &FeatureDrawer::imageCallback, this);
    m_image_pub = m_it.advertise("/image_converter/output_video", 1);
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

void FeatureDrawer::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    m_image_pub.publish(cv_ptr->toImageMsg());
}

} //namespace vio_slam

