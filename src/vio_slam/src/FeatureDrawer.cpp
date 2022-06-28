#include "FeatureDrawer.h"

static const std::string OPENCV_WINDOW = "Features Detected";

namespace vio_slam
{

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, std::string* featureMatchStrat) : m_it(*nh)
{
    nh->getParam("Camera_l_path", mLeftCameraPath);
    nh->getParam("Camera_r_path", mRightCameraPath);
    mFeatureMatchStrat = *featureMatchStrat;
    ROS_INFO("Feature Matching Strategy Option : [%s]", mFeatureMatchStrat.c_str());
    mLeftImageSub = m_it.subscribe(mLeftCameraPath, 1, &FeatureDrawer::leftImageCallback, this);
    mRightImageSub = m_it.subscribe(mRightCameraPath, 1, &FeatureDrawer::rightImageCallback, this);
    mLeftImagePub = m_it.advertise("/left_camera/features", 1);
    mRightImagePub = m_it.advertise("/right_camera/features", 1);
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

void FeatureDrawer::leftImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if (mFeatureMatchStrat == "ORB")
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
      // detect features and descriptor
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors1;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), keypoints, descriptors1 );
      cv::drawKeypoints(cv_ptr->image, keypoints, descriptors1, {255, 0, 0, 255} );
      leftImage = msg->data;
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = descriptors1; // Your cv::Mat
      mLeftImagePub.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::rightImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if (mFeatureMatchStrat == "ORB")
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
      // detect features and descriptor
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors1;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), keypoints, descriptors1 );
      cv::drawKeypoints(cv_ptr->image, keypoints, descriptors1, {255, 0, 0, 255} );
      rightImage = msg->data;
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = descriptors1; // Your cv::Mat
      mRightImagePub.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::addFeatures()
{

}

} //namespace vio_slam

