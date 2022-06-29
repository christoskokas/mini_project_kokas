#include "FeatureDrawer.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
# include "opencv2/opencv_modules.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/features2d.hpp"

static const std::string OPENCV_WINDOW = "Features Detected";

namespace vio_slam
{

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, FeatureStrategy& featureMatchStrat) : m_it(*nh)
{
    nh->getParam("Camera_l_path", mLeftCameraPath);
    nh->getParam("Camera_r_path", mRightCameraPath);
    std::cout << "Feature Matching Strategy Option : ";
    mFeatureMatchStrat = featureMatchStrat;
    switch (featureMatchStrat)
    {
    case FeatureStrategy::orb :
      std::cout << "[ORB]";
      break;
    case FeatureStrategy::fast :
      std::cout << "[FAST]";
      break;
    case FeatureStrategy::brisk :
      std::cout << "[BRISK], It is not recommended to use BRISK because it is too slow for real time applications";
      break;
    default:
      break;
    }
    std::cout << '\n';
    mLeftImageSub = m_it.subscribe(mLeftCameraPath, 1, &FeatureDrawer::leftImageCallback, this);
    mRightImageSub = m_it.subscribe(mRightCameraPath, 1, &FeatureDrawer::rightImageCallback, this);
    mLeftImagePub = m_it.advertise("/left_camera/features", 1);
    mRightImagePub = m_it.advertise("/right_camera/features", 1);
    // message_filters::Subscriber<sensor_msgs::Image> leftIm(*nh,"/left_camera/features",1);
    // message_filters::Subscriber<sensor_msgs::Image> rightIm(*nh,"/right_camera/features",1);
    // message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(leftIm, rightIm, 10);
    // sync.registerCallback(boost::bind(&FeatureDrawer::FeatureDetectionCallback, _1, _2));
}

void FeatureDrawer::FeatureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{

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
    
    if (mFeatureMatchStrat == FeatureStrategy::orb)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), leftKeypoints, leftDescript);
      cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {255, 0, 0, 255} );
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      leftImage = outImage;
      mLeftImagePub.publish(out_msg.toImageMsg());
    }
    if (mFeatureMatchStrat == FeatureStrategy::fast)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detect( cv_ptr->image, leftKeypoints);
      cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {0, 255, 0, 255} );
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      leftImage = outImage;
      mLeftImagePub.publish(out_msg.toImageMsg());
    }
    if (mFeatureMatchStrat == FeatureStrategy::brisk)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), leftKeypoints, leftDescript);
      cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {0, 0, 255, 255} );
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      leftImage = outImage;
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
    if (mFeatureMatchStrat == FeatureStrategy::orb)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), rightKeypoints, rightDescript);
      cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {255, 0, 0, 255} );
      cv_bridge::CvImage out_msg;
      // rightKeypoints[0].size;
      // rightKeypoints[0].pt.x;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      rightImage = outImage;
      mRightImagePub.publish(out_msg.toImageMsg());
    }
    if (mFeatureMatchStrat == FeatureStrategy::fast)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detect( cv_ptr->image, rightKeypoints);
      cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {0, 255, 0, 255} );
      cv_bridge::CvImage out_msg;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      rightImage = outImage;
      mRightImagePub.publish(out_msg.toImageMsg());
    }
    if (mFeatureMatchStrat == FeatureStrategy::brisk)
    {
      cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
      // detect features and descriptor
      cv::Mat outImage;
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), rightKeypoints, rightDescript);
      cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {0, 0, 255, 255} );
      cv_bridge::CvImage out_msg;
      // rightKeypoints[0].size;
      // rightKeypoints[0].pt.x;
      out_msg.header   = msg->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      rightImage = outImage;
      mRightImagePub.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::addFeatures()
{

}

} //namespace vio_slam

