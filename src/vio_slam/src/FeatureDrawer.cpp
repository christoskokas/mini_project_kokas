#include "FeatureDrawer.h"

static const std::string OPENCV_WINDOW = "Features Detected";

namespace vio_slam
{

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, FeatureStrategy& featureMatchStrat) : m_it(*nh), img_sync(MySyncPolicy(10), leftIm, rightIm)
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
      std::cout << "[BRISK]. It is not recommended to use BRISK because it is too slow for real time applications";
      break;
    default:
      break;
    }
    std::cout << '\n';
    // mLeftImageSub = m_it.subscribe(mLeftCameraPath, 1, &FeatureDrawer::leftImageCallback, this);
    // mRightImageSub = m_it.subscribe(mRightCameraPath, 1, &FeatureDrawer::rightImageCallback, this);
    leftIm.subscribe(*nh, mLeftCameraPath, 1);
    rightIm.subscribe(*nh, mRightCameraPath, 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    // img_sync {MySyncPolicy(10), leftIm, rightIm};
    // img_sync.ApproximateTimeSynchronizer(MySyncPolicy(10), leftIm, rightIm);
    // img_sync.init();
    // img_sync.init(MySyncPolicy(10));
    img_sync.registerCallback(boost::bind(&FeatureDrawer::FeatureDetectionCallback, this, _1, _2));
    mLeftImagePub = m_it.advertise("/left_camera/features", 1);
    mRightImagePub = m_it.advertise("/right_camera/features", 1);
    mImageMatches = m_it.advertise("/camera/matches", 1);
}

// void FeatureDrawer::FeatureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
// {
//     if (mFeatureMatchStrat == FeatureStrategy::orb)
//     {
//       cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
//       std::vector< std::vector<cv::DMatch> > knn_matches;
//       matcher->knnMatch( leftDescript, rightDescript, knn_matches, 2 );
//     //-- Filter matches using the Lowe's ratio test
//     const float ratio_thresh = 0.7f;
//     std::vector<cv::DMatch> good_matches;
//     for (size_t i = 0; i < knn_matches.size(); i++)
//     {
//         if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//         {
//             good_matches.push_back(knn_matches[i][0]);
//         }
//     }
//     //-- Draw matches
//     cv::Mat img_matches;
//     drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
//                  Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//     }
// }

void FeatureDrawer::findFeatures(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm, cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, image_transport::Publisher publish)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(lIm, sensor_msgs::image_encodings::RGB8);
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
      detector->detectAndCompute( cv_ptr->image, cv::Mat(), keypoints, descriptor);
      cv::drawKeypoints(cv_ptr->image, keypoints, outImage, {255, 0, 0, 255} );
      cv_bridge::CvImage out_msg;
      out_msg.header   = lIm->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = outImage; // Your cv::Mat
      image = cv_ptr->image;
      publish.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::findMatches(const sensor_msgs::ImageConstPtr& lIm)
{
    if (mFeatureMatchStrat == FeatureStrategy::orb)
    {
      if ( leftDescript.empty() )
        cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
      if ( rightDescript.empty() )
        cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

      std::vector<cv::DMatch> matches;
    	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true);  
      matcher.match(leftDescript, rightDescript, matches);
      for (int i = 0; i < matches.size(); i++)
      {
        for (int j = 0; j < matches.size() - 1; j++)
        {
          if (matches[j].distance > matches[j + 1].distance)
          {
            auto temp = matches[j];
            matches[j] = matches[j + 1];
            matches[j + 1] = temp;
          }
        }
      }
      if (matches.size() > 100)
      {
        matches.resize(100);
      }
      cv::Mat img_matches;
      drawMatches( leftImage, leftKeypoints, rightImage, rightKeypoints, matches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


      cv_bridge::CvImage out_msg;
      out_msg.header   = lIm->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = img_matches; // Your cv::Mat
      mImageMatches.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::FeatureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
    findFeatures(lIm, rIm, leftImage, leftKeypoints, leftDescript, mLeftImagePub);
    findFeatures(lIm, rIm, rightImage, rightKeypoints, rightDescript, mRightImagePub);
    findMatches(lIm);
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
      leftImage = cv_ptr->image;
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
      leftImage = cv_ptr->image;
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
      leftImage = cv_ptr->image;
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
      rightImage = cv_ptr->image;
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
      rightImage = cv_ptr->image;
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
      rightImage = cv_ptr->image;
      mRightImagePub.publish(out_msg.toImageMsg());
    }
}

void FeatureDrawer::addFeatures()
{

}

} //namespace vio_slam

// FLANN MATCHER

// cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
// std::vector< std::vector<cv::DMatch> > knn_matches;
// leftDescript.convertTo(leftDescript, CV_32F); 
// rightDescript.convertTo(rightDescript, CV_32F);
// matcher->knnMatch( leftDescript, rightDescript, knn_matches, 2 );
// //-- Filter matches using the Lowe's ratio test
// const float ratio_thresh = 0.7f;
// std::vector<cv::DMatch> good_matches;

// for (size_t i = 0; i < knn_matches.size(); i++)
// {
//     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//     {
//         good_matches.push_back(knn_matches[i][0]);
//     }
// }
// //-- Draw matchleftImagees
// cv::Mat img_matches;
// drawMatches( leftImage, leftKeypoints, rightImage, rightKeypoints, good_matches, img_matches, cv::Scalar::all(-1),
//             cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );