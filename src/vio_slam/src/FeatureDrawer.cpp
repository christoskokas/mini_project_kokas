#include "FeatureDrawer.h"


static const std::string OPENCV_WINDOW = "Features Detected";

namespace vio_slam
{

void FeatureDrawer::setUndistortMap(ros::NodeHandle *nh)
{
    std::cout << '\n'; 
    cv::Size imgSize = cv::Size(zedcamera->mWidth, zedcamera->mHeight);
    cv::stereoRectify(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, imgSize, zedcamera->sensorsRotate, zedcamera->sensorsTranslate, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, R1, P1, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, R2, P2, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    
}

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, FeatureStrategy& featureMatchStrat, const Zed_Camera* zedptr) : m_it(*nh), img_sync(MySyncPolicy(10), leftIm, rightIm)
{
    this->zedcamera = zedptr;
    if (!zedcamera->rectified)
    {
      setUndistortMap(nh);
    }
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
    leftIm.subscribe(*nh, zedcamera->cameraLeft.path, 1);
    rightIm.subscribe(*nh, zedcamera->cameraRight.path, 1);
    img_sync.registerCallback(boost::bind(&FeatureDrawer::featureDetectionCallback, this, _1, _2));
    mImageMatches = m_it.advertise("/camera/matches", 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    // img_sync {MySyncPolicy(10), leftIm, rightIm};
    // img_sync.ApproximateTimeSynchronizer(MySyncPolicy(10), leftIm, rightIm);
    // img_sync.init();
    // img_sync.init(MySyncPolicy(10));
    // mLeftImagePub = m_it.advertise("/left_camera/features", 1);
    // mRightImagePub = m_it.advertise("/right_camera/features", 1);
}



void FeatureDrawer::setImage(const sensor_msgs::ImageConstPtr& imageRef, cv::Mat& image)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(imageRef, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    image = cv_ptr->image;
}



void Features::findFeatures()
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // detect features and descriptor
    cv::Mat outImage;
    detector->detectAndCompute( image, cv::Mat(), keypoints, descriptors);
    cv::drawKeypoints(image, keypoints, outImage, {255, 0, 0, 255} );
}

void Features::getPoints()
{
    std::cout << "x : " << pointsPosition.at(0).x << " y : " << pointsPosition.at(0).y << " z : " << pointsPosition.at(0).z << '\n'; 
}

std::vector<cv::DMatch> Features::findMatches(const Features& secondImage, const std_msgs::Header& header, image_transport::Publisher& mImageMatches)
{
    if ( descriptors.empty() )
      cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
    if ( secondImage.descriptors.empty() )
      cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true);  
    matcher.match(descriptors, secondImage.descriptors, matches);
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
    drawMatches( image, keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    cv_bridge::CvImage out_msg;
    out_msg.header   = header; // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
    out_msg.image    = img_matches; // Your cv::Mat
    mImageMatches.publish(out_msg.toImageMsg());
    return matches;
}

void FeatureDrawer::calculateFeaturePosition(const std::vector<cv::DMatch>& matches)
{
    for (size_t i = 0; i < matches.size(); i++)
    {

      double x = zedcamera->mBaseline*(leftImage.keypoints[matches[i].queryIdx].pt.x - zedcamera->cameraLeft.cx)/(leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x);
      double y = zedcamera->mBaseline * zedcamera->cameraLeft.fx * (leftImage.keypoints[matches[i].queryIdx].pt.y - zedcamera->cameraLeft.cy)/(zedcamera->cameraLeft.fy * (leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x));
      double z = zedcamera->mBaseline*zedcamera->cameraLeft.fx/(leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x);
      // std::cout << " x : " << x << " y : " << y << " z : " << z << '\n'; 
      leftImage.pointsPosition.push_back(pcl::PointXYZ(x,y,z));
      x = zedcamera->mBaseline*(leftImage.keypoints[matches[i].queryIdx].pt.x - zedcamera->cameraRight.cx)/(leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x);
      y = zedcamera->mBaseline * zedcamera->cameraRight.fx * (leftImage.keypoints[matches[i].queryIdx].pt.y - zedcamera->cameraRight.cy)/(zedcamera->cameraRight.fy * (leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x));
      z = zedcamera->mBaseline*zedcamera->cameraRight.fx/(leftImage.keypoints[matches[i].queryIdx].pt.x - rightImage.keypoints[matches[i].trainIdx].pt.x);
      // std::cout << " x : " << x << " y : " << y << " z : " << z << '\n' << "XDD " << '\n'; 
      rightImage.pointsPosition.push_back(pcl::PointXYZ(x,y,z));
    }
    rightImage.getPoints();
  
}

void FeatureDrawer::featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
    leftImage.pointsPosition.clear();
    rightImage.pointsPosition.clear();
    setImage(lIm, leftImage.image);
    setImage(rIm, rightImage.image);
    cv::Mat dstle, dstri;
    if (!zedcamera->rectified)
    {
      cv::remap(leftImage.image, dstle, rmap[0][0], rmap[0][1],cv::INTER_LINEAR);
      cv::remap(rightImage.image, dstri, rmap[1][0], rmap[1][1],cv::INTER_LINEAR);
      cv::hconcat(leftImage.image, dstle, dstle);                       //add 2 images horizontally (image1, image2, destination)
      cv::hconcat(rightImage.image, dstri, dstri);                      //add 2 images horizontally (image1, image2, destination)
      cv::vconcat(dstle, dstri, dstle);                           //add 2 images vertically
      cv_bridge::CvImage out_msg;
      out_msg.header   = lIm->header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = dstle; // Your cv::Mat
      // mImageMatches.publish(out_msg.toImageMsg());
      // std::cout << "NOT RECTIFIED" << '\n';
    }
    leftImage.findFeatures();
    rightImage.findFeatures();
    std::vector<cv::DMatch> matches = leftImage.findMatches(rightImage, lIm->header, mImageMatches);
    calculateFeaturePosition(matches);
    if (!firstImage)
    {

    }
    leftImage.previousimage = leftImage.image;
    rightImage.previousimage = rightImage.image;
    
    // std::cout << "POINT X query : " << leftImage.keypoints[matches[0].queryIdx].pt.x << '\n';
    // std::cout << "POINT X train : " << rightImage.keypoints[matches[0].trainIdx].pt.x << '\n';
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
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

// void FeatureDrawer::findFeatures(const cv::Mat& imageRef, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, image_transport::Publisher publish)
// {

//     if (mFeatureMatchStrat == FeatureStrategy::orb)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detectAndCompute( imageRef, cv::Mat(), keypoints, descriptors);
//       cv::drawKeypoints(imageRef, keypoints, outImage, {255, 0, 0, 255} );
//       // cv_bridge::CvImage out_msg;
//       // out_msg.header   = imageRef->header; // Same timestamp and tf frame as input image
//       // out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       // out_msg.image    = outImage; // Your cv::Mat
//       // publish.publish(out_msg.toImageMsg());
//     }
// }

// std::vector<cv::DMatch> FeatureDrawer::findMatches(const std_msgs::Header& header)
// {
//     if (mFeatureMatchStrat == FeatureStrategy::orb)
//     {
//       if ( leftImage.descriptors.empty() )
//         cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
//       if ( rightImage.descriptors.empty() )
//         cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

//       std::vector<cv::DMatch> matches;
//     	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true);  
//       matcher.match(leftImage.descriptors, rightImage.descriptors, matches);
//       for (int i = 0; i < matches.size(); i++)
//       {
//         for (int j = 0; j < matches.size() - 1; j++)
//         {
//           if (matches[j].distance > matches[j + 1].distance)
//           {
//             auto temp = matches[j];
//             matches[j] = matches[j + 1];
//             matches[j + 1] = temp;
//           }
//         }
//       }
//       if (matches.size() > 100)
//       {
//         matches.resize(100);
//       }
//       cv::Mat img_matches;
//       drawMatches( leftImage.image, leftImage.keypoints, rightImage.image, rightImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
//             cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


//       cv_bridge::CvImage out_msg;
//       out_msg.header   = header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = img_matches; // Your cv::Mat
//       // mImageMatches.publish(out_msg.toImageMsg());
//       return matches;
//     }
    
// }

// void FeatureDrawer::leftImageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//       ROS_ERROR("cv_bridge exception: %s", e.what());
//       return;
//     }
    
//     if (mFeatureMatchStrat == FeatureStrategy::orb)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detectAndCompute( cv_ptr->image, cv::Mat(), leftKeypoints, leftDescript);
//       cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {255, 0, 0, 255} );
//       cv_bridge::CvImage out_msg;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       leftImage = cv_ptr->image;
//       mLeftImagePub.publish(out_msg.toImageMsg());
//     }
//     if (mFeatureMatchStrat == FeatureStrategy::fast)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detect( cv_ptr->image, leftKeypoints);
//       cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {0, 255, 0, 255} );
//       cv_bridge::CvImage out_msg;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       leftImage = cv_ptr->image;
//       mLeftImagePub.publish(out_msg.toImageMsg());
//     }
//     if (mFeatureMatchStrat == FeatureStrategy::brisk)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detectAndCompute( cv_ptr->image, cv::Mat(), leftKeypoints, leftDescript);
//       cv::drawKeypoints(cv_ptr->image, leftKeypoints, outImage, {0, 0, 255, 255} );
//       cv_bridge::CvImage out_msg;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       leftImage = cv_ptr->image;
//       mLeftImagePub.publish(out_msg.toImageMsg());
//     }
// }

// void FeatureDrawer::rightImageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//       ROS_ERROR("cv_bridge exception: %s", e.what());
//       return;
//     }
//     if (mFeatureMatchStrat == FeatureStrategy::orb)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detectAndCompute( cv_ptr->image, cv::Mat(), rightKeypoints, rightDescript);
//       cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {255, 0, 0, 255} );
//       cv_bridge::CvImage out_msg;
//       // rightKeypoints[0].size;
//       // rightKeypoints[0].pt.x;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       rightImage = cv_ptr->image;
//       mRightImagePub.publish(out_msg.toImageMsg());
//     }
//     if (mFeatureMatchStrat == FeatureStrategy::fast)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detect( cv_ptr->image, rightKeypoints);
//       cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {0, 255, 0, 255} );
//       cv_bridge::CvImage out_msg;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       rightImage = cv_ptr->image;
//       mRightImagePub.publish(out_msg.toImageMsg());
//     }
//     if (mFeatureMatchStrat == FeatureStrategy::brisk)
//     {
//       cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
//       // detect features and descriptor
//       cv::Mat outImage;
//       detector->detectAndCompute( cv_ptr->image, cv::Mat(), rightKeypoints, rightDescript);
//       cv::drawKeypoints(cv_ptr->image, rightKeypoints, outImage, {0, 0, 255, 255} );
//       cv_bridge::CvImage out_msg;
//       // rightKeypoints[0].size;
//       // rightKeypoints[0].pt.x;
//       out_msg.header   = msg->header; // Same timestamp and tf frame as input image
//       out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//       out_msg.image    = outImage; // Your cv::Mat
//       rightImage = cv_ptr->image;
//       mRightImagePub.publish(out_msg.toImageMsg());
//     }
// }

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