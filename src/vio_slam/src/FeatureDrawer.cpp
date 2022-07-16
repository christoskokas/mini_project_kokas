#include "FeatureDrawer.h"
#include <nav_msgs/Odometry.h>
#include <math.h>       /* tan */
#include <boost/assign.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "ceres/ceres.h"
#include "Optimizer.h"


#define PI 3.14159265


static const std::string OPENCV_WINDOW = "Features Detected";

namespace vio_slam
{

bool matIsEqual(const cv::Mat Mat1, const cv::Mat Mat2)
{
  if( Mat1.dims == Mat2.dims && 
    Mat1.size == Mat2.size && 
    Mat1.elemSize() == Mat2.elemSize())
  {
    if( Mat1.isContinuous() && Mat2.isContinuous())
    {
      return 0==memcmp( Mat1.ptr(), Mat2.ptr(), Mat1.total()*Mat1.elemSize());
    }
    else
    {
      const cv::Mat* arrays[] = {&Mat1, &Mat2, 0};
      uchar* ptrs[2];
      cv::NAryMatIterator it( arrays, ptrs, 2);
      for(unsigned int p = 0; p < it.nplanes; p++, ++it)
        if( 0!=memcmp( it.ptrs[0], it.ptrs[1], it.size*Mat1.elemSize()) )
          return false;

      return true;
    }
  }

  return false;
}

void FeatureDrawer::setUndistortMap(ros::NodeHandle *nh)
{
    std::cout << '\n'; 

    std::cout << "CL : " << '\n';
    printMat(zedcamera->cameraLeft.cameraMatrix);
    std::cout << "CR : " << '\n';
    printMat(zedcamera->cameraRight.cameraMatrix);
    cv::Size imgSize = cv::Size(zedcamera->mWidth, zedcamera->mHeight);
    cv::stereoRectify(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, imgSize, zedcamera->sensorsRotate, zedcamera->sensorsTranslate, R1, R2, P1, P2, Q);
    std::cout << "P1 : " << '\n';
    printMat(P1);
    std::cout << "P2 : " << '\n';
    printMat(P2);
    std::cout << "CL : " << '\n';
    printMat(zedcamera->cameraLeft.cameraMatrix);
    std::cout << "CR : " << '\n';
    printMat(zedcamera->cameraRight.cameraMatrix);
    std::cout << "DL : " << '\n';
    printMat(zedcamera->cameraLeft.distCoeffs);
    std::cout << "DR : " << '\n';
    printMat(zedcamera->cameraRight.distCoeffs);
    std::cout << "SR : " << '\n';
    printMat(zedcamera->sensorsRotate);
    std::cout << "ST : " << '\n';
    printMat(zedcamera->sensorsTranslate);
    std::cout << "\n fx" << zedcamera->cameraRight.fx;
    std::cout << "\n fy" << zedcamera->cameraRight.fy;
    std::cout << "\n k1" << zedcamera->cameraRight.k1;
    cv::initUndistortRectifyMap(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, R1, P1, imgSize, CV_32FC1, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, R2, P2, imgSize, CV_32FC1, rmap[1][0], rmap[1][1]);
    
}

void FeatureDrawer::printMat(cv::Mat matrix)
{
  for (size_t i = 0; i < matrix.cols; i++)
  {
    for (size_t j = 0; j < matrix.rows; j++)
    {
      std::cout << matrix.at<double>(j,i) << "  ";
    }
    std::cout << '\n';
  }
  
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
    cv::Mat rod;
    cv::Rodrigues(R1, rod);
    camera[0] = rod.at<double>(0);
    camera[1] = rod.at<double>(1);
    camera[2] = rod.at<double>(2);
    camera[3] = zedcamera->sensorsTranslate.at<double>(0);
    camera[4] = zedcamera->sensorsTranslate.at<double>(1);
    camera[5] = zedcamera->sensorsTranslate.at<double>(2);

    leftIm.subscribe(*nh, zedcamera->cameraLeft.path, 1);
    rightIm.subscribe(*nh, zedcamera->cameraRight.path, 1);
    img_sync.registerCallback(boost::bind(&FeatureDrawer::featureDetectionCallback, this, _1, _2));
    mImageMatches = m_it.advertise("/camera/matches", 1);
    pose_pub = nh->advertise<nav_msgs::Odometry>("odom/ground_truth/2",1);
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

void FeatureDrawer::setPrevious(std::vector<cv::DMatch> matches, cv::Mat points4D)
{
    // if (matIsEqual(previousLeftImage.image, leftImage.image))
    // {
    //   std::cout << "FIRST MATS ARE EQUAL \n";
    // }
    previousLeftImage = leftImage;
    previousRightImage = rightImage;
    // if (matIsEqual(previousLeftImage.image, leftImage.image))
    // {
    //   std::cout << "MATS ARE EQUAL \n";
    // }
    previousPoints4D = points4D;
    previousMatches.clear();
    previousMatches = matches;
    for (size_t i = 0; i < sizeof(sums)/sizeof(sums[0]); i++)
    {
      previousSums[i] = sums[i];
    }
}

void Features::getPoints()
{
    std::cout << "x : " << pointsPosition.at(0).x << " y : " << pointsPosition.at(0).y << " z : " << pointsPosition.at(0).z << '\n'; 
}

void FeatureDrawer::keepMatches(std::vector<cv::DMatch> matches, std::vector<cv::DMatch> matches2, bool left)
{
  cv::Mat points4D = calculateFeaturePosition(matches);
  std::vector<cv::DMatch> matched;
  ceres::Problem problem;
  ceres::LossFunction* lossfunction = NULL;
  for (size_t i = 0; i < matches.size(); i++)
  {
    for (size_t j = 0; j < matches2.size(); j++)
    {
      if ((left && (leftImage.keypoints[matches[i].queryIdx].pt.x == leftImage.keypoints[matches2[j].queryIdx].pt.x) && (leftImage.keypoints[matches[i].queryIdx].pt.y == leftImage.keypoints[matches2[j].queryIdx].pt.y)) || (!left && (rightImage.keypoints[matches[i].trainIdx].pt.x == rightImage.keypoints[matches2[j].trainIdx].pt.x) && (rightImage.keypoints[matches[i].trainIdx].pt.y == rightImage.keypoints[matches2[j].trainIdx].pt.y)))
      {
        // matched.push_back(matches[i]);
        // float sumx = points4D.at<float>(0,i)/points4D.at<float>(3,i) - previousPoints4D.at<float>(0,j)/previousPoints4D.at<float>(3,j);
        // float sumy  = points4D.at<float>(1,i)/points4D.at<float>(3,i) - previousPoints4D.at<float>(1,j)/previousPoints4D.at<float>(3,j);
        // float sumz  = points4D.at<float>(2,i)/points4D.at<float>(3,i) - previousPoints4D.at<float>(2,j)/previousPoints4D.at<float>(3,j);
        // if (!(isnan(abs(sumx)) || isnan(abs(sumy)) || isnan(abs(sumz))) && !(isinf(sumx) || isinf(sumy) || isinf(sumz)) && !((abs(sumx)> zedcamera->mBaseline * 10000) || (abs(sumy) > zedcamera->mBaseline * 10000) || (abs(sumz) > zedcamera->mBaseline * 10000)))
        // {
        if (points4D.at<double>(2,i)/points4D.at<double>(3,i) < 100)
        {
          Eigen::Vector3d p3d(points4D.at<double>(0,i)/points4D.at<double>(3,i), points4D.at<double>(1,i)/points4D.at<double>(3,i), points4D.at<double>(2,i)/points4D.at<double>(3,i));
          Eigen::Vector3d pp3d(previousPoints4D.at<double>(0,j)/previousPoints4D.at<double>(3,j), previousPoints4D.at<double>(1,j)/previousPoints4D.at<double>(3,j), previousPoints4D.at<double>(2,j)/previousPoints4D.at<double>(3,j));
          ceres::CostFunction* costfunction = Reprojection3dError::Create(p3d, pp3d);
          problem.AddResidualBlock(costfunction, lossfunction, camera);
        }
        //   break;
        // }
        // Eigen::Vector2d p2d;
        // Eigen::Vector2d pp2d;
        // if(left)
        // {
        //   p2d(leftImage.keypoints[matches[i].queryIdx].pt.x,leftImage.keypoints[matches[i].queryIdx].pt.y);
        //   pp2d(previousLeftImage.keypoints[matches2[j].queryIdx].pt.x,previousLeftImage.keypoints[matches2[j].queryIdx].pt.y);
        // }
        // else
        // {
        //   p2d(rightImage.keypoints[matches[i].trainIdx].pt.x,rightImage.keypoints[matches[i].trainIdx].pt.y);
        //   pp2d(previousRightImage.keypoints[matches2[j].trainIdx].pt.x,previousRightImage.keypoints[matches2[j].trainIdx].pt.y);
        // }
      }
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << std::endl;
  // std::cout << "After Optimizing: "  << std::endl;
  
  double quat[4];
  ceres::AngleAxisToQuaternion(camera, quat);
  Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
  Eigen::Isometry3d Transform(q.matrix());
  Transform.pretranslate(Eigen::Vector3d(camera[3], camera[4], camera[5]));
  T = Transform.matrix();
  std::cout << "T=\n" << T << std::endl;
}

void FeatureDrawer::allMatches(const std_msgs::Header& header)
{
    leftImage.findFeatures();
    rightImage.findFeatures();
    bool LR = true;
    std::vector<cv::DMatch> matches = leftImage.findMatches(rightImage, header, mImageMatches, LR);
    cv::Mat points4D = calculateFeaturePosition(matches);
    if (!firstImage)
    {
      int difmatches = 2;
      LR = false;
      std::vector<cv::DMatch> matchesLpL = leftImage.findMatches(previousLeftImage, header, mImageMatches, LR);
      keepMatches(matches, matchesLpL, true);
      std::vector<cv::DMatch> matchesRpR = rightImage.findMatches(previousRightImage, header, mImageMatches, LR);
      // keepMatches(matches, matchesRpR, false);
      // std::cout << "NEW ONE \n";
      for (size_t i = 0; i < sizeof(sums)/sizeof(sums[0]); i++)
      {
        sums[i] = sums[i]/difmatches;
      }
      publishMovement(header);
    }
    
    setPrevious(matches, points4D);

}

void FeatureDrawer::publishMovement(const std_msgs::Header& header)
{
  nav_msgs::Odometry position;
  if (abs(T(0,3)) < 100 && abs(T(1,3)) < 100 && abs(T(2,3)) < 100)
  {
    sumsMovement[0] += T(0,3);
    sumsMovement[1] += T(1,3);
    sumsMovement[2] += T(2,3);

    // tf2::Matrix3x3 tf2_rot(T(0,0),T(0,1),T(0,2),
    //                        T(1,0),T(1,1),T(1,2),
    //                        T(2,0),T(2,1),T(2,2));
    // tf2::Transform tf2_transform(tf2_rot,tf2::Vector3());
    double quat[4];
    ceres::AngleAxisToQuaternion(camera, quat);
    tf::poseTFToMsg(tf::Pose(tf::Quaternion(quat[1],quat[2],quat[3],quat[0]),  tf::Vector3(T(0,3), T(1,3), T(2,3))), position.pose.pose); //Aria returns pose in mm.
    // tf2::toMsg(tf2_transform,position.pose.pose);
    position.pose.covariance =  boost::assign::list_of(1e-3) (0) (0)  (0)  (0)  (0)
                                                        (0) (1e-3)  (0)  (0)  (0)  (0)
                                                        (0)   (0)  (1e6) (0)  (0)  (0)
                                                        (0)   (0)   (0) (1e6) (0)  (0)
                                                        (0)   (0)   (0)  (0) (1e6) (0)
                                                        (0)   (0)   (0)  (0)  (0)  (1e3) ;

    position.twist.twist.linear.x = 0.0;                  //(sumsMovement[0]-previoussumsMovement[0])*15 //15 fps
    position.twist.twist.angular.z = 0.0;
    position.twist.covariance =  boost::assign::list_of(1e-3) (0)   (0)  (0)  (0)  (0)
                                                        (0) (1e-3)  (0)  (0)  (0)  (0)
                                                        (0)   (0)  (1e6) (0)  (0)  (0)
                                                        (0)   (0)   (0) (1e6) (0)  (0)
                                                        (0)   (0)   (0)  (0) (1e6) (0)
                                                        (0)   (0)   (0)  (0)  (0)  (1e3) ; 

    position.header.frame_id = header.frame_id;
    position.header.stamp = ros::Time::now();
    // std::cout << " x sum : " << sumsMovement[0] << " y sum : " << sumsMovement[1] << " z sum : " << sumsMovement[2] << '\n';
    for (size_t i = 0; i < 3; i++)
    {
      previoussumsMovement[i] = sumsMovement[i];
    }
    pose_pub.publish(position);
  }
}

void FeatureDrawer::calculateMovementFeatures(std::vector<cv::DMatch> matches, std::vector<cv::DMatch> matches2, cv::Mat Points4D, bool left)
{
  int count {0};
  float xSum {0.0f};
  float ySum {0.0f};
  float zSum {0.0f};
  for (size_t i = 0; i < matches.size(); i++)
  {
    for (size_t j = 0; j < matches2.size(); j++)
    {
      if ((left && (leftImage.keypoints[matches[i].queryIdx].pt == leftImage.keypoints[matches2[j].queryIdx].pt))|| (!left && (rightImage.keypoints[matches[i].trainIdx].pt == rightImage.keypoints[matches2[j].trainIdx].pt)))
      if ((left && (matches[i].queryIdx == matches2[j].queryIdx)) || (!left && (matches[i].trainIdx == matches2[j].trainIdx)))
      {
        float sumx = Points4D.at<float>(0,i)/Points4D.at<float>(3,i) - previousPoints4D.at<float>(0,j)/previousPoints4D.at<float>(3,j);
        float sumy  = Points4D.at<float>(1,i)/Points4D.at<float>(3,i) - previousPoints4D.at<float>(1,j)/previousPoints4D.at<float>(3,j);
        float sumz  = Points4D.at<float>(2,i)/Points4D.at<float>(3,i) - previousPoints4D.at<float>(2,j)/previousPoints4D.at<float>(3,j);
        if (!(isnan(abs(sumx)) || isnan(abs(sumy)) || isnan(abs(sumz))) && !(isinf(sumx) || isinf(sumy) || isinf(sumz)) && !((abs(sumx)> zedcamera->mBaseline * 10000) || (abs(sumy) > zedcamera->mBaseline * 10000) || (abs(sumz) > zedcamera->mBaseline * 10000)))
        {
          xSum += sumx;
          ySum += sumy;
          zSum += sumz;
          count++;
          break;
        }
      }
    }
  }
  sums[0] = xSum/count;
  sums[1] = ySum/count;
  sums[2] = zSum/count;
  
  
}

std::vector<cv::DMatch> Features::findMatches(const Features& secondImage, const std_msgs::Header& header, image_transport::Publisher& mImageMatches, bool LR)
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
    if (matches.size() > 20)
    {
      matches.resize(20);
    }
    cv::Mat img_matches;
    drawMatches( image, keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    if (LR)
    {
      cv_bridge::CvImage out_msg;
      out_msg.header   = header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = img_matches; // Your cv::Mat
      mImageMatches.publish(out_msg.toImageMsg());
    }
    return matches;
}

cv::Mat FeatureDrawer::calculateFeaturePosition(const std::vector<cv::DMatch>& matches)
{

  cv::Mat points4D(4,1,CV_64F);
  if ( matches.size() > 0 )
  {
    std::vector<cv::Point2d> pointsL;
    std::vector<cv::Point2d> pointsR;
    for (size_t i = 0; i < matches.size(); i++)
    {
      pointsL.push_back(leftImage.keypoints[matches[i].queryIdx].pt);
      pointsR.push_back(rightImage.keypoints[matches[i].trainIdx].pt);
    }

    // cv::Mat leftKey(pointsL);
    // cv::Mat rightKey(pointsR);
    // cv::Mat caml(3,4,CV_64F);
    // cv::Mat camr(3,4,CV_64F);
    // caml = P1;
    // camr = P2;
    // printMat(caml);
    // printMat(P1);
    // cv::undistortPoints(pointsL,pointsL,zedcamera->cameraLeft.cameraMatrix,zedcamera->cameraLeft.distCoeffs, R1, P1);
    // cv::undistortPoints(pointsR,pointsR,zedcamera->cameraRight.cameraMatrix,zedcamera->cameraRight.distCoeffs, R2, P2);
    // cv::Mat C1 = (cv::Mat_<float>(3,4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    // cv::Mat C2 = (cv::Mat_<float>(3,4) << 1, 0, 0, zedcamera->sensorsTranslate.at<float>(0,0), 0, 1, 0, 0, 0, 0, 1, 0);
    // std::cout << pointsL.at(0).x << " y " <<  pointsL.at(0).y << "LEL \n";
    // std::cout << pointsR.at(0).x << " y " <<  pointsR.at(0).y << "LEL \n";
    cv::triangulatePoints(P1, P2, pointsL, pointsR, points4D);
    // std::cout << leftImage.keypoints[matches[0].queryIdx].pt << '\n';
    // std::cout << rightImage.keypoints[matches[0].trainIdx].pt << '\n';
    // std::cout << "x : " << points4D.at<float>(0,0)/points4D.at<float>(3,0) << " y : " << points4D.at<float>(1,0)/points4D.at<float>(3,0) << " z : " << points4D.at<float>(2,0)/points4D.at<float>(3,0) << '\n';
    // std::cout << pointsL.at(0).x << " y " <<  pointsL.at(0).y << "LEL \n";
    // std::cout << pointsR.at(0).x << " y " <<  pointsR.at(0).y << "LEL \n";
    // std::cout << "points \n";
    // printMat(points4D);
    // std::cout << "right \n";
    // printMat(rightKey);

  }
  return points4D;
}

void FeatureDrawer::clearFeaturePosition()
{
    leftImage.pointsPosition.clear();
    rightImage.pointsPosition.clear();
    previousLeftImage.pointsPosition.clear();
    previousRightImage.pointsPosition.clear();
}

void FeatureDrawer::featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
    prevTime = ros::Time::now();
    setImage(lIm, leftImage.image);
    setImage(rIm, rightImage.image);
    // cv::Mat dstle, dstri;
    if (!zedcamera->rectified)
    {
      cv::remap(leftImage.image, leftImage.image, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
      cv::remap(rightImage.image, rightImage.image, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
      // cv::hconcat(leftImage.image, dstle, dstle);                       //add 2 images horizontally (image1, image2, destination)
      // cv::hconcat(rightImage.image, dstri, dstri);                      //add 2 images horizontally (image1, image2, destination)
      // cv::vconcat(dstle, dstri, dstle);                           //add 2 images vertically
      // cv_bridge::CvImage out_msg;
      // out_msg.header   = lIm->header; // Same timestamp and tf frame as input image
      // out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      // out_msg.image    = dstle; // Your cv::Mat
      // mImageMatches.publish(out_msg.toImageMsg());
      // std::cout << "NOT RECTIFIED" << '\n';
    }
    allMatches(lIm->header);
    
    
    firstImage = false;
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