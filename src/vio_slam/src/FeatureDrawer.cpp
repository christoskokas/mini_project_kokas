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


FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, const Zed_Camera* zedptr) : m_it(*nh), img_sync(MySyncPolicy(10), leftIm, rightIm)
{
    this->zedcamera = zedptr;
    if (!zedcamera->rectified)
    {
      setUndistortMap(nh);
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
    std::string position_path;
    nh->getParam("ground_truth_path", position_path);
    leftIm.subscribe(*nh, zedcamera->cameraLeft.path, 1);
    rightIm.subscribe(*nh, zedcamera->cameraRight.path, 1);
    img_sync.registerCallback(boost::bind(&FeatureDrawer::featureDetectionCallback, this, _1, _2));
    mImageMatches = m_it.advertise("/camera/matches", 1);
    pose_pub = nh->advertise<nav_msgs::Odometry>(position_path,1);
}

void FeatureDrawer::featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
    prevTime = ros::Time::now();
    setImage(lIm, leftImage.image);
    setImage(rIm, rightImage.image);
    if (!zedcamera->rectified)
    {
      cv::remap(leftImage.image, leftImage.image, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
      cv::remap(rightImage.image, rightImage.image, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
    }
    allMatches(lIm->header);
    
    
    firstImage = false;
}

void FeatureDrawer::setUndistortMap(ros::NodeHandle *nh)
{
    cv::Size imgSize = cv::Size(zedcamera->mWidth, zedcamera->mHeight);
    cv::stereoRectify(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, imgSize, zedcamera->sensorsRotate, zedcamera->sensorsTranslate, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, R1, P1, imgSize, CV_32FC1, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, R2, P2, imgSize, CV_32FC1, rmap[1][0], rmap[1][1]);
    
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
    cv::triangulatePoints(P1, P2, pointsL, pointsR, points4D);
    std::cout << "NEW IMAGE\n";
    for (size_t i = 0; i < points4D.cols; i++)
    {
      // for (size_t j = 0; j < 3; j++)
      // {
      //   std::cout << points4D.at<double>(j,i)/points4D.at<double>(3,i) << "  ";
      // }
      // std::cout << '\n';
      // std::cout << pointsL.at(i);
      // std::cout << pointsR.at(i);
      // std::cout << '\n';
    }
    
    
  }
  return points4D;
}

void Features::findFeatures()
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // detect features and descriptor
    cv::Mat outImage;
    detector->detectAndCompute( image, cv::Mat(), keypoints, descriptors);
    cv::drawKeypoints(image, keypoints, outImage, {255, 0, 0, 255} );
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
      LR = false;
      std::vector<cv::DMatch> matchesLpL = leftImage.findMatches(previousLeftImage, header, mImageMatches, LR);
      keepMatches(matches, matchesLpL, true);
      publishMovement(header);
    }
    
    setPrevious(matches, points4D);

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
        if ((abs(points4D.at<double>(0,i)/points4D.at<double>(3,i)) < 100) && (abs(points4D.at<double>(1,i)/points4D.at<double>(3,i)) < 100) && (abs(points4D.at<double>(2,i)/points4D.at<double>(3,i)) < 100) && !isnan(abs(points4D.at<double>(0,i))) && !isnan(abs(points4D.at<double>(1,i))) && !isnan(abs(points4D.at<double>(2,i))) && !isnan(abs(points4D.at<double>(3,i))))
        {
          Eigen::Vector3d p3d(points4D.at<double>(0,i)/points4D.at<double>(3,i), points4D.at<double>(1,i)/points4D.at<double>(3,i), points4D.at<double>(2,i)/points4D.at<double>(3,i));
          Eigen::Vector3d pp3d(previousPoints4D.at<double>(0,j)/previousPoints4D.at<double>(3,j), previousPoints4D.at<double>(1,j)/previousPoints4D.at<double>(3,j), previousPoints4D.at<double>(2,j)/previousPoints4D.at<double>(3,j));
          // std::cout << p3d << '\n';
          // std::cout << '\n';
          ceres::CostFunction* costfunction = Reprojection3dError::Create(p3d, pp3d);
          problem.AddResidualBlock(costfunction, lossfunction, camera);
        }
        break;
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
  // std::cout << "T=\n" << T << std::endl;
}

void FeatureDrawer::publishMovement(const std_msgs::Header& header)
{
  nav_msgs::Odometry position;
  if (abs(T(0,3)) < 100 && abs(T(1,3)) < 100 && abs(T(2,3)) < 100)
  {
    sumsMovement[0] += T(0,3);
    sumsMovement[1] += T(1,3);
    sumsMovement[2] += T(2,3);
    Eigen::Matrix3d Rot;
    {
      Eigen::Matrix4d temp = previousT;
      for (size_t i = 0; i < 3; i++)
      {
        for (size_t j = 0; j < 4; j++)
        {
          previousT(i,j) = temp(i,0)*T(0,j) + temp(i,1)*T(1,j) + temp(i,2)*T(2,j) + temp(i,3)*T(3,j);
          if (j < 3)
            Rot(i,j) = previousT(i,j);
        }
      }
    }
    Eigen::Quaterniond quat(Rot.topLeftCorner<3,3>());
    // std::cout << "T=\n" << T << std::endl;
    // std::cout << "Tprev=\n" << previousT << std::endl;
    // std::cout << "Rot=\n" << Rot << std::endl;
    // std::cout << "quat=\n" << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
    // ceres::AngleAxisToQuaternion(camera, quat);
    tf::poseTFToMsg(tf::Pose(tf::Quaternion(quat.x(),quat.y(),quat.z(),quat.w()),  tf::Vector3(previousT(0,3), previousT(1,3), previousT(2,3))), position.pose.pose); //Aria returns pose in mm.
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
    for (size_t i = 0; i < 3; i++)
    {
      previoussumsMovement[i] = sumsMovement[i];
    }
    pose_pub.publish(position);
  }
}

std::vector<cv::DMatch> Features::findMatches(const Features& secondImage, const std_msgs::Header& header, image_transport::Publisher& mImageMatches, bool LR)
{
    if ( descriptors.empty() )
      cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
    if ( secondImage.descriptors.empty() )
      cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

    std::vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true);  
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
    if (matches.size() > 300)
    {
      matches.resize(300);
    }
    

    if (LR)
    {
      std::cout << "size before removal : " << matches.size();
      for (size_t i = 0; i < matches.size(); i++)
      {
        if (abs(keypoints[matches[i].queryIdx].pt.y - secondImage.keypoints[matches[i].trainIdx].pt.y) > 3)
        {
          matches.erase(matches.begin() + i);
          i--;
        }
      }
      std::cout << "\n size after removal : " << matches.size() << '\n';
      cv_bridge::CvImage out_msg;
      cv::Mat img_matches;
      drawMatches( image, keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      out_msg.header   = header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = img_matches; // Your cv::Mat
      mImageMatches.publish(out_msg.toImageMsg());
    }
    return matches;
}

void FeatureDrawer::clearFeaturePosition()
{
    leftImage.pointsPosition.clear();
    rightImage.pointsPosition.clear();
    previousLeftImage.pointsPosition.clear();
    previousRightImage.pointsPosition.clear();
}

void FeatureDrawer::setPrevious(std::vector<cv::DMatch> matches, cv::Mat points4D)
{
    previousLeftImage = leftImage;
    previousRightImage = rightImage;
    previousPoints4D = points4D;
    previousMatches.clear();
    previousMatches = matches;
    for (size_t i = 0; i < sizeof(sums)/sizeof(sums[0]); i++)
    {
      previousSums[i] = sums[i];
    }
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

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

} //namespace vio_slam