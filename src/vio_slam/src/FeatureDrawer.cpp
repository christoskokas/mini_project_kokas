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
    std::cout << "enter matches \n";
    if ( descriptors.empty() )
      cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
    if ( secondImage.descriptors.empty() )
      cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

    // std::vector<cv::DMatch> matches;
    std::vector< std::vector<cv::DMatch> > matches, matches2;
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // cv::BFMatcher matcher(cv::NORM_HAMMING);  
    // matcher.knnMatch(descriptors, secondImage.descriptors, matches, 2);
    // matcher.knnMatch(secondImage.descriptors, descriptors, matches2, 2);
    // for(size_t i = 0; i < matches.size(); i++) {
    //   if(matches[i].size() != 0 && matches2[matches[i][0].trainIdx].size() != 0)
    //   {
    //     cv::DMatch first = matches[i][0];
    //     cv::DMatch second = matches2[matches[i][0].trainIdx][0];
    //     if (second.trainIdx == first.queryIdx)
    //     {
    //       float dist1 = matches[i][0].distance;
    //       float dist2 = matches[i][1].distance;
    //       if(dist1 < 0.8f * dist2) {
    //           matched1.push_back(keypoints[first.queryIdx]);
    //           matched2.push_back(secondImage.keypoints[first.trainIdx]);
    //           pointl.push_back(keypoints[first.queryIdx].pt);
    //           pointr.push_back(secondImage.keypoints[first.trainIdx].pt);
    //       }
    //     }
    //   }
    // }
    std::vector <std::vector< std::vector<cv::DMatch> > > match;
    std::vector< std::vector<cv::DMatch> >  matchedscale;
    
    std::vector<cv::KeyPoint> matched1, matched2;
    std::vector<cv::Point2f> pointl, pointr;
    int steps = 4;
    for (size_t i = 0; i < steps; i++)
    {
      std::cout << "0 size : " << multiDescriptors[0].size() <<  " next size : " <<  secondImage.multiDescriptors[i].size() << '\n';
      std::vector< std::vector<cv::DMatch> > matchedd;
      matcher.knnMatch(multiDescriptors[0], secondImage.multiDescriptors[i], matchedd, 2);
      match.push_back(matchedd);
      std::cout << matchedd.size()<< "exit matcheeeer \n";
      std::cout << match[i].size()<< "exit matcheeeer \n";
    }

    for (size_t j = 0; j < steps; j++)
    {
      std::vector<cv::DMatch> matcheddd;
      for(size_t i = 0; i < match[j].size(); i++) 
      {

        // std::cout << match[j][i].size()<< "yes \n";
        if(match[j][i].size() != 0)
        {
          
          cv::DMatch first = match[j][i][0];
          float dist1 = match[j][i][0].distance;
          float dist2 = match[j][i][1].distance;
          if(dist1 < 0.8f * dist2) 
          {

            matcheddd.push_back(match[j][i][0]);
          }
        }
      }
    for (int i = 0; i < matcheddd.size(); i++)
    {
      for (int j = 0; j < matcheddd.size() - 1; j++)
      {
        if (matcheddd[j].distance > matcheddd[j + 1].distance)
        {
          auto temp = matcheddd[j];
          matcheddd[j] = matcheddd[j + 1];
          matcheddd[j + 1] = temp;
        }
      }
    }
    if (matcheddd.size() > 200)
    {
      matcheddd.resize(200);
    }
    matchedscale.push_back(matcheddd);
    matcheddd.clear();
    }

    // Find Minimum distance
    
    for (size_t i = 0; i < matchedscale[0].size(); i++)
    {
      for (size_t j = 1; j < steps; j++)
      {
        for (size_t k = 0; k < matchedscale[j].size(); k++)
        {
          if (multiKeypoints[0][matchedscale[0][i].queryIdx].pt == secondImage.multiKeypoints[j][matchedscale[j][k].queryIdx].pt)
          {
            if (matchedscale[0][i].distance > matchedscale[j][i].distance)
            {
              matchedscale[0][i].distance = matchedscale[j][i].distance;
              // matchedscale[0][i].trainIdx = matchedscale[j][i].trainIdx;
              multiKeypoints[0][matchedscale[0][i].trainIdx].pt.x = secondImage.multiKeypoints[j][matchedscale[j][k].trainIdx].pt.x/(1+1/(steps*j));
              multiKeypoints[0][matchedscale[0][i].trainIdx].pt.y = secondImage.multiKeypoints[j][matchedscale[j][k].trainIdx].pt.y/(1+1/(steps*j));

              break;
            }
          }
          
        }
        
        
      }
      matched1.push_back(multiKeypoints[0][matchedscale[0][i].queryIdx]);
      matched2.push_back(multiKeypoints[0][matchedscale[0][i].trainIdx]);
      pointl.push_back(multiKeypoints[0][matchedscale[0][i].queryIdx].pt);
      pointr.push_back(multiKeypoints[0][matchedscale[0][i].trainIdx].pt);
    }
    
  
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::KeyPoint> inliers1, inliers2;
    cv::Mat h = findHomography( pointl, pointr, cv::RANSAC);
    std::cout << "matches size : " << matched1.size() << '\n'; 
    for(size_t i = 0; i < matched1.size(); i++) {
        cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = h * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < 2.5f) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(cv::DMatch(new_i, new_i, 0));
        }
    }
    std::cout << "good matches size : " << good_matches.size() << '\n'; 


    cv::Mat img_matches;
    cv_bridge::CvImage out_msg;
    drawMatches( image, inliers1, secondImage.image, inliers2, good_matches, img_matches, cv::Scalar::all(-1),
          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    // drawMatches( image, keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
    //       cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    if(!LR)
    {
      out_msg.header   = header; // Same timestamp and tf frame as input image
      out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
      out_msg.image    = img_matches; // Your cv::Mat
      mImageMatches.publish(out_msg.toImageMsg());
    }
    // for (int i = 0; i < matches.size(); i++)
    // {
    //   for (int j = 0; j < matches.size() - 1; j++)
    //   {
    //     if (matches[j].distance > matches[j + 1].distance)
    //     {
    //       auto temp = matches[j];
    //       matches[j] = matches[j + 1];
    //       matches[j + 1] = temp;
    //     }
    //   }
    // }
    // if (matches.size() > 300)
    // {
    //   matches.resize(300);
    // }
    

    // if (LR)
    // {
    //   std::cout << "size before removal : " << matches.size();
    //   for (size_t i = 0; i < matches.size(); i++)
    //   {
    //     if (abs(keypoints[matches[i].queryIdx].pt.y - secondImage.keypoints[matches[i].trainIdx].pt.y) > 3)
    //     {
    //       matches.erase(matches.begin() + i);
    //       i--;
    //     }
    //   }
    //   std::cout << "\n size after removal : " << matches.size() << '\n';
    // }
    // else
    // {
    //   std::vector<cv::KeyPoint> points1, points2;
    //   std::vector<cv::Point2f> pointsl, pointsr;
    //   for( size_t i = 0; i < matches.size(); i++ )
    //     {
    //       points1.push_back( keypoints[ matches[i].queryIdx ] );
    //       points2.push_back( secondImage.keypoints[ matches[i].trainIdx ] );
    //       pointsl.push_back( keypoints[ matches[i].queryIdx ].pt );
    //       pointsr.push_back( secondImage.keypoints[ matches[i].trainIdx ].pt );
    //     }
    //   cv::Mat h = findHomography( pointsl, pointsr, cv::RANSAC );
    //   std::vector<cv::DMatch> good_matches;
    //   std::vector<cv::KeyPoint> inliers1, inliers2;
    //   for(size_t i = 0; i < matches.size(); i++) 
    //   {
    //     cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
    //     col.at<double>(0) = points1[i].pt.x;
    //     col.at<double>(1) = points1[i].pt.y;
    //     col = h * col;
    //     col /= col.at<double>(2);
    //     double dist = sqrt( pow(col.at<double>(0) - points2[i].pt.x, 2) +
    //                         pow(col.at<double>(1) - points2[i].pt.y, 2));
    //     if(dist < 2.5f) {
    //         int new_i = static_cast<int>(inliers1.size());
    //         inliers1.push_back(points1[i]);
    //         inliers2.push_back(points2[i]);
    //         good_matches.push_back(cv::DMatch(new_i, new_i, 0));
    //     }
    //   }
    //   cv::Mat img_matches;
    //   std::cout << " BEST MATCH : " << matches[100].distance << '\n';
    //   cv_bridge::CvImage out_msg;
    //   drawMatches( image, inliers1, secondImage.image, inliers2, good_matches, img_matches, cv::Scalar::all(-1),
    //         cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //   // drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    //   // drawMatches( image, keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
    //   //       cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //   out_msg.header   = header; // Same timestamp and tf frame as input image
    //   out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
    //   out_msg.image    = img_matches; // Your cv::Mat
    //   mImageMatches.publish(out_msg.toImageMsg());

    // }
    std::cout << "exit matches \n";
    return good_matches;
}

void Features::findFeatures()
{
    std::cout << "enter features \n";
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // detect features and descriptor
    cv::Mat outImage;
    detector->detectAndCompute( image, cv::Mat(), keypoints, descriptors);
    cv::drawKeypoints(image, keypoints, outImage, {255, 0, 0, 255} );
    if (true)
    {
      std::vector<cv::KeyPoint> xdkey;
      detector->detect(image,xdkey, cv::Mat());
      int steps {4};
      for (int i = 0; i < steps; i++)
      {
        std::vector<cv::KeyPoint> pyrKeypoints = xdkey;
        for (size_t j = 0; j < xdkey.size(); j++)
        {
          pyrKeypoints[j].pt = pyrKeypoints[j].pt*(1 + static_cast<float>(i)/steps);
        }
        multiKeypoints.push_back(pyrKeypoints);
        cv::Mat imageScale, descr;
        cv::resize(image, imageScale, cv::Size(static_cast<int>(image.cols*(1 + static_cast<float>(i)/steps)), static_cast<int>(image.rows*(1 + static_cast<float>(i)/steps))));
        cv::blur(imageScale, imageScale,cv::Size(51,51));
        // cv::pyrUp(image, imageScale, cv::Size(static_cast<int>(image.cols*(1 + static_cast<float>(i)/steps)), static_cast<int>(image.rows*(1 + static_cast<float>(i)/steps))));
        detector->compute(imageScale,pyrKeypoints,descr);
        multiDescriptors.push_back(descr);
        
      }
      std::cout << "exit features \n";
    }
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
    leftImage.multiDescriptors.clear();
    rightImage.multiDescriptors.clear();
    leftImage.multiKeypoints.clear();
    rightImage.multiKeypoints.clear();
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
    cv::cvtColor(cv_ptr->image, image, cv::COLOR_BGR2GRAY);
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

} //namespace vio_slam