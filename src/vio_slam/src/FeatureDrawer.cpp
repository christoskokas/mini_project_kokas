#include "FeatureDrawer.h"
#include <nav_msgs/Odometry.h>
#include <math.h>       /* tan */
#include <boost/assign.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "ceres/ceres.h"
#include <algorithm>
#include <functional>
#include "Optimizer.h"
#include <opencv2/flann.hpp>


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

void printMat(cv::Mat matrix)
{
  for (size_t i = 0; i < matrix.rows; i++)
  {
    for (size_t j = 0; j < matrix.cols; j++)
    {
      std::cout << matrix.at<double>(i,j) << "  ";
    }
    std::cout << '\n';
  }
  
}

float distance(int x1, int y1, int x2, int y2)
{
    // Calculating distance
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
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
    camera[3] = 0;
    camera[4] = 0;
    camera[5] = 0;
    std::string position_path;
    nh->getParam("ground_truth_path", position_path);
    leftIm.subscribe(*nh, zedcamera->cameraLeft.path, 1);
    rightIm.subscribe(*nh, zedcamera->cameraRight.path, 1);
    img_sync.registerCallback(boost::bind(&FeatureDrawer::featureDetectionCallback, this, _1, _2));
    mImageMatches = m_it.advertise("/camera/matches", 1);
    pose_pub = nh->advertise<nav_msgs::Odometry>(position_path,1);
}

// ROWS of descriptors are each keypoint
// keys size : 2353
// descriptors size : 2353 32


std::vector<cv::DMatch> FeatureDrawer::matchesLR(Features& leftImage, Features& rightImage)
{
  int ind = 0;
  std::vector<cv::DMatch> trueMatches;
  std::vector<cv::KeyPoint> inliers1, inliers2;
  std::sort(leftImage.keypoints.begin(), leftImage.keypoints.end(),[](cv::KeyPoint a, cv::KeyPoint b) {return a.pt.y < b.pt.y;});
  std::sort(rightImage.keypoints.begin(), rightImage.keypoints.end(),[](cv::KeyPoint a, cv::KeyPoint b) {return a.pt.y < b.pt.y;});
  int steps = 10;
  int pxlStep = leftImage.image.rows/steps;
  int startLeft = 0;
  int startRight = 0;
  std::cout << "start " <<  leftImage.keypoints[0].pt << '\n';
  std::cout << "finish " <<  leftImage.keypoints[leftImage.keypoints.size()-1].pt << '\n';
  cv::BFMatcher matcher(cv::NORM_HAMMING); 
  for (size_t iii = 0; iii < steps; iii++)
  {
    std::vector < cv::KeyPoint> leftkeys;
    for (size_t jjj = startLeft; jjj < leftImage.keypoints.size(); jjj++)
    {
      if (leftImage.keypoints[jjj].pt.y <= pxlStep * (1 + iii))
        leftkeys.push_back(leftImage.keypoints[jjj]);
      else
      {
        startLeft = jjj;
        break;
      }
    }
    std::cout << "start " <<  startLeft << " key " << leftImage.keypoints[startLeft].pt << " step : " << pxlStep * (1 + iii) << '\n';
    
    std::vector < cv::KeyPoint> rightkeys;
    for (size_t jjj = startRight; jjj < rightImage.keypoints.size(); jjj++)
    {
      if ((rightImage.keypoints[jjj].pt.y <= pxlStep * (1 + iii)) || abs(rightImage.keypoints[jjj].pt.y - pxlStep * (1 + iii)) < 2)
        rightkeys.push_back(rightImage.keypoints[jjj]);
      else
      {
        startRight = jjj;
        break;
      }
    }
    std::cout << "start " <<  startRight << " key " << rightImage.keypoints[startRight].pt << " step : " << pxlStep * (1 + iii) << '\n';
    // std::cout << "start " <<  leftkeys.size() << '\n';
    // std::cout << "finish " <<  rightkeys.size() << '\n';
    cv::Mat leftDesc,rightDisc;
    leftImage.getDescriptors(leftImage.image, leftkeys, leftDesc);
    rightImage.getDescriptors(rightImage.image, rightkeys, rightDisc);
    std::vector<cv::DMatch> matches;
    if (!(leftDesc.empty() || rightDisc.empty()))
    {
      matcher.match(leftDesc,rightDisc, matches, cv::Mat());

      if (matches.size() > 0)
      {
        for (auto m : matches)
        {
          inliers1.push_back(leftkeys[m.queryIdx]);
          inliers2.push_back(rightkeys[m.trainIdx]);
          trueMatches.push_back(m);
        }
      }
    }
  }

  

  // leftImage.getDescriptors(leftImage.image);
  // for (auto key : leftImage.keypoints)
  // {
  //   std::vector < cv::KeyPoint> rightkeys;
  //   Features trial;
  //   // trial.image = rightImage.image.clone();
  //   for (auto rightkey : rightImage.keypoints)
  //   {
  //     if (abs(key.pt.y - rightkey.pt.y) < 2)
  //     {
  //       trial.keypoints.push_back(rightkey);
  //     }
  //   }
  //   trial.getDescriptors(rightImage.image);
  //   std::vector<cv::DMatch> matches;
  //   cv::BFMatcher matcher(cv::NORM_HAMMING); 
  //   // std::cout << " LOL 1\n"; 
  //   if (!(leftImage.descriptors.empty() || trial.descriptors.empty()))
  //   {
  //     matcher.match(leftImage.descriptors.row(ind),trial.descriptors, matches, cv::Mat());
  //     // std::cout << " LOL 2\n"; 

  //     if (matches.size() > 0)
  //     {
  //       inliers1.push_back(key);
  //       inliers2.push_back(trial.keypoints[matches[0].trainIdx]);
  //       trueMatches.push_back(matches[0]);
  //     }
  //   }
  //   ind ++;
  // }
  cv::Mat img_matches;
  cv_bridge::CvImage out_msg;
  std::cout << "matches size : " << trueMatches.size() << '\n';
  drawMatches( leftImage.image, inliers1, rightImage.image, inliers2, trueMatches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  out_msg.header   = leftImage.header; // Same timestamp and tf frame as input image
  out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
  out_msg.image    = img_matches; // Your cv::Mat
  mImageMatches.publish(out_msg.toImageMsg());
  return trueMatches;
}

/**
 * @brief Find Matches between Images according to Computed Descriptors. Using KnnMatcher to find better matches, removing some outliers with Lowe's ratio, and remove more outliers with Homography. Returns good matches.
 * 
 * @param secondImage second Image for Feature Matching (Features class)
 * @param mImageMatches ROS Publisher 
 * @param LR If the images are Left and Right Image or not
 * @return std::vector<cv::DMatch> 
 */
std::vector<cv::DMatch> Features::getMatches(Features& secondImage, image_transport::Publisher& mImageMatches, std::vector<cv::KeyPoint>& previousleftKeypoints, bool LR)
{
  if (!LR)
  {
    previousleftKeypoints = secondImage.keypoints;
  }
  getDescriptors(image, keypoints, descriptors);
  secondImage.getDescriptors(secondImage.image, secondImage.keypoints, secondImage.descriptors);
  if ( descriptors.empty() )
    cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
  if ( secondImage.descriptors.empty() )
    cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
  std::vector< std::vector<cv::DMatch> > matches;
  cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  // cv::BFMatcher matcher(cv::NORM_HAMMING);  
  matcher.knnMatch(descriptors, secondImage.descriptors, matches, 2);
  std::cout << "keys size : " << keypoints.size() << '\n';
  std::cout << "descriptors size : " << descriptors.rows <<  " " << descriptors.cols << '\n';
  std::vector<cv::KeyPoint> matched1, matched2;
  std::vector<cv::Point2f> pointl, pointr;
  for(size_t i = 0; i < matches.size(); i++) 
  {
    if(matches[i].size() >= 2)
    {
      cv::DMatch first = matches[i][0];
      float dist1 = matches[i][0].distance;
      float dist2 = matches[i][1].distance;

      if(dist1 < 0.8f * dist2) 
      {
        matched1.push_back(keypoints[first.queryIdx]);
        matched2.push_back(secondImage.keypoints[first.trainIdx]);
        pointl.push_back(keypoints[first.queryIdx].pt);
        pointr.push_back(secondImage.keypoints[first.trainIdx].pt);
      }
    }

  }
  std::vector<cv::DMatch> good_matches;
  if (pointl.size() > 4)
  {
    std::vector<cv::KeyPoint> inliers1, inliers2;
    cv::Mat h = findHomography( pointl, pointr, cv::RANSAC);
    if (h.rows == 3)
    {
      for(size_t i = 0; i < matched1.size(); i++) 
      {
        
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
            // std::cout << "first : " << matched1[i].pt << " second : " << matched2[i].pt << '\n';
            good_matches.push_back(cv::DMatch(new_i, new_i, 0));
        }
      
      }
      keypointsLR = inliers1;
      secondImage.keypointsLR = inliers2;
      keypoints = inliers1;
      secondImage.keypoints = inliers2;
      cv::Mat img_matches;
      cv_bridge::CvImage out_msg;
      drawMatches( image, inliers1, secondImage.image, inliers2, good_matches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      if(LR)
      {
        // out_msg.header   = header; // Same timestamp and tf frame as input image
        // out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
        // out_msg.image    = img_matches; // Your cv::Mat
        // mImageMatches.publish(out_msg.toImageMsg());
      }
    }
  }
    
    
  

  return good_matches;
}


void Features::getDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(75,1.2f,8,30, 0, 2, cv::ORB::HARRIS_SCORE,30);
  detector->compute(image, keypoints, descriptors);
  // 48 is rows*cols for function again()
  for (size_t iii = 0; iii < 48; iii++)
  {
    cv::Mat temp = descriptors(cv::Range(indicesOfGrids[iii],indicesOfGrids[iii + 1]),cv::Range::all());
    descriptorsGrids.push_back(temp);
  }
  
}


/**
 * @brief Find Features with Adaptive Fast Threshold. This is used to be certain that a number of features will be found on each grid image so that features are homogeneous on the whole image
 * 
 * @param patch the cropped image
 * @param step the step with which the fast threshold is changed per iteration
 * @param iterations the number of times the fast threshold is changed
 * @return std::vector< cv::KeyPoint > returns the resulting keypoints
 */
std::vector< cv::KeyPoint > Features::featuresAdaptiveThreshold(cv::Mat& patch, int step = 5, unsigned int iterations = 3)
{
  int fastThreshold = 20;
  std::vector< cv::KeyPoint > tempkeys;
  for (size_t iii = 0; iii < iterations; iii++)
  {
    int numbOfFeatures = 500;
    int numbNeeded = 50;
    int edgeThreshold = 0;
    findORBFeatures(patch, tempkeys, numbOfFeatures, edgeThreshold, fastThreshold);
    if (tempkeys.size() >=(numbNeeded-5))
    {
      cv::KeyPointsFilter::retainBest(tempkeys, numbNeeded);
      // std::cout << "fast : " << fastThreshold << " size : " << tempkeys.size() << '\n';
      return tempkeys;
    }
    else
      fastThreshold -= step;
  }
  return tempkeys;
}

/**
 * @brief Get Features from image with adaptive threshold and grid based feature extraction
 * 
 * @param rows How many rows to separate image
 * @param cols How many columns to separate image
 * @param mImageMatches ROS Image Publisher 
 */
void Features::getFeatures(int rows, int cols,image_transport::Publisher& mImageMatches, bool left)
{
  // separate image to grid for homogeneity of features

  // Crop image 30 pixels round to have 0 edge threshold
  const int edgeThreshold = 30; 
  cv::Mat croppedImage;
  {
    cv::Rect crop(edgeThreshold, edgeThreshold, image.cols - 2*edgeThreshold, image.rows - 2*edgeThreshold);
    croppedImage = image(crop);
  }

  for (int iii = 0; iii < rows; iii++)
  {
    for (int jjj = 0; jjj < cols; jjj++)
    {
      int grid[2] {iii, jjj};
      cv::Size imgSize = cv::Size(croppedImage.cols/cols,croppedImage.rows/rows);
      cv::Mat patch = gridBasedFeatures(croppedImage, grid, imgSize);
      std::vector< cv::KeyPoint > tempkeys = featuresAdaptiveThreshold(patch);
      keypointsGrids.push_back(tempkeys);
      std::for_each(tempkeys.begin(),tempkeys.end(), [&](cv::KeyPoint &n){n.pt.x +=jjj*imgSize.width + edgeThreshold;
                                                                          n.pt.y +=iii*imgSize.height + edgeThreshold;});
      indicesOfGrids.push_back(keypoints.size());

      keypoints.insert(keypoints.end(),tempkeys.begin(),tempkeys.end());
      // if (iii == 2 && jjj == 5)
      // {
      //   for (auto key:keypoints)
      //   {
      //     std::cout << key.pt << "  ";
      //   }
      //   std::cout << "\nrighttttttttttttttt\n";
      // }
    }
    
  }
  indicesOfGrids.push_back(keypoints.size());
  // std::cout << "NEW IMAGE \n";
  // if (left)
  // {
  //   cv::Mat trial;
  //   cv::drawKeypoints(image, keypoints,trial, cv::Scalar(255,0,0,255));
  //   std::cout << " NEW IMAGE \n";
  //   cv_bridge::CvImage out_msg;
  //   out_msg.header   = header; // Same timestamp and tf frame as input image
  //   out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
  //   out_msg.image    = trial; // Your cv::Mat
  //   mImageMatches.publish(out_msg.toImageMsg());
  // }
  
  
}

int Features::updateFeatureRemoval(std::vector < int > indexes, std::vector < cv::Point2f >& points, int& count, bool& first)
{
  int minInd = findMinimumResponse(indexes);
  std::sort(indexes.begin(), indexes.end(),[](int a, int b) {return a > b;});
  for (auto ind : indexes)
  {
    if (ind == 0 && !first)
      break;
    if (ind == 0 && first)
      first = false;
    if (ind != minInd)
    {
      keypoints.erase(keypoints.begin() + ind);
      points.erase(points.begin() + ind);
      if (count > ind)
      {
        count --;
      }
    }
    else
    {
      keypoints.push_back(keypoints[ind]);
      keypoints.erase(keypoints.begin() + ind + 1);
    }
  }
  return 0;
}

void Features::removeClosestNeighbors(std::vector < bool >& removed)
{
  for (int iii = keypoints.size()-1; iii >= 0; iii--)
  {
    if (removed[iii])
      keypoints.erase(keypoints.begin()+iii);
  }
  
}

int Features::findMinimumResponse(std::vector < int > indexes)
{
  int min = indexes[0];
  int count = 0;
  for (auto ind : indexes)
  {
    if (ind == 0 && count != 0)
      break;
    if (keypoints[min].response > keypoints[ind].response)
      min = ind;
    count ++;
  }
  return min;
}

void Features::sortFeaturesKdTree()
{
  PointsWithIndexes featurePoints;
  cv::Mat_<float> features(0,2);
  int count = 0;
  std::cout << "size before : " << keypoints.size() << '\n';
  for (auto key:keypoints)
  {
    featurePoints.points.push_back(cv::Point2f(key.pt.x,key.pt.y));
    featurePoints.removed.push_back(false);

    cv::Mat row = (cv::Mat_<float>(1, 2) << key.pt.x, key.pt.y);
    features.push_back(row);
    count ++;
  }
  
  cvflann::KDTreeIndexParams indexParams;
  const cvflann::SearchParams params(32);
  cv::flann::GenericIndex<cvflann::L2<float> >* kdtrees;
  int keySize = keypoints.size();
  count = 0;
  bool first = true;
  while (count < keypoints.size())
  {
    auto kdtree = cv::flann::GenericIndex<cvflann::L2<float> >(cv::Mat(featurePoints.points).reshape(1), cvflann::KDTreeIndexParams {10 },cvflann::L2<float> {});
    int numOfPoints = 8;
    std::vector < int > indices(numOfPoints);
    std::vector < float > dists(numOfPoints);
    double radius = 10.0f;
    kdtree.radiusSearch({featurePoints.points[count].x, featurePoints.points[count].y }, indices, dists, radius * radius, cvflann::SearchParams {numOfPoints});
    if (indices.size()>0)
    {
      updateFeatureRemoval(indices, featurePoints.points, count, first);

    }
    count ++;
  }
  std::cout << "size after : " << keypoints.size() << '\n';


}

cv::Mat Features::gridBasedFeatures(cv::Mat croppedImage, const int grid[2], cv::Size imgSize)
{
  cv::Rect crop(grid[1]*imgSize.width, grid[0]*imgSize.height, imgSize.width, imgSize.height);
  cv::Mat patch =  croppedImage(crop);
  return patch;
}
  

void Features::findORBFeatures(cv::Mat& image, std::vector< cv::KeyPoint >& keypoints, int numbOfFeatures, int edgeThreshold, int fastThreshold)
{
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(numbOfFeatures,1.2f,8, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE,30, fastThreshold);

  detector->detect(image, keypoints,cv::Mat());
}

cv::Mat FeatureDrawer::featurePosition(std::vector < cv::Point2f>& pointsL, std::vector < cv::Point2f>& pointsR, std::vector<bool>& left, std::vector<bool>& right)
{
  cv::Mat points4D(4,1,CV_32F);
  std::vector < cv::Point2f> pL, pR;
  for (size_t iii = 0; iii < pointsL.size(); iii++)
  {
    if (left[iii] && right[iii])
    {
      pL.push_back(pointsL[iii]);
      pR.push_back(pointsR[iii]);
    }
  }
  pointsL = pL;
  pointsR = pR;
  // std::cout << "l size : " << pointsL.size() << " r size : " << pointsR.size() << '\n';
  if (pointsL.size() >= 4)
  {

    cv::triangulatePoints(P1, P2, pointsL, pointsR, points4D);
    cv::Mat points3D(3, points4D.cols,CV_32F);
    leftImage.close.clear();
    for (size_t i = 0; i < points4D.cols; i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        points3D.at<float>(j,i) = points4D.at<float>(j,i)/points4D.at<float>(3,i);
        
      }
      if (abs(points3D.at<float>(2,i)) > zedcamera->mBaseline*40)
      {
        leftImage.close.push_back(false);
      }
      else
      {
        leftImage.close.push_back(true);
      }
    }
    return points3D;
  }
  else
    return cv::Mat();
  
}

void Features::removeOutliersOpticalFlow(std::vector < cv::Point2f>& pointL, std::vector < cv::Point2f>& pointpL, cv::Mat status)
{
  std::vector < cv::Point2f> inliersL, inlierspL;
  for(size_t i = 0; i < pointpL.size(); i++) 
  {
    if (status.at<bool>(i))
    {
      statusOfKeys.push_back(true);
      inliersL.push_back(pointL[i]);
      inlierspL.push_back(pointpL[i]);
    }
    else
    {
      statusOfKeys.push_back(false);
    }
  }
  pointL = inliersL;
  pointpL = inlierspL;
}

std::vector < cv::Point2f> Features::opticalFlow(Features& prevImage, image_transport::Publisher& mImageMatches, bool left)
{
  cv::Mat status, err;
  std::vector < cv::Point2f> pointL, pointpL;
  // std::for_each(leftImage.keypoints.begin(), leftImage.keypoints.end(),[&](cv::KeyPoint &n){pointL.push_back(n.pt);});
  std::for_each(prevImage.keypointsLR.begin(), prevImage.keypointsLR.end(),[&](cv::KeyPoint &n){pointpL.push_back(n.pt);});
  // pointpL = prevImage.inlierPoints;
  cv::calcOpticalFlowPyrLK(prevImage.image, image, pointpL, pointL, status, err);
  cv::Mat optFlow = image.clone();
  std::cout << "prev size : " << pointpL.size() << " new size : " << pointL.size() << '\n';
  // std::cout << err.at<float>(j) << "prev point : " << pointpL[j] << "new point : " << pointL[j] << '\n';
  removeOutliersOpticalFlow(pointL, pointpL, status);
  for(int j=0; j<pointpL.size(); j++)
  {
    cv::line(optFlow,pointpL[j],pointL[j],cv::Scalar(255,0,0,255));
  }
  if (!left)
  {
    cv_bridge::CvImage out_msg;
    out_msg.header   = header; // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
    out_msg.image    = optFlow; // Your cv::Mat
    mImageMatches.publish(out_msg.toImageMsg());
  }
  prevImage.inlierPoints = pointpL;
  return pointL;
}

void FeatureDrawer::drawFeatureMatches(const std::vector<cv::DMatch>& matches, const Features& firstImage, const Features& secondImage)
{
  cv::Mat img_matches;
  drawMatches( firstImage.image, firstImage.keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv_bridge::CvImage out_msg;
  out_msg.header   = firstImage.header; // Same timestamp and tf frame as input image
  out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
  out_msg.image    = img_matches; // Your cv::Mat
  mImageMatches.publish(out_msg.toImageMsg());
}

std::vector< cv::DMatch > FeatureDrawer::loweRatioTest(std::vector< std::vector<cv::DMatch> >& knnmatches)
{
  std::vector< cv::DMatch > matches;
  for(size_t i = 0; i < knnmatches.size(); i++) 
  {
    if(knnmatches[i].size() >= 2)
    {
      cv::DMatch first = knnmatches[i][0];
      float dist1 = knnmatches[i][0].distance;
      float dist2 = knnmatches[i][1].distance;

      if(dist1 < 0.8f * dist2) 
      {
        matches.push_back(knnmatches[i][0]);
      }
    }

  }
  return matches;
}

std::vector< cv::DMatch > FeatureDrawer::knnMatcher(const Features& firstImage, const Features& secondImage, const bool LR)
{
  std::vector< std::vector<cv::DMatch> > knnmatches;
  std::vector< cv::DMatch > good_matches;
  // cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  // cv::BFMatcher matcher(cv::NORM_HAMMING, true);  
  auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  matcher->knnMatch(firstImage.descriptors, secondImage.descriptors, knnmatches, 2);
  // std::cout << "size : " << firstImage.keypoints.size() << '\n';
  // std::cout << "size : " << secondImage.keypoints.size() << '\n';
  std::cout << "matches size : " << knnmatches.size() << '\n';
  std::vector< cv::DMatch > matches = loweRatioTest(knnmatches);
  if (LR)
  {
    good_matches = removeOutliersStereoMatch(matches, firstImage, secondImage);
  }
  else
  {
    good_matches = removeOutliersHomography(matches, firstImage, secondImage);
    std::cout << "good matches size : " << good_matches.size() << '\n';
    drawFeatureMatches(good_matches, firstImage, secondImage);
  }
  
  return good_matches;
}

void FeatureDrawer::positionOfMatchedFeatures(const std::vector<cv::DMatch>& matches, Features& leftImage, const Features& rightImage, const Features& previousLeftImage, const Features& previousRightImage)
{
  cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  // cv::BFMatcher matcher(cv::NORM_HAMMING);  
  std::vector < cv::KeyPoint > matchedKeypoints, matchedPreviousKeypoints;
  for (auto m:matches)
  {
    matchedKeypoints.push_back(leftImage.keypoints[m.queryIdx]);
  }

  // USE MATCHEDKEYPOINTS get descriptors and get left only with matches that are on both images
  cv::Mat ldisc, prevldisc;
  leftImage.getDescriptors(leftImage.image, matchedKeypoints, ldisc);
  std::vector< std::vector<cv::DMatch> > knnmatches;
  matcher.knnMatch(ldisc, rightImage.descriptors, knnmatches, 2);
  std::vector< cv::DMatch > matchesLR = loweRatioTest(knnmatches);
  int count = 0;
  for (auto m:matches)
  {
    matchedPreviousKeypoints.push_back(previousLeftImage.keypoints[m.trainIdx]);
    count ++;
  }

  leftImage.getDescriptors(previousLeftImage.image, matchedPreviousKeypoints, ldisc);
  matcher.knnMatch(prevldisc, previousRightImage.descriptors, knnmatches, 2);
  std::vector< cv::DMatch > matchesPrev = loweRatioTest(knnmatches);

  // std::vector< cv::DMatch > matches, good_matches;
}

std::vector< cv::DMatch > FeatureDrawer::removeOutliersHomography(const std::vector< cv::DMatch >& matches, const Features& firstImage, const Features& secondImage)
{
  std::vector<cv::KeyPoint> matched1, matched2;
  std::vector<cv::Point2f> pointl, pointr;
  for (auto m:matches)
  {
    pointl.push_back(firstImage.keypoints[m.queryIdx].pt);
    pointr.push_back(secondImage.keypoints[m.trainIdx].pt);
  }
  std::vector<cv::KeyPoint> inliers1, inliers2;
  std::vector< cv::DMatch > good_matches;
  cv::Mat h = findHomography( pointl, pointr, cv::RANSAC);
  if (h.rows == 3)
  {
    for (auto m:matches)
    {
      cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
      col.at<double>(0) = firstImage.keypoints[m.queryIdx].pt.x;
      col.at<double>(1) = firstImage.keypoints[m.queryIdx].pt.y;
      col = h * col;
      col /= col.at<double>(2);
      double dist = sqrt( pow(col.at<double>(0) - secondImage.keypoints[m.trainIdx].pt.x, 2) +
                          pow(col.at<double>(1) - secondImage.keypoints[m.trainIdx].pt.y, 2));
      if(dist < 2.5f) 
        good_matches.push_back(m);
    }
  }
  else
    good_matches = matches;
  
  return good_matches;
}

std::vector< cv::DMatch > FeatureDrawer::removeOutliersStereoMatch(const std::vector< cv::DMatch >& matches, const Features& leftImage, const Features& rightImage)
{
  std::vector< cv::DMatch > good_matches;
  for (auto m:matches)
  {
    if (abs(leftImage.keypoints[m.queryIdx].pt.y - rightImage.keypoints[m.trainIdx].pt.y) < 4)
      good_matches.push_back(m);
  }
  return good_matches;
}

std::vector<cv::DMatch> FeatureDrawer::findMatches(Features& firstImage, Features& secondImage, bool LR)
{
  firstImage.getDescriptors(firstImage.image, firstImage.keypoints, firstImage.descriptors);
  secondImage.getDescriptors(secondImage.image, secondImage.keypoints, secondImage.descriptors);

  if ( firstImage.descriptors.empty() )
    cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
  if ( secondImage.descriptors.empty() )
    cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
  std::vector< std::vector<cv::DMatch> > matches;
  cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  // cv::BFMatcher matcher(cv::NORM_HAMMING);  
  matcher.knnMatch(firstImage.descriptors, secondImage.descriptors, matches, 2);
  // std::cout << "keys size : " << keypoints.size() << '\n';
  // std::cout << "descriptors size : " << descriptors.rows <<  " " << descriptors.cols << '\n';
  std::vector<cv::KeyPoint> matched1, matched2;
  std::vector<cv::Point2f> pointl, pointr;
  for(size_t i = 0; i < matches.size(); i++) 
  {
    if(matches[i].size() >= 2)
    {
      cv::DMatch first = matches[i][0];
      float dist1 = matches[i][0].distance;
      float dist2 = matches[i][1].distance;

      if(dist1 < 0.8f * dist2) 
      {
        matched1.push_back(firstImage.keypoints[first.queryIdx]);
        matched2.push_back(secondImage.keypoints[first.trainIdx]);
        pointl.push_back(firstImage.keypoints[first.queryIdx].pt);
        pointr.push_back(secondImage.keypoints[first.trainIdx].pt);
      }
    }

  }
  std::vector<cv::DMatch> good_matches;
  if (pointl.size() > 4)
  {
    std::vector<cv::KeyPoint> inliers1, inliers2;
    cv::Mat h = findHomography( pointl, pointr, cv::RANSAC);
    if (h.rows == 3)
    {
      for(size_t i = 0; i < matched1.size(); i++) 
      {
        
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
            // std::cout << "first : " << matched1[i].pt << " second : " << matched2[i].pt << '\n';
            good_matches.push_back(cv::DMatch(new_i, new_i, 0));
        }
      
      }
      firstImage.keypointsLR = inliers1;
      secondImage.keypointsLR = inliers2;
      firstImage.keypoints = inliers1;
      secondImage.keypoints = inliers2;
      cv::Mat img_matches;
      cv_bridge::CvImage out_msg;
      drawMatches( firstImage.image, inliers1, secondImage.image, inliers2, good_matches, img_matches, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      if(LR)
      {
        // out_msg.header   = header; // Same timestamp and tf frame as input image
        // out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
        // out_msg.image    = img_matches; // Your cv::Mat
        // mImageMatches.publish(out_msg.toImageMsg());
      }
    }
  }
    
    
  

  return good_matches;

}

std::vector<cv::DMatch> FeatureDrawer::matchEachGrid(Features& firstImage, Features& secondImage, int row, int col, int rows, int cols, bool LR)
{
  std::vector< std::vector < cv::DMatch > > knnmatches;
  cv::Mat firDesc, secDesc;
  if (LR)
  {
    if (col > 0)
    {
      secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col - 1],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
    }
    else
    {
      secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
    }
  }
  firDesc = firstImage.descriptors(cv::Range(firstImage.indicesOfGrids[row*cols + col],firstImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
  // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
  // std::cout << "  row*cols + col - 1 : " << row*cols + col - 1 << " row : " << row << " col :  " << col << '\n';
  // std::cout << " secDesc    rows : " << secDesc.rows << " cols : " << secDesc.cols << '\n';
  // std::cout << " secDesc    rows : " << firstImage.descriptors.rows << " cols : " << firstImage.descriptors.cols << '\n';
  // std::cout << " firDesc    rows : " << firDesc.rows << " cols : " << firDesc.cols << '\n';
  std::vector< cv::DMatch > matches;
  if (secDesc.rows != 0 && firDesc.rows != 0)
  {
    cv::Mat mask = cv::Mat::ones(firDesc.rows, secDesc.rows, CV_8U);
    if (featuresMatched.size() > 0)
    {
      for (int it:featuresMatched)
      {
        // std::cout << "it : " << it << " rows : " << mask.rows << " cols : " << mask.cols << '\n';
        mask.col(it) = 0;
      }
    }
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    // cv::Mat mask = cv::Mat::zeros(firDesc.rows, secDesc.rows, CV_8U);
    // std::cout << " mask    rows : " << mask.rows << " cols : " << mask.cols << '\n';

    // mask(cv::Rect(0,0,2,firDesc.rows)) = 1;
    // matcher->knnMatch(firDesc, secDesc, knnmatches, 2, mask, true);
    matcher->knnMatch(firDesc, secDesc, knnmatches, 2, mask, true);
    matches = loweRatioTest(knnmatches);
    // cv::Mat trial = cv::Mat::zeros(10, 15, CV_8U);
    // trial.col(2) = 1;
    // trial(cv::Rect(cv::Point(1,2), cv::Point(5,7))) = 1;
    // std::cout << "trial " << trial << '\n';

  }
  return matches;
}

std::vector<cv::DMatch> FeatureDrawer::matchWithGridsUsingMask(Features& firstImage, Features& secondImage, int row, int col, int rows, int cols, bool LR)
{
  std::vector< cv::DMatch > matches;
  cv::Mat mask = cv::Mat::zeros(firstImage.descriptors.rows, secondImage.descriptors.rows, CV_8U);
  cv::Mat trial = cv::Mat::zeros(10, 15, CV_8U);
  trial.col(2) = 1;
  // trial(cv::Rect(cv::Point(1,2), cv::Point(5,7))) = 1;
  std::cout << "trial " << trial << '\n';
  if (col > 0)
  {
    cv::Point p1(secondImage.indicesOfGrids[row*cols + col - 1], firstImage.indicesOfGrids[row*cols + col]);
    cv::Point p2(secondImage.indicesOfGrids[row*cols + col + 1], firstImage.indicesOfGrids[row*cols + col + 1]);
    // std::cout << " p1 " << p1 << " p2 " << p2 << " rows " << mask.rows << " cols " << mask.cols << '\n';
    mask(cv::Rect(p1,p2)) = 1;
  }
  else
  {
    cv::Point p1(secondImage.indicesOfGrids[row*cols + col], firstImage.indicesOfGrids[row*cols + col]);
    cv::Point p2(secondImage.indicesOfGrids[row*cols + col + 1], firstImage.indicesOfGrids[row*cols + col + 1]);
    // std::cout << " p1 " << p1 << " p2 " << p2 << " rows " << mask.rows << " cols " << mask.cols << '\n';
    mask(cv::Rect(p1,p2)) = 1;
    
  }
  auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  std::vector< std::vector < cv::DMatch > > knnmatches;
  matcher->knnMatch(firstImage.descriptors, secondImage.descriptors, knnmatches, 2, mask, true);
  matches = loweRatioTest(knnmatches);
  return matches;
}

std::vector<cv::DMatch> FeatureDrawer::matchesWithGrids(Features& firstImage, Features& secondImage, int rows, int cols, bool LR)
{
  std::vector < cv::DMatch > allMatchesOfGrids;
  for (size_t iii = 0; iii < rows; iii++)
  {
    for (size_t jjj = 0; jjj < cols; jjj++)
    {
      std::vector < cv::DMatch > matches = matchEachGrid(firstImage, secondImage, iii, jjj, rows, cols, LR);
      // std::cout << "matches size : " << matches.size() << "\n";
      if (jjj > 0)
      {
        std::for_each(matches.begin(),matches.end(), [&](cv::DMatch &n){n.queryIdx += firstImage.indicesOfGrids[iii*cols + jjj];
                                                                        n.trainIdx += secondImage.indicesOfGrids[iii*cols + jjj-1];});
      }
      else
      {
        std::for_each(matches.begin(),matches.end(), [&](cv::DMatch &n){n.queryIdx += firstImage.indicesOfGrids[iii*cols + jjj];
                                                                        n.trainIdx += secondImage.indicesOfGrids[iii*cols + jjj];});
      }
      std::vector < cv::DMatch > goodMatches = removeOutliersStereoMatch(matches, firstImage, secondImage);
      featuresMatched.clear();
      if (jjj < (cols - 1))
      {
        for (auto m:goodMatches)
        {
          if (m.trainIdx > secondImage.indicesOfGrids[iii*cols + jjj])
          {
            featuresMatched.push_back(m.trainIdx - secondImage.indicesOfGrids[iii*cols + jjj]);
          }
        } 
      }
      
      allMatchesOfGrids.insert(allMatchesOfGrids.end(), goodMatches.begin(), goodMatches.end());
    }
  }
  // std::cout << "matches before : " << allMatchesOfGrids.size() << '\n';
  // std::vector < cv::DMatch > goodMatches = removeOutliersStereoMatch(allMatchesOfGrids, firstImage, secondImage);
  // std::cout << "f size : " << firstImage.keypoints.size() << " s size : " << secondImage.keypoints.size()  << " matches size : " << allMatchesOfGrids.size() << '\n';
  // drawFeatureMatches(allMatchesOfGrids, firstImage, secondImage);
  return allMatchesOfGrids;
}

void FeatureDrawer::addMatRows(std::vector < cv::Mat >& descriptorsGrids, int index, cv::Mat& descriptor)
{
  if (descriptorsGrids[index].rows != 0)
    descriptor.push_back(descriptorsGrids[index]);
}

void FeatureDrawer::addIndexToMatch(int row, int col, int rows, int cols, std::vector <int>& indices, cv::DMatch& m)
{
  if (row == 0)
  {
    if (col == 0)
    {

    }
  }
}


std::vector<cv::DMatch> FeatureDrawer::matchesWithGridsPrev(Features& firstImage, Features& secondImage, int rows, int cols, bool LR)
{
  std::vector < cv::DMatch > allMatchesOfGrids;
  for (size_t iii = 0; iii < rows; iii++)
  {
    for (size_t jjj = 0; jjj < cols; jjj++)
    {
      if (jjj == 0 && iii == 0)
      {
        featuresMatched.clear();
      }
      std::vector < cv::DMatch > matches = matchEachGridPrev(firstImage, secondImage, iii, jjj, rows, cols, LR);
      // std::cout << "matches size : " << matches.size() << "\n";
      if (jjj != 0)
      {
        std::for_each(matches.begin(),matches.end(), [&](cv::DMatch &n){n.queryIdx += firstImage.indicesOfGrids[iii*cols + jjj];});
      }
      std::vector < cv::DMatch > goodMatches = removeOutliersStereoMatch(matches, firstImage, secondImage);
      
      for (auto m:goodMatches)
      {
        featuresMatched.push_back(m.trainIdx);
      } 
      
      allMatchesOfGrids.insert(allMatchesOfGrids.end(), goodMatches.begin(), goodMatches.end());
    }
  }
  // std::cout << "matches before : " << allMatchesOfGrids.size() << '\n';
  // std::vector < cv::DMatch > goodMatches = removeOutliersStereoMatch(allMatchesOfGrids, firstImage, secondImage);
  // std::cout << "f size : " << firstImage.keypoints.size() << " s size : " << secondImage.keypoints.size()  << " matches size : " << allMatchesOfGrids.size() << '\n';
  drawFeatureMatches(allMatchesOfGrids, firstImage, secondImage);
  return allMatchesOfGrids;
}

std::vector<cv::DMatch> FeatureDrawer::matchEachGridPrev(Features& firstImage, Features& secondImage, int row, int col, int rows, int cols, bool LR)
{
  std::vector< std::vector < cv::DMatch > > knnmatches;
  cv::Mat firDesc, secDesc;

  if (LR)
  {
    if (row == 0)
    {
      if (col == 0)
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[0],
        //                       secondImage.descriptorsGrids[1],
        //                       secondImage.descriptorsGrids[cols],
        //                       secondImage.descriptorsGrids[cols + 1]};
        // cv::vconcat(matArray, 4, secDesc);
        secDesc.push_back(secondImage.descriptorsGrids[0]);
        secDesc.push_back(secondImage.descriptorsGrids[1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[cols] - secondImage.indicesOfGrids[1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[cols]);
        secDesc.push_back(secondImage.descriptorsGrids[cols + 1]);
        // std::cout << " rows : " << secondImage.descriptorsGrids[0].rows << " " << secondImage.descriptorsGrids[1].rows << " " << secondImage.descriptorsGrids[cols].rows << " " << secondImage.descriptorsGrids[cols + 1].rows << " yes rows : " << secDesc.rows << "\n";
      }
      else if (col == (cols - 1))
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[cols - 2],
        //                       secondImage.descriptorsGrids[cols - 1],
        //                       secondImage.descriptorsGrids[2*cols - 2],
        //                       secondImage.descriptorsGrids[2*cols - 1]};
        // cv::vconcat(matArray, 4, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[cols - 2] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[cols - 2]);
        secDesc.push_back(secondImage.descriptorsGrids[cols - 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[2*cols - 2] - secondImage.indicesOfGrids[cols - 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[2*cols - 2]);
        secDesc.push_back(secondImage.descriptorsGrids[2*cols - 1]);
      }
      else
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[row*cols + col - 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col + 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col + 1]};
        // cv::vconcat(matArray, 6, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row + 1)*cols + col - 1] - secondImage.indicesOfGrids[row*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col + 1]);
      }
    }
    else if (row == (rows - 1))
    {
      if (col == 0)
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col + 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col + 1]};
        // cv::vconcat(matArray, 4, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
      }
      else if (col == (cols - 1))
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col - 1],
        //                       secondImage.descriptorsGrids[row*cols + col]};
        // cv::vconcat(matArray, 4, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
      }
      else
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col + 1],
        //                       secondImage.descriptorsGrids[row*cols + col - 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col + 1]};
        // cv::vconcat(matArray, 6, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
      }
    }
    else
    {
      if (col ==0)
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col + 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col + 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col + 1]};
        // cv::vconcat(matArray, 6, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row + 1)*cols + col] - secondImage.indicesOfGrids[row*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col + 1]);
      }
      else if (col == (cols - 1))
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col - 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col]};
        // cv::vconcat(matArray, 6, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row + 1)*cols + col - 1] - secondImage.indicesOfGrids[row*cols + col + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col]);
      }
      else
      {
        // cv::Mat matArray[] = {secondImage.descriptorsGrids[(row - 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row - 1)*cols + col + 1],
        //                       secondImage.descriptorsGrids[row*cols + col - 1],
        //                       secondImage.descriptorsGrids[row*cols + col],
        //                       secondImage.descriptorsGrids[row*cols + col + 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col - 1],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col],
        //                       secondImage.descriptorsGrids[(row + 1)*cols + col + 1]};
        // cv::vconcat(matArray, 9, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        addMatRows(secondImage.descriptorsGrids, (row - 1)*cols + col - 1, secDesc);
        addMatRows(secondImage.descriptorsGrids, (row - 1)*cols + col, secDesc);
        addMatRows(secondImage.descriptorsGrids, (row - 1)*cols + col + 1, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        addMatRows(secondImage.descriptorsGrids, row*cols + col - 1, secDesc);
        addMatRows(secondImage.descriptorsGrids, row*cols + col, secDesc);
        addMatRows(secondImage.descriptorsGrids, row*cols + col + 1, secDesc);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row + 1)*cols + col - 1] - secondImage.indicesOfGrids[row*cols + col + 1 + 1]),CV_8U));
        addMatRows(secondImage.descriptorsGrids, (row + 1)*cols + col - 1, secDesc);
        addMatRows(secondImage.descriptorsGrids, (row + 1)*cols + col, secDesc);
        addMatRows(secondImage.descriptorsGrids, (row + 1)*cols + col + 1, secDesc);
        // std::cout << " secDesc    rows : " << secDesc.rows << " real rows : " << secondImage.descriptors.rows << '\n';
        
      }
    }
  }
  // std::cout << " row : " << row << " col :  " << col << '\n';
  firDesc = firstImage.descriptorsGrids[row*cols + col].clone();
  // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
  // std::cout << " secDesc    rows : " << firstImage.descriptors.rows << " cols : " << firstImage.descriptors.cols << '\n';
  // std::cout << " firDesc    rows : " << firDesc.rows << " cols : " << firDesc.cols << '\n';
  std::vector< cv::DMatch > matches;
  if (secDesc.rows != 0 && firDesc.rows != 0)
  {
    cv::Mat mask = cv::Mat::ones(firDesc.rows, secDesc.rows, CV_8U);
    // std::cout << " rows : " << mask.rows << " cols : " << mask.cols << '\n';
    if (featuresMatched.size() > 0)
    {
      for (int it:featuresMatched)
      {
        if (it < mask.cols)
          mask.col(it) = 0;
      }
    }
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    // cv::Mat mask = cv::Mat::zeros(firDesc.rows, secDesc.rows, CV_8U);
    // std::cout << " mask    rows : " << mask.rows << " cols : " << mask.cols << '\n';

    // mask(cv::Rect(0,0,2,firDesc.rows)) = 1;
    // matcher->knnMatch(firDesc, secDesc, knnmatches, 2, mask, true);
    matcher->knnMatch(firDesc, secDesc, knnmatches, 2, mask, true);
    matches = loweRatioTest(knnmatches);
    // cv::Mat trial = cv::Mat::zeros(10, 15, CV_8U);
    // trial.col(2) = 1;
    // trial(cv::Rect(cv::Point(1,2), cv::Point(5,7))) = 1;
    // std::cout << "trial " << trial << '\n';

  }
  return matches;
}

void FeatureDrawer::ceresSolver(std::vector<cv::DMatch>& matches, const cv::Mat& points3D, const cv::Mat& prevpoints3D)
{
  ceres::Problem problem;
  ceres::LossFunction* lossfunction = NULL;
  int count = 0;
  for (size_t iii = 0; iii < matches.size(); iii++)
  {
    for (size_t jjj = 0; jjj < previousleftKeypoints.size(); jjj++)
    {
      if (leftImage.close[matches[iii].queryIdx] && previousLeftImage.close[matches[iii].trainIdx] && previousleftKeypoints[jjj].pt == previousLeftImage.keypoints[matches[iii].trainIdx].pt)
      {
        double x = points3D.at<double>(0,matches[iii].queryIdx);
        double y = points3D.at<double>(1,matches[iii].queryIdx);
        double z = points3D.at<double>(2,matches[iii].queryIdx);
        double xp = prevpoints3D.at<double>(0,jjj);
        double yp = prevpoints3D.at<double>(1,jjj);
        double zp = prevpoints3D.at<double>(2,jjj);
        // std::cout << "PREVIOUS : " <<  xp << ' ' << yp  << " " << zp << '\n';
        // std::cout << "OBSERVED : " <<  x << ' ' << y  << " " << z << '\n';
        // std::cout << "x : " << x << " p3d : " << p3d.x() << '\n';
        Eigen::Vector3d p3d(x, y, z);
        Eigen::Vector3d pp3d(xp, yp, zp);
        ceres::CostFunction* costfunction = Reprojection3dError::Create(pp3d, p3d);
        problem.AddResidualBlock(costfunction, lossfunction, camera);
        count ++;
        break;
      }
    }
    
    
  }
  std::cout << "Count : " << count << '\n';
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
  
}

void FeatureDrawer::again()
{
  // Get Features of Each Image
  int rows {6};
  int cols {8};
  bool LR {true};

  leftImage.getFeatures(rows, cols, mImageMatches, true);
  rightImage.getFeatures(rows, cols, mImageMatches, false);

  // leftImage.sortFeaturesKdTree();
  // rightImage.sortFeaturesKdTree();
  leftImage.getDescriptors(leftImage.image, leftImage.keypoints, leftImage.descriptors);
  rightImage.getDescriptors(rightImage.image, rightImage.keypoints, rightImage.descriptors);
  std::vector < cv::DMatch > goodMatches = matchesWithGrids(leftImage, rightImage,rows,cols,true);
  if (!firstImage)
  {
    std::vector < cv::DMatch > goodMatchesPrev = matchesWithGridsPrev(previousLeftImage, leftImage,rows,cols,true);
    featuresMatched.clear();
    std::cout << "matches prev : " << goodMatchesPrev.size() << '\n';
  }

  // for (auto lel:leftImage.indicesOfGrids)
  // {
  //   std::cout << lel << "  ";
  // }
  // std::cout << "good matches size : " << goodMatches.size() << '\n';
  // if (!firstImage)
  //   knnMatcher(leftImage,previousLeftImage, false);
  // knnMatcher(leftImage, rightImage, true);
  // Get Descriptors of Each Image
  // std::vector<cv::DMatch> matches = matchesLR(leftImage, rightImage);

  // get feature position

  // TODO KMean clustering of features.

  // TODO Optical Flow for rotation? maybe

  // std::vector<cv::DMatch> matches = leftImage.getMatches(rightImage, mImageMatches, previousleftKeypoints, true);

  // cv::Mat points3D = calculateFeaturePosition(matches);
  // if (!firstImage)
  // {
  //   previousLeftImage.getDescriptors(previousLeftImage.image, previousLeftImage.keypoints, previousLeftImage.descriptors);
  //   std::vector<cv::KeyPoint> realleftkeypoints = leftImage.keypoints;
  //   std::vector<cv::DMatch> matchesLpL = leftImage.getMatches(previousLeftImage, mImageMatches, previousleftKeypoints, false);

  //   // std::vector<cv::DMatch> matches = previousLeftImage.getMatches(previousRightImage, mImageMatches, true);
  //   // pointsL = leftImage.opticalFlow(previousLeftImage, mImageMatches, true);
  //   // pointsR = rightImage.opticalFlow(previousRightImage, mImageMatches, false);
  //   // cv::Mat points3D = featurePosition(pointsL, pointsR, leftImage.statusOfKeys, rightImage.statusOfKeys);
  //   // cv::Mat prevPoints3D = featurePosition(previousLeftImage.inlierPoints, previousRightImage.inlierPoints, leftImage.statusOfKeys, rightImage.statusOfKeys);
  //   ceresSolver(matchesLpL, points3D, previouspoints3D);
  //   publishMovement();
  //   leftImage.keypoints = realleftkeypoints;
  // }

  // // cv::Mat points3D = calculateFeaturePosition(matches);
  // setPrevious(points3D);
  previousLeftImage.image = leftImage.image.clone();
  previousLeftImage.keypoints = leftImage.keypoints;
  previousLeftImage.indicesOfGrids = leftImage.indicesOfGrids;
  previousLeftImage.descriptorsGrids = leftImage.descriptorsGrids;
  previousLeftImage.keypointsGrids = leftImage.keypointsGrids;
  previousLeftImage.descriptors = leftImage.descriptors.clone(); 
  previousRightImage.image = rightImage.image.clone();
  previousRightImage.keypoints = rightImage.keypoints;
  previousRightImage.indicesOfGrids = rightImage.indicesOfGrids;
  previousRightImage.descriptorsGrids = rightImage.descriptorsGrids;
  previousRightImage.keypointsGrids = rightImage.keypointsGrids;
  previousRightImage.descriptors = rightImage.descriptors.clone(); 
  leftImage.clearFeatures();
  rightImage.clearFeatures();
  firstImage = false;
}

void Features::clearFeatures()
{
  keypoints.clear();
  indicesOfGrids.clear();
  keypointsGrids.clear();
  descriptorsGrids.clear();
}

void FeatureDrawer::featureDetectionCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
      if (!zedcamera->rectified)
      {
        leftImage.setImage(lIm);
        rightImage.setImage(rIm);
        cv::remap(leftImage.image, leftImage.image, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
        cv::remap(rightImage.image, rightImage.image, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
        // std::cout << "MATRIX SAME : " << matIsEqual(left,leftImage.image) << '\n';
        // std::cout << "MATRIX SAME : " << matIsEqual(right,rightImage.image) << '\n';
      }
      else
      {
        leftImage.setImage(lIm);
        rightImage.setImage(rIm);
      }
      // allMatches(lIm->header);
      again();
      
      firstImage = false;
      count = 0;
}

void Features::findFeaturesTrial()
{
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

  detector->detect(image, keypoints,cv::Mat());
  cv::Mat data_pts = cv::Mat(keypoints.size(), 2, CV_64F);
  for (int i = 0; i < data_pts.rows; i++)
  {
      data_pts.at<double>(i, 0) = keypoints[i].pt.x;
      data_pts.at<double>(i, 1) = keypoints[i].pt.y;
  }
  cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Mat normkeypoints = pca_analysis.project(data_pts);
  std::cout << "norm keys : " << normkeypoints.at<double>(0,0) << " " << normkeypoints.at<double>(0,1) << "\n";
  std::cout << "keypoint : " << keypoints[0].pt << '\n';
  std::cout << "mean : " << pca_analysis.mean << '\n';
  cv::Mat back = pca_analysis.backProject(normkeypoints);
  std::cout << "back keys : " << back.at<double>(0,0) << " " << back.at<double>(0,1) << "\n";
  std::cout << "image size : " << image.size()  << "  " << image.cols << '\n';
  // printMat(normkeypoints);
  // cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
  // std::cout << "center : " << cntr << '\n';
  detector->compute(image, keypoints, descriptors);

}

void FeatureDrawer::setUndistortMap(ros::NodeHandle *nh)
{
    cv::Size imgSize = cv::Size(zedcamera->mWidth, zedcamera->mHeight);
    cv::stereoRectify(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, imgSize, zedcamera->sensorsRotate, zedcamera->sensorsTranslate, R1, R2, P1, P2, Q);
    cv::Mat leftCamera = cv::getOptimalNewCameraMatrix(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs,imgSize, 0);
    cv::Mat rightCamera = cv::getOptimalNewCameraMatrix(zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs,imgSize, 0);
    cv::initUndistortRectifyMap(zedcamera->cameraLeft.cameraMatrix, zedcamera->cameraLeft.distCoeffs, R1, leftCamera, imgSize, CV_32FC1, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(zedcamera->cameraRight.cameraMatrix, zedcamera->cameraRight.distCoeffs, R2, rightCamera, imgSize, CV_32FC1, rmap[1][0], rmap[1][1]);
    std::cout << "P1 : \n";
    printMat(P1);
    std::cout << "P2 : \n";
    printMat(P2);
    
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
      pointsL.push_back(leftImage.keypointsLR[matches[i].queryIdx].pt);
      pointsR.push_back(rightImage.keypointsLR[matches[i].trainIdx].pt);
    }
    cv::triangulatePoints(P1, P2, pointsL, pointsR, points4D);
    // std::cout << "LEFT : " << leftImage.keypoints[matches[0].queryIdx].pt << '\n';
    // std::cout << "RIGHT : " << rightImage.keypoints[matches[0].trainIdx].pt << '\n';
    // std::cout << "points4D : " << " x :" << points4D.at<double>(0,0)/points4D.at<double>(3,0) << " y :" << points4D.at<double>(1,0)/points4D.at<double>(3,0) << " z :" << points4D.at<double>(2,0)/points4D.at<double>(3,0) << '\n';
    cv::Mat points3D(3, points4D.cols,CV_64F);
    leftImage.close.clear();
    for (size_t i = 0; i < points4D.cols; i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        points3D.at<double>(j,i) = points4D.at<double>(j,i)/points4D.at<double>(3,i);
        
      }
      if (abs(points3D.at<double>(2,i)) > zedcamera->mBaseline*40)
      {
        leftImage.close.push_back(false);
        // std::cout << "left : " << pointsL[i] <<'\n'; 
        // std::cout << "right : " << pointsR[i] <<'\n'; 
        // std::cout << "size points : " << points4D.cols <<'\n'; 
        // std::cout << "size leftkeys : " << pointsL.size() <<'\n'; 
        // std::cout << "size rightkeys : " << pointsR.size() <<'\n'; 
      }
      else
      {
        leftImage.close.push_back(true);
      }
    }
    return points3D;
  }
  return points4D;
}

void Features::findFeatures(bool LR)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // detect features and descriptor
    if (LR)
    {
      detector->detectAndCompute( image, cv::Mat(), keypoints, descriptors);
    }
    else
    {
      detector->compute(image, keypoints, descriptors);
    }
}

void FeatureDrawer::allMatches(const std_msgs::Header& header)
{
    bool LR = true;
    leftImage.findFeatures(LR);
    rightImage.findFeatures(LR);
    std::vector<cv::DMatch> matches = leftImage.findMatches(rightImage, header, mImageMatches, LR);
    cv::Mat points3D = calculateFeaturePosition(matches);
    std::vector<cv::KeyPoint> tempKeypoints = leftImage.keypoints;
    if (!firstImage)
    {
      LR = false;
      std::vector<cv::DMatch> matchesLpL = leftImage.findMatches(previousLeftImage, header, mImageMatches, LR);
      keepMatches(matches, matchesLpL, leftImage, tempKeypoints, points3D, header, true);
      publishMovement();
    }
    previousleftKeypoints = tempKeypoints;
    
    setPrevious(points3D);

}

void FeatureDrawer::keepMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& LpLmatches, const vio_slam::Features& secondImage, std::vector<cv::KeyPoint> tempKeypoints, const cv::Mat& points3D, const std_msgs::Header& header, bool left)
{
  cv::Mat trial = leftImage.image.clone();
  ceres::Problem problem;
  ceres::LossFunction* lossfunction = NULL;
  for (auto mLpL:LpLmatches)
  {
    for (auto mLR:matches)
    {
      if (leftImage.keypoints[mLpL.queryIdx].pt == tempKeypoints[mLR.queryIdx].pt)
      {
        int keypos = 0;
        for (auto keys:previousleftKeypoints)
        {
          if (previousLeftImage.keypoints[mLpL.trainIdx].pt == keys.pt && leftImage.close[mLR.queryIdx] && previousLeftImage.close[keypos])
          {
            double x = points3D.at<double>(0,mLR.queryIdx);
            double y = points3D.at<double>(1,mLR.queryIdx);
            double z = points3D.at<double>(2,mLR.queryIdx);
            double xp = previouspoints3D.at<double>(0,keypos);
            double yp = previouspoints3D.at<double>(1,keypos);
            double zp = previouspoints3D.at<double>(2,keypos);
            Eigen::Vector3d p3d(x, y, z);
            Eigen::Vector3d pp3d(xp, yp, zp);
            // cv::Point(leftImage.keypoints[mLR.queryIdx].pt);
            leftImage.keypoints[mLR.queryIdx].pt;
            // std::cout << "PREVIOUS : " <<  xp << ' ' << yp  << " " << zp << '\n';
            // std::cout << "OBSERVED : " <<  x << ' ' << y  << " " << z << '\n';
            // cv2.line(image, point1, point2, [0, 255, 0], 2) 
            cv::line(trial,cv::Point(leftImage.keypoints[mLR.queryIdx].pt.x,leftImage.keypoints[mLR.queryIdx].pt.y), cv::Point(previousleftKeypoints[keypos].pt.x, previousleftKeypoints[keypos].pt.y), (255,255,255));
            // std::cout << "leftImage : " <<  leftImage.keypoints[mLR.queryIdx].pt.x << " " << leftImage.keypoints[mLR.queryIdx].pt.y << '\n';
            // std::cout << "previousLeftImage : " <<  previousleftKeypoints[keypos].pt.x << " " << previousleftKeypoints[keypos].pt.y << '\n';
            ceres::CostFunction* costfunction = Reprojection3dError::Create(pp3d, p3d);
            problem.AddResidualBlock(costfunction, lossfunction, camera);
            
            break;
          }
          keypos ++;
        }
        break;
      }
    }
  }
  cv_bridge::CvImage out_msg;
  out_msg.header   = header; // Same timestamp and tf frame as input image
  out_msg.encoding = sensor_msgs::image_encodings::MONO8; // Or whatever
  out_msg.image    = trial; // Your cv::Mat
  mImageMatches.publish(out_msg.toImageMsg());
  // for (auto m:matches)
  // {
  //   std::cout << "LR matches : " << leftImage.keypoints[m.queryIdx].pt << '\n';
  // }
  // for (auto m:matches2)
  // {
  //   std::cout << "LpL matches : " << previousLeftImage.keypoints[m.trainIdx].pt << '\n';
  // }
  std::cout << "NEW IMAGE\n";
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
  
}

void FeatureDrawer::publishMovement()
{
  nav_msgs::Odometry position;
  if (abs(T(0,3)) < 100 && abs(T(1,3)) < 100 && abs(T(2,3)) < 100)
  {
    sumsMovement[0] += T(0,3);
    sumsMovement[1] += T(1,3);
    sumsMovement[2] += T(2,3);
    Eigen::Matrix3d Rot;
    previousT = previousT * T;
    Eigen::Quaterniond quat(previousT.topLeftCorner<3,3>());
    tf::poseTFToMsg(tf::Pose(tf::Quaternion(quat.x(),quat.y(),quat.z(),quat.w()),  tf::Vector3(previousT(0,3), previousT(1,3), previousT(2,3))), position.pose.pose); //Aria returns pose in mm.
    std::cout << "T : " << previousT << '\n';
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

    position.header.frame_id = leftImage.header.frame_id;
    position.header.stamp = ros::Time::now();
    pose_pub.publish(position);
  }
}

std::vector<cv::DMatch> Features::findMatches(Features& secondImage, const std_msgs::Header& header, image_transport::Publisher& mImageMatches, bool LR)
{

    // if (!LR)
    // {
    //   if (matIsEqual(image, secondImage.image))
    //   {
    //     std::cout << "Left and Prev Left image are the same\n";
    //   }
    //   else
    //   {
    //     std::cout << "Left and Prev NOT same\n";
    //   }
      
    // }
    if (!LR)
    {
      findFeatures(false);
      secondImage.findFeatures(true);
    }
    
    if ( descriptors.empty() )
      cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
    if ( secondImage.descriptors.empty() )
      cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
    std::vector< std::vector<cv::DMatch> > matches;
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // cv::BFMatcher matcher(cv::NORM_HAMMING);  
    matcher.knnMatch(descriptors, secondImage.descriptors, matches, 2);
    std::vector<cv::KeyPoint> matched1, matched2;
    std::vector<cv::Point2f> pointl, pointr;
    for(size_t i = 0; i < matches.size(); i++) 
    {
      if(matches[i].size() >= 2)
      {
        cv::DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;

        if(dist1 < 0.8f * dist2) 
        {

          matched1.push_back(keypoints[first.queryIdx]);
          matched2.push_back(secondImage.keypoints[first.trainIdx]);
          pointl.push_back(keypoints[first.queryIdx].pt);
          pointr.push_back(secondImage.keypoints[first.trainIdx].pt);
          if (!LR)
          {

          }
          
          
        }
      }

    }
    std::vector<cv::DMatch> good_matches;
    if (pointl.size() > 4)
    {
      std::vector<cv::KeyPoint> inliers1, inliers2;
      cv::Mat h = findHomography( pointl, pointr, cv::RANSAC);
      if (h.rows == 3)
      {
        for(size_t i = 0; i < matched1.size(); i++) 
        {
          
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
        keypoints = inliers1;
        secondImage.keypoints = inliers2;
        cv::Mat img_matches;
        cv_bridge::CvImage out_msg;
        drawMatches( image, inliers1, secondImage.image, inliers2, good_matches, img_matches, cv::Scalar::all(-1),
              cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        if(!LR)
        {
          // out_msg.header   = header; // Same timestamp and tf frame as input image
          // out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
          // out_msg.image    = img_matches; // Your cv::Mat
          // mImageMatches.publish(out_msg.toImageMsg());
        }
      }
    }
    
    
  

    return good_matches;
}

void FeatureDrawer::setPrevious(cv::Mat& points3D)
{

    // std::cout << "MATRICES EQUAL : " << matIsEqual(previousLeftImage.image, leftImage.image) << '\n';
    previousLeftImage.image = leftImage.image.clone();
    previousLeftImage.descriptors = leftImage.descriptors.clone();
    previousLeftImage.keypoints = leftImage.keypoints;
    previousLeftImage.keypointsLR = leftImage.keypointsLR;
    previousLeftImage.statusOfKeys = leftImage.statusOfKeys;
    previousLeftImage.close = leftImage.close;
    // std::cout << "MATRICES EQUAL AFTER : " << matIsEqual(previousLeftImage.image, leftImage.image) << '\n';
    previousRightImage.image = rightImage.image.clone();
    previousRightImage.descriptors = rightImage.descriptors.clone();
    previousRightImage.keypoints = rightImage.keypoints;
    previousRightImage.keypointsLR = rightImage.keypointsLR;
    previousRightImage.statusOfKeys = rightImage.statusOfKeys;
    previouspoints3D = points3D.clone();
    // previousLeftImage.close = leftImage.close;
}

void Features::setImage(const sensor_msgs::ImageConstPtr& imageRef)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(imageRef, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    cv::cvtColor(cv_ptr->image, image, cv::COLOR_BGR2GRAY);
    // image = cv_ptr->image.clone();
    header = cv_ptr->header;
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

} //namespace vio_slam