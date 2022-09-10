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

std::vector<cv::KeyPoint> ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints,
                         float tolerance, int cols, int rows) {
  // several temp expression variables to simplify solution equation
  int exp1 = rows + cols + 2 * numRetPoints;
  long long exp2 =
      ((long long)4 * cols + (long long)4 * numRetPoints +
       (long long)4 * rows * numRetPoints + (long long)rows * rows +
       (long long)cols * cols - (long long)2 * rows * cols +
       (long long)4 * rows * cols * numRetPoints);
  double exp3 = sqrt(exp2);
  double exp4 = numRetPoints - 1;

  double sol1 = -std::round((exp1 + exp3) / exp4); // first solution
  double sol2 = -std::round((exp1 - exp3) / exp4); // second solution

  // binary search range initialization with positive solution
  int high = (sol1 > sol2) ? sol1 : sol2;
  int low = std::floor(sqrt((double)keyPoints.size() / numRetPoints));
  low = std::max(1, low);

  int width;
  int prevWidth = -1;

  std::vector<int> ResultVec;
  bool complete = false;
  unsigned int K = numRetPoints;
  unsigned int Kmin = std::round(K - (K * tolerance));
  unsigned int Kmax = std::round(K + (K * tolerance));

  std::vector<int> result;
  result.reserve(keyPoints.size());
  while (!complete) {
    width = low + (high - low) / 2;
    if (width == prevWidth ||
        low >
            high) { // needed to reassure the same radius is not repeated again
      ResultVec = result; // return the keypoints from the previous iteration
      break;
    }
    result.clear();
    double c = (double)width / 2.0; // initializing Grid
    int numCellCols = std::floor(cols / c);
    int numCellRows = std::floor(rows / c);
    std::vector<std::vector<bool>> coveredVec(numCellRows + 1,
                                    std::vector<bool>(numCellCols + 1, false));

    for (unsigned int i = 0; i < keyPoints.size(); ++i) {
      int row =
          std::floor(keyPoints[i].pt.y /
                c); // get position of the cell current point is located at
      int col = std::floor(keyPoints[i].pt.x / c);
      if (coveredVec[row][col] == false) { // if the cell is not covered
        result.push_back(i);
        int rowMin = ((row - std::floor(width / c)) >= 0)
                         ? (row - std::floor(width / c))
                         : 0; // get range which current radius is covering
        int rowMax = ((row + std::floor(width / c)) <= numCellRows)
                         ? (row + std::floor(width / c))
                         : numCellRows;
        int colMin =
            ((col - std::floor(width / c)) >= 0) ? (col - std::floor(width / c)) : 0;
        int colMax = ((col + std::floor(width / c)) <= numCellCols)
                         ? (col + std::floor(width / c))
                         : numCellCols;
        for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov) {
          for (int colToCov = colMin; colToCov <= colMax; ++colToCov) {
            if (!coveredVec[rowToCov][colToCov])
              coveredVec[rowToCov][colToCov] =
                  true; // cover cells within the square bounding box with width
                        // w
          }
        }
      }
    }

    if (result.size() >= Kmin && result.size() <= Kmax) { // solution found
      ResultVec = result;
      complete = true;
    } else if (result.size() < Kmin)
      high = width - 1; // update binary search range
    else
      low = width + 1;
    prevWidth = width;
  }
  // retrieve final keypoints
  std::vector<cv::KeyPoint> kp;
  for (unsigned int i = 0; i < ResultVec.size(); i++)
    kp.push_back(keyPoints[ResultVec[i]]);

  return kp;
}

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

float distance(int x1, int y1, int x2, int y2)
{
    // Calculating distance
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
}

FeatureDrawer::FeatureDrawer(ros::NodeHandle *nh, const Zed_Camera* zedptr) : m_it(*nh), img_sync(MySyncPolicy(10), subLeftIm, subRightIm)
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
    subLeftIm.subscribe(*nh, zedcamera->cameraLeft.path, 3);
    subRightIm.subscribe(*nh, zedcamera->cameraRight.path, 3);
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
  detector = cv::ORB::create(1500,1.2f,8,10, 0, 2, cv::ORB::HARRIS_SCORE,10);
  // std::cout << "size : " << keypoints.size() << '\n';
  detector->compute(image, keypoints, descriptors);
  // std::cout << "size after : " << keypoints.size() << '\n';
}


/**
 * @brief Find Features with Adaptive Fast Threshold. This is used to be certain that a number of features will be found on each grid image so that features are homogeneous on the whole image
 * 
 * @param patch the cropped image
 * @param step the step with which the fast threshold is changed per iteration
 * @param iterations the number of times the fast threshold is changed
 * @return std::vector< cv::KeyPoint > returns the resulting keypoints
 */
std::vector< cv::KeyPoint > Features::featuresAdaptiveThreshold(cv::Mat& patch, int step = 8, unsigned int iterations = 2)
{
  int fastThreshold = 15;
  std::vector< cv::KeyPoint > tempkeys;
  for (size_t iii = 0; iii < iterations; iii++)
  {
    int numbOfFeatures = 100;
    int numbNeeded = 10;
    int edgeThreshold = 0;
    findORBFeatures(patch, tempkeys, numbOfFeatures, edgeThreshold, fastThreshold);
    if (tempkeys.size() >= numbNeeded)
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
  const int totalFeatures = 1500;
  const int edgeThreshold = 10;
  const int featuresPerCell = totalFeatures/(rows*cols);
  const int featuresPerCellFind = 2*totalFeatures/(rows*cols);
  cv::Mat cImage = image.rowRange(edgeThreshold, image.rows - edgeThreshold).colRange(edgeThreshold, image.cols - edgeThreshold);
  for (int iii = 0; iii < rows; iii++)
  {
    for (int jjj = 0; jjj < cols; jjj++)
    {
      // int grid[2] {iii, jjj};
      cv::Size imgSize = cv::Size(cImage.cols/cols,cImage.rows/rows);
      // std::cout << "height : " << imgSize.height << " width : " << imgSize.width << '\n';

      // cv::Mat patch = gridBasedFeatures(croppedImage, grid, imgSize);
      cv::Mat patch = cImage.rowRange(iii*imgSize.height, (iii+1)*imgSize.height).colRange(jjj*imgSize.width, (jjj+1)*imgSize.width);
      // std::cout << "size : " << patch.rows << " cols " << patch.cols << '\n' << "iii*imgSize.height " << iii*imgSize.height << " jjj*imgSize.width " << jjj*imgSize.width << '\n';
      // cv::Mat patch = croppedImage(cv::Rect(jjj*imgSize.width, iii*imgSize.height, imgSize.width, imgSize.height));
      // std::vector< cv::KeyPoint > tempkeys = featuresAdaptiveThreshold(patch);
      std::vector< cv::KeyPoint > tempkeys;
      findORBFeatures(patch,tempkeys,featuresPerCellFind,edgeThreshold, 15);
      // cv::FAST(patch,tempkeys,10,true);
      if(tempkeys.size() < featuresPerCell)
      {
        findORBFeatures(patch,tempkeys,featuresPerCellFind,edgeThreshold, 10);
        // cv::FAST(patch,tempkeys,5,true);
      }
      indicesOfGrids.push_back(keypoints.size());
      cv::KeyPointsFilter::retainBest(tempkeys,featuresPerCell);
      if(!tempkeys.empty())
      {
          for (auto key:tempkeys)
          {
              key.pt.x +=jjj*imgSize.width + edgeThreshold;
              key.pt.y +=iii*imgSize.height + edgeThreshold;
              keypoints.push_back(key);
          }
      }
    }
    
  }
  indicesOfGrids.push_back(keypoints.size());
  
  
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

void Features::findORBFeatures(cv::Mat& image, std::vector< cv::KeyPoint >& keypoints, int featuresPerCell, int edgeThreshold, int fastThreshold)
{
  // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(numbOfFeatures,1.2f,8, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE,30, fastThreshold);
  detector = cv::ORB::create(featuresPerCell,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,edgeThreshold, fastThreshold);
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
  cv::Mat img_matches = firstImage.realImage.clone();
  // drawMatches( firstImage.image, firstImage.keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
            // cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  for (auto m:matches)
  {
    cv::circle(img_matches, firstImage.keypoints[m.queryIdx].pt,2,cv::Scalar(0,255,0));
    cv::line(img_matches,firstImage.keypoints[m.queryIdx].pt, secondImage.keypoints[m.trainIdx].pt,cv::Scalar(0,0,255));
    cv::circle(img_matches, secondImage.keypoints[m.trainIdx].pt,2,cv::Scalar(255,0,0));
  }
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
  // std::cout << "l size : " << pointl.size() << " r size : " << pointr.size() << '\n';
  if (pointl.size() > 3 && pointr.size() > 3)
  {
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
    if (abs(leftImage.keypoints[m.queryIdx].pt.y - rightImage.keypoints[m.trainIdx].pt.y) < 4 && abs(leftImage.keypoints[m.queryIdx].pt.x - rightImage.keypoints[m.trainIdx].pt.x) < 20)
      good_matches.push_back(m);
  }
  return good_matches;
}

std::vector< cv::DMatch > FeatureDrawer::removeOutliersMatch(const std::vector< cv::DMatch >& matches, const Features& leftImage, const Features& rightImage, bool LR)
{
  std::vector< cv::DMatch > good_matches;
  int yDif = 4;

  for (auto m:matches)
  {
    if (abs(leftImage.keypoints[m.queryIdx].pt.y - rightImage.keypoints[m.trainIdx].pt.y) < yDif)
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
  // std::cout << "  row*cols + col - 1 : " << row*cols + col - 1 << " row : " << row << " col :  " << col << '\n';
  // std::cout << "indices  : " << secondImage.indicesOfGrids[row*cols + col + 1] << " kets size : " << secondImage.keypoints.size() << '\n';
  if (LR)
  {
    if (col > 0)
    {
      secDesc = secondImage.descriptors.rowRange(secondImage.indicesOfGrids[row*cols + col - 1],secondImage.indicesOfGrids[row*cols + col + 1]).colRange(cv::Range::all());
      // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col - 1],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
    }
    else
    {
      secDesc = secondImage.descriptors.rowRange(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]).colRange(cv::Range::all());
      // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
    }
  }
  // std::cout << "indices size : " << firstImage.indicesOfGrids.size() << " row*cols + col + 1 : " << row*cols + col + 1 << '\n';
  // std::cout << "indices  : " << firstImage.indicesOfGrids[row*cols + col + 1] << " kets size : " << firstImage.keypoints.size() << '\n';
  firDesc = firstImage.descriptors.rowRange(firstImage.indicesOfGrids[row*cols + col],firstImage.indicesOfGrids[row*cols + col + 1]).colRange(cv::Range::all());
  // firDesc = firstImage.descriptors(cv::Range(firstImage.indicesOfGrids[row*cols + col],firstImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
  // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
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
      // std::cout << "matches BEFORE size : " << matches.size() << "\n";
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
      std::vector < cv::DMatch > goodMatches = removeOutliersMatch(matches, firstImage, secondImage, true);
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
      std::vector < cv::DMatch > goodMatches = removeOutliersMatch(matches, firstImage, secondImage, false);
      
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
        secDesc.push_back(secondImage.descriptorsGrids[0]);
        secDesc.push_back(secondImage.descriptorsGrids[1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[cols] - secondImage.indicesOfGrids[1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[cols]);
        secDesc.push_back(secondImage.descriptorsGrids[cols + 1]);
      }
      else if (col == (cols - 1))
      {
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[cols - 2] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[cols - 2]);
        secDesc.push_back(secondImage.descriptorsGrids[cols - 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[2*cols - 2] - secondImage.indicesOfGrids[cols - 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[2*cols - 2]);
        secDesc.push_back(secondImage.descriptorsGrids[2*cols - 1]);
      }
      else
      {

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
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
      }
      else if (col == (cols - 1))
      {
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
      }
      else
      {
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
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row - 1)*cols + col - 1] - secondImage.indicesOfGrids[0]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row - 1)*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[row*cols + col - 1] - secondImage.indicesOfGrids[(row - 1)*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[row*cols + col + 1]);
        secDesc.push_back(cv::Mat::zeros(cv::Size(32, secondImage.indicesOfGrids[(row + 1)*cols + col - 1] - secondImage.indicesOfGrids[row*cols + col + 1 + 1]),CV_8U));
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col - 1]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col]);
        secDesc.push_back(secondImage.descriptorsGrids[(row + 1)*cols + col + 1]);
      }
    }
  }
  // std::cout << " row : " << row << " col :  " << col << '\n';
  firDesc = firstImage.descriptorsGrids[row*cols + col].clone();
  // secDesc = secondImage.descriptors(cv::Range(secondImage.indicesOfGrids[row*cols + col],secondImage.indicesOfGrids[row*cols + col + 1]),cv::Range::all());
  // std::cout << " image number : " << row*cols + col << " row : " << row << " col : " << col << '\n';
  // std::cout << " secDesc    rows : " << secDesc.rows << '\n';
  // std::cout << " firDesc    rows : " << firDesc.rows << '\n';
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

int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch>>& matches) {
  int removed=0;
  // for all matches
  for (std::vector<std::vector<cv::DMatch>>::iterator
  matchIterator= matches.begin();
  matchIterator!= matches.end(); ++matchIterator) 
  {
    // if 2 NN has been identified
    if (matchIterator->size() > 1) {
    // check distance ratio
    if ((*matchIterator)[0].distance/
    (*matchIterator)[1].distance > ratio) 
    {
      matchIterator->clear(); // remove match
      removed++;
    }
    } 
    else 
    { // does not have 2 neighbours
      matchIterator->clear(); // remove match
      removed++;
    }
  }
  return removed;
 }

void RobustMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches) 
{
  // for all matches image 1 -> image 2
  for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1) 
  {
    // ignore deleted matches
    if (matchIterator1->size() < 2)
    continue;
    // for all matches image 2 -> image 1
    for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end(); ++matchIterator2) 
    {
      // ignore deleted matches
      if (matchIterator2->size() < 2)
      continue;
      // Match symmetry test
      if ((*matchIterator1)[0].queryIdx ==
      (*matchIterator2)[0].trainIdx &&
      (*matchIterator2)[0].queryIdx ==
      (*matchIterator1)[0].trainIdx) 
      {
        // add symmetrical match
        symMatches.push_back(
        cv::DMatch((*matchIterator1)[0].queryIdx,
        (*matchIterator1)[0].trainIdx,
        (*matchIterator1)[0].distance));
        break; // next match in image 1 -> image 2
      }
    }
  }
 }

cv::Mat RobustMatcher::match(cv::Mat& image1,cv::Mat& image2, std::vector<cv::DMatch>& matches,std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2) 
 {
  // 1a. Detection of the SURF features
  // detector->detect(image1,keypoints1);
  // detector->detect(image2,keypoints2);
  // 1b. Extraction of the SURF descriptors
  cv::Mat descriptors1, descriptors2;
  detector->compute(image1,keypoints1,descriptors1);
  detector->compute(image2,keypoints2,descriptors2);
  // 2. Match the two image descriptors
  // Construction of the matcher
  // cv::BruteForceMatcher<cv::L2<float>> matcher;
  cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  // from image 1 to image 2
  // based on k nearest neighbours (with k=2)
  std::vector<std::vector<cv::DMatch>> matches1;
  matcher.knnMatch(descriptors1,descriptors2,matches1, 2); // return 2 nearest neighbours
  // from image 2 to image 1
  // based on k nearest neighbours (with k=2)
  // std::vector<std::vector<cv::DMatch>> matches2;
  // matcher.knnMatch(descriptors2,descriptors1,matches2, 2); // return 2 nearest neighbours

  int removed = ratioTest(matches1);
  // removed= ratioTest(matches2);
  // 4. Remove non-symmetrical matches
  // std::vector<cv::DMatch> symMatches;
  // symmetryTest(matches1,matches2,symMatches);
  // 5. Validate matches using RANSAC
  std::vector< cv::DMatch > matches2;
  for(size_t i = 0; i < matches1.size(); i++) 
  {
    if(matches1[i].size() >= 2)
    {
      cv::DMatch first = matches1[i][0];
      float dist1 = matches1[i][0].distance;
      float dist2 = matches1[i][1].distance;

      if(dist1 < 0.8f * dist2) 
      {
        matches2.push_back(matches1[i][0]);
      }
    }

  }
  cv::Mat fundemental= ransacTest(matches2,
  keypoints1, keypoints2, matches);
  // return the found fundemental matrix
  return fundemental;
 }

cv::Mat RobustMatcher::ransacTest(const std::vector<cv::DMatch>& matches,const std::vector<cv::KeyPoint>& keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& outMatches) 
 {
  // Convert keypoints into Point2f
  std::vector<cv::Point2f> points1, points2;
  if (matches.size() < 7)
  {
    return cv::Mat();
  }
  for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it) 
  {
    // Get the position of left keypoints
    float x= keypoints1[it->queryIdx].pt.x;
    float y= keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x,y));
    // Get the position of right keypoints
    x= keypoints2[it->trainIdx].pt.x;
    y= keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x,y));
  }
  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(),0);
  cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), inliers, cv::FM_RANSAC,1.0, 0.98); // confidence probability
  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator itIn= inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM= matches.begin();
  // for all matches
  for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
  {
    if (*itIn) { // it is a valid match
      outMatches.push_back(*itM);
    }
  }
  if (refineF) 
  {
    // The F matrix will be recomputed with
    // all accepted matches
    // Convert keypoints into Point2f
    // for final F computation
    points1.clear();
    points2.clear();
    for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();it!= outMatches.end(); ++it) 
    {
      // Get the position of left keypoints
      float x= keypoints1[it->queryIdx].pt.x;
      float y= keypoints1[it->queryIdx].pt.y;
      points1.push_back(cv::Point2f(x,y));
      // Get the position of right keypoints
      x= keypoints2[it->trainIdx].pt.x;
      y= keypoints2[it->trainIdx].pt.y;
      points2.push_back(cv::Point2f(x,y));
    }
    // Compute 8-point F from all accepted matches
    fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), cv::FM_8POINT); // 8-point method
  }
  return fundemental;
 }

std::vector<cv::DMatch> FeatureDrawer::matchFund(Features& firstImage, Features& secondImage, bool LR)
{
  auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  std::vector< std::vector < cv::DMatch > > knnmatches;
  matcher->knnMatch(firstImage.descriptors, secondImage.descriptors, knnmatches, 2);
  std::vector <cv::DMatch> matches = loweRatioTest(knnmatches);

  // find fund funct
  std::vector < cv::Point2f > points1, points2;
  for (auto m:matches)
  {
    points1.push_back(firstImage.keypoints[m.queryIdx].pt);
    points2.push_back(secondImage.keypoints[m.trainIdx].pt);
  }
  std::vector<uchar> inliers(points1.size(),0);
  cv::Mat fund = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, cv::FM_RANSAC, 3, 0.99);
  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator itIn= inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM= matches.begin();
  std::vector <cv::DMatch> outMatches;
  // for all matches
  for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
  {
    if (*itIn) 
    { // it is a valid match
     outMatches.push_back(*itM);
    }
 }
  drawFeatureMatches(outMatches, firstImage, secondImage);
  return outMatches;
}

std::vector<cv::DMatch> FeatureDrawer::matchFundTrial(Features& firstImage, Features& secondImage, bool LR)
{
  std::vector<cv::DMatch> matches;
  cv::Mat fundemental = rmatcher.match(firstImage.image,secondImage.image,matches, firstImage.keypoints, secondImage.keypoints);
  // drawFeatureMatches(matches, firstImage, secondImage);
  // std::cout << "LpL matches size : " << matches.size() << '\n';
  return matches;
}

void FeatureDrawer::publishImage(cv::Mat& image, const std_msgs::Header& header)
{
  cv_bridge::CvImage out_msg;
  out_msg.header   = header; // Same timestamp and tf frame as input image
  out_msg.encoding = sensor_msgs::image_encodings::MONO16; // Or whatever
  out_msg.image    = image; // Your cv::Mat
  mImageMatches.publish(out_msg.toImageMsg());
}

void FeatureDrawer::findDisparity(cv::Mat& lImage, cv::Mat& rImage, cv::Mat& disparity)
{
  int minDisparity = 0;
  int numDisparities = 32;
  int block = 11;
  int P1 = block * block * 8;
  int P2 = block * block * 32;
  auto sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, block, P1, P2);
  sgbm->compute(lImage, rImage, disparity);
}

void FeatureDrawer::again()
{
  // Get Features of Each Image
  clock_t fastStart = clock();
  int rows {5};
  int cols {5};
  // std::cout << "rows : " << rows << " cols : " << cols << '\n';
  // int rows {6};
  // int cols {8};
  bool LR {true};
  // std::cout << "rows :W " << rows << " cols : " << cols << '\n';
  // clock_t fastStart = clock();
  leftImage.getFeatures(rows, cols, mImageMatches, true);
  rightImage.getFeatures(rows, cols, mImageMatches, false);
  // cv::FAST(leftImage.image, leftImage.keypoints,10,true);
  // cv::FAST(rightImage.image, rightImage.keypoints,10,true);
  // cv::KeyPointsFilter::retainBest(leftImage.keypoints,1200);
  // cv::KeyPointsFilter::retainBest(rightImage.keypoints,1200);
  // clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
  std::cout << "lkeys : " << leftImage.keypoints.size() << " rkeys : " << rightImage.keypoints.size() << '\n'; 
  // cv::Mat outImage;
  // cv::drawKeypoints(leftImage.image, leftImage.keypoints,outImage);
  // publishImage(outImage, leftImage.header);
  // auto detecter = cv::FastFeatureDetector::create(10);
  // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(5000,1.2f,8, 10, 0, 2, cv::ORB::HARRIS_SCORE,10, 10);
  // // cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
  // std::vector < cv::KeyPoint > lkey, rkey;
  // detector->detect(leftImage.image, leftImage.keypoints);
  // detector->detect(rightImage.image, rightImage.keypoints);
  // {
  //   std::vector<float> responseVector;
  //   for (unsigned int i = 0; i < leftImage.keypoints.size(); i++)
  //     responseVector.push_back(leftImage.keypoints[i].response);
  //   std::vector<int> Indx(responseVector.size());
  //   std::iota(std::begin(Indx), std::end(Indx), 0);
  //   cv::sortIdx(responseVector, Indx, cv::SORT_DESCENDING);
  //   std::vector<cv::KeyPoint> keyPointsSorted;
  //   for (unsigned int i = 0; i < leftImage.keypoints.size(); i++)
  //     keyPointsSorted.push_back(leftImage.keypoints[Indx[i]]);
  //   leftImage.keypoints = ssc(leftImage.keypoints, 1250, 0.3, cols, rows);
  // }
  // {
  //   std::vector<float> responseVector;
  //   for (unsigned int i = 0; i < rightImage.keypoints.size(); i++)
  //     responseVector.push_back(rightImage.keypoints[i].response);
  //   std::vector<int> Indx(responseVector.size());
  //   std::iota(std::begin(Indx), std::end(Indx), 0);
  //   cv::sortIdx(responseVector, Indx, cv::SORT_DESCENDING);
  //   std::vector<cv::KeyPoint> keyPointsSorted;
  //   for (unsigned int i = 0; i < rightImage.keypoints.size(); i++)
  //     keyPointsSorted.push_back(rightImage.keypoints[Indx[i]]);
  //   rightImage.keypoints = ssc(rightImage.keypoints, 1250, 0.3, cols, rows);
  // }
  // leftImage.sortFeaturesKdTree();
  // rightImage.sortFeaturesKdTree();
  // for (auto k:leftImage.keypoints)
  // {
  //   std::cout << k.pt << "\n";
  // }
  // detector->compute(leftImage.image, leftImage.keypoints, leftImage.descriptors);
  // detector->compute(rightImage.image, rightImage.keypoints, rightImage.descriptors);
  // std::cout << "passed\n";
  // clock_t fastStart = clock();
  // std::vector < cv::DMatch > goodMatches = matchFundTrial(leftImage, rightImage, true);
  leftImage.getDescriptors(leftImage.image, leftImage.keypoints, leftImage.descriptors);
  rightImage.getDescriptors(rightImage.image, rightImage.keypoints, rightImage.descriptors);
  // std::cout << "cols : " << leftImage.descriptors.cols << '\n';
  // std::cout << "lkeys : " << leftImage.descriptors.rows << " rkeys : " << rightImage.descriptors.rows << '\n'; 
  // for (int ind:leftImage.indicesOfGrids)
  // {
  //   std::cout << " " << ind << " ";
  // }
  // std::cout << '\n';
  // for (int ind:rightImage.indicesOfGrids)
  // {
  //   std::cout << " " << ind << " ";
  // }
  std::vector < cv::DMatch > goodMatches = matchesWithGrids(leftImage, rightImage,rows,cols,true);
  // clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
  drawFeatureMatches(goodMatches, leftImage, rightImage);
  std::vector < cv::KeyPoint> first,second;
  std::vector < cv::DMatch > matchesLR;
  int count = 0;
  for (auto m:goodMatches)
  {
    first.push_back(leftImage.keypoints[m.queryIdx]);
    second.push_back(rightImage.keypoints[m.trainIdx]);
    matchesLR.push_back(cv::DMatch(count,count,0));
    count ++;
  }
  leftImage.keypoints = first;
  rightImage.keypoints = second;
  std::cout << "LR matches size : " << goodMatches.size() << '\n';

  // cv::Mat disparity;
  // findDisparity(leftImage.image, rightImage.image, disparity);
  // publishImage(disparity, leftImage.header);
  // std::vector < cv::DMatch > matcheslr = matchFundTrial(leftImage, rightImage, true);

  // std::cout << "LR matches size : " << matcheslr.size() << '\n';
  // drawFeatureMatches(matcheslr, leftImage, rightImage);
  if (!firstImage)
  {
    // auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    // std::vector< std::vector < cv::DMatch > > knnmatches;
    // matcher->knnMatch(leftImage.descriptors, previousLeftImage.descriptors, knnmatches, 2);
    // std::vector < cv::DMatch> matches;
    // matches = loweRatioTest(knnmatches);
    // // std::vector < cv::DMatch > goodMatchesPrev = matchesWithGridsPrev(previousLeftImage, leftImage,rows,cols,true);
    // std::vector < cv::DMatch > goodMatchesPrev = removeOutliersMatch(matches, leftImage, previousLeftImage, false);
    // drawFeatureMatches(goodMatchesPrev, leftImage, previousLeftImage);

    // featuresMatched.clear();
    // std::cout << "matches prev : " << goodMatchesPrev.size() << '\n';
    // std::cout << "lkey size : " << lkey.size() << '\n';
    if (previousLeftImage.keypoints.size() > 5 && leftImage.keypoints.size() > 5)
    {
      std::vector < cv::DMatch > matchesl = matchFundTrial(leftImage, previousLeftImage, false);
      std::cout << "LpL matches size : " << matchesl.size() << '\n';
    }
    // std::cout << "matches size : " << ma.size() << '\n';
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
  previousLeftImage.descriptors = leftImage.descriptors.clone(); 
  previousRightImage.image = rightImage.image.clone();
  previousRightImage.keypoints = rightImage.keypoints;
  previousRightImage.indicesOfGrids = rightImage.indicesOfGrids;
  previousRightImage.descriptors = rightImage.descriptors.clone(); 
  leftImage.clearFeatures();
  rightImage.clearFeatures();
  firstImage = false;
  clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
  std::cout << "-------------------------\n";
  std::cout << "\nTotal Time      : " << fastTotalTime        << " milliseconds." << '\n';
  std::cout << "-------------------------\n";
}

void Features::clearFeatures()
{
  keypoints.clear();
  indicesOfGrids.clear();
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
    std::cout << P1 << '\n';
    std::cout << "P2 : \n";
    std::cout << P2 << '\n';

    
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
    realImage = cv_ptr->image.clone();
    header = cv_ptr->header;
}

FeatureDrawer::~FeatureDrawer()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

} //namespace vio_slam