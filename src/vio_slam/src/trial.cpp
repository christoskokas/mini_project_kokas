#include "trial.h"

namespace vio_slam
{

void RobustMatcher2::findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    cv::FAST(image, keypoints,15,true);
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);

}

void RobustMatcher2::findFeaturesORB(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    detector = cv::ORB::create(5000,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,31,15);
    detector->detect(image,keypoints); 
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);

}

void RobustMatcher2::findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            cv::Mat patch = image.rowRange(row*imgSize.height, (row+1)*imgSize.height).colRange(col*imgSize.width, (col+1)*imgSize.width);
            std::vector< cv::KeyPoint > tempkeys;
            cv::FAST(patch,tempkeys,15,true);
            if(tempkeys.empty())
            {
                cv::FAST(patch,tempkeys,10,true);
            }
            if(!tempkeys.empty())
            {
                for (auto key:tempkeys)
                {
                    key.pt.x +=col*imgSize.width;
                    key.pt.y +=row*imgSize.height;
                    key.class_id = row*cols + col;
                    keypoints.push_back(key);
                }
            }
        }
        
    }
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);
    
}

void RobustMatcher2::findFeaturesORBAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            cv::Mat patch = image.rowRange(row*imgSize.height, (row+1)*imgSize.height).colRange(col*imgSize.width, (col+1)*imgSize.width);
            std::vector< cv::KeyPoint > tempkeys;
            detector->detect(patch,tempkeys); 
            if(tempkeys.size() < numberPerCell)
            {
                detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,10);
                detector->detect(patch,tempkeys); 
                detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
            }
            if(!tempkeys.empty())
            {
                cv::KeyPointsFilter::retainBest(tempkeys,numberPerCell);
                for (auto key:tempkeys)
                {
                    key.pt.x +=col*imgSize.width;
                    key.pt.y +=row*imgSize.height;
                    key.class_id = row*cols + col;
                    keypoints.push_back(key);
                }
            }
        }
        
    }
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);
    
}

void RobustMatcher2::drawFeaturesWithLines(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& outImage)
{
    cv::drawKeypoints(image, keypoints,outImage);
    for (size_t row = 0; row < rows; row++)
    {
        cv::line(outImage,cv::Point(row*image.cols/rows,0), cv::Point(row*image.cols/rows,image.rows),cv::Scalar(0,255,0,255));
        cv::line(outImage,cv::Point(0,row*image.rows/rows), cv::Point(image.cols,row*image.rows/rows),cv::Scalar(0,255,0,255));
        
    }
    
}

void RobustMatcher2::testFeatureExtraction(cv::Mat& image)
{
    // std::string imagePath = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/city.jpg";
    // image = cv::imread(imagePath,cv::IMREAD_COLOR);
    assert(!image.empty() && "Could not read the image");
    // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    clock_t fastStart = clock();
    findFeatures(image, keypoints);
    clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat fastImage;
    drawFeaturesWithLines(image, keypoints,fastImage);
    std::cout << "fast size : " << keypoints.size() << '\n';
    keypoints.clear();
    cv::imshow("fast features", fastImage);
    clock_t fastGridStart = clock();
    findFeaturesAdaptive(image, keypoints);
    clock_t fastGridTotalTime = double(clock() - fastGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat fastAdaptiveImage;
    drawFeaturesWithLines(image, keypoints,fastAdaptiveImage);
    std::cout << "fast grid size : " << keypoints.size() << '\n';
    keypoints.clear();
    cv::imshow("fast features GRID", fastAdaptiveImage);
    clock_t ORBStart = clock();
    findFeaturesORB(image, keypoints);
    clock_t ORBTotalTime = double(clock() - ORBStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBImage;
    drawFeaturesWithLines(image, keypoints,ORBImage);
    std::cout << "ORB size : " << keypoints.size() << '\n';
    keypoints.clear();
    cv::imshow("ORB features", ORBImage);
    clock_t ORBGridStart = clock();
    findFeaturesORBAdaptive(image, keypoints);
    clock_t ORBGridTotalTime = double(clock() - ORBGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBAdaptiveImage;
    drawFeaturesWithLines(image, keypoints,ORBAdaptiveImage);
    std::cout << "ORB grid size : " << keypoints.size() << '\n';
    keypoints.clear();
    cv::imshow("ORB features GRID", ORBAdaptiveImage);
    std::cout << "\nFast Features Time      : " << fastTotalTime        << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nFast Grid Features Time : " << fastGridTotalTime    << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nORB Features Time       : " << ORBTotalTime         << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nORB Grid Features Time  : " << ORBGridTotalTime     << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    cv::waitKey(0);
}

void RobustMatcher2::testFeatureMatching()
{
    std::cout << "-------------------------\n";
    std::cout << "Feature Matching Trials \n";
    std::cout << "-------------------------\n";
    const int times = 3;
    for (int frame = 0; frame < times; frame++)
    {
        start = clock();
        getImage(leftImage.image, leftImage.realImage, frame, "left");
        getImage(rightImage.image, rightImage.realImage, frame, "right");
        rectifyImage(leftImage.image,rmap[0][0],rmap[0][1]);
        rectifyImage(rightImage.image,rmap[1][0],rmap[1][1]);
        std::vector<cv::KeyPoint> leftKeys,rightKeys;
        findFeaturesORBAdaptive(leftImage.image, leftImage.keys);
        findFeaturesORBAdaptive(rightImage.image, rightImage.keys);
        cv::Mat lImage,rImage;
        drawFeaturesWithLines(leftImage.image, leftImage.keys,leftImage.desc);
        drawFeaturesWithLines(rightImage.image, rightImage.keys,rightImage.desc);
        // cv::imshow("left Image",lImage);
        // cv::imshow("right Image",rImage);
        cv::Mat leftDesc,rightDesc;
        detector->compute(leftImage.image, leftImage.keys,leftImage.desc);
        detector->compute(rightImage.image, rightImage.keys,rightImage.desc);
        std::vector < cv::DMatch > matches;
        match(leftImage, rightImage, matches);
        cv::Mat matchesImage;
        drawFeatureMatches(matches,leftImage, rightImage,matchesImage);
        cv::imshow("Matches", matchesImage);
        std::cout << "Matches size : " << matches.size() << '\n';

        // cv::imshow("left Image Rectified", leftImage.image);
        total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
        std::cout << "-------------------------\n";
        std::cout << "\n Total Processing Time  : " << total  << " milliseconds." << '\n';
        std::cout << "-------------------------\n";
        cv::waitKey(0);
    }
    
}

void RobustMatcher2::match(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matches)
{
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector < std::vector < cv::DMatch > > knnmatches1, knnmatches2;
    matcher.knnMatch(first.desc, second.desc,knnmatches1,2);
    matcher.knnMatch(second.desc, first.desc,knnmatches2,2);
    ratioTest(knnmatches1);
    ratioTest(knnmatches2);
    std::vector < cv::DMatch > matchesSym;
    symmetryTest(knnmatches1, knnmatches2,matchesSym);
    classIdCheck(first, second, matchesSym, matches);
}

void RobustMatcher2::classIdCheck(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesSym, std::vector < cv::DMatch >& matches)
{
    for (std::vector<cv::DMatch>::iterator matchIterator= matchesSym.begin(); matchIterator!= matchesSym.end(); ++matchIterator)
    {
        if ((abs(first.keys[(*matchIterator).queryIdx].class_id - second.keys[(*matchIterator).trainIdx].class_id) < 2) && second.keys[(*matchIterator).trainIdx].class_id <=first.keys[(*matchIterator).queryIdx].class_id)
        {
            matches.push_back(*matchIterator);
        }
    }
}

void RobustMatcher2::symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches) 
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

void RobustMatcher2::ratioTest(std::vector<std::vector<cv::DMatch>>& matches) 
{
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
    }
    } 
    else 
    { // does not have 2 neighbours
      matchIterator->clear(); // remove match
    }
  }
 }

void RobustMatcher2::getImage(cv::Mat& image, cv::Mat& realImage, int frameNumber, const char* whichImage)
{
    std::string imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/") + whichImage +std::string("/frame000") + std::to_string(frameNumber) + std::string(".jpg");
    image = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    realImage = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void RobustMatcher2::testImageRectify()
{
    std::cout << "-------------------------\n";
    std::cout << "\n Image Rectify Testing \n";
    std::cout << "-------------------------\n";
    std::string imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/left/frame0000.jpg");
    leftImage.image = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    cv::imshow("left Image", leftImage.image);
    rectifyImage(leftImage.image,rmap[0][0],rmap[0][1]);
    cv::imshow("left Image Rectified", leftImage.image);
    cv::waitKey(0);
}

void RobustMatcher2::undistortMap()
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

void RobustMatcher2::rectifyImage(cv::Mat& image, cv::Mat& map1, cv::Mat& map2)
{
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
}

void RobustMatcher2::drawFeatureMatches(const std::vector<cv::DMatch>& matches, const ImageFrame& firstImage, const ImageFrame& secondImage, cv::Mat& outImage)
{
    outImage = firstImage.realImage.clone();
    // drawMatches( firstImage.image, firstImage.keypoints, secondImage.image, secondImage.keypoints, matches, img_matches, cv::Scalar::all(-1),
                // cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    for (auto m:matches)
    {
        cv::circle(outImage, firstImage.keys[m.queryIdx].pt,2,cv::Scalar(0,255,0));
        cv::line(outImage,firstImage.keys[m.queryIdx].pt, secondImage.keys[m.trainIdx].pt,cv::Scalar(0,0,255));
        cv::circle(outImage, secondImage.keys[m.trainIdx].pt,2,cv::Scalar(255,0,0));
    }

}

}