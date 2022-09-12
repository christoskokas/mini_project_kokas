#include "trial.h"

namespace vio_slam
{

void ImageFrame::findFeaturesFAST()
{
    cv::FAST(image, keypoints,15,true);
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);

}

void ImageFrame::findFeaturesORB()
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(5000,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,31,15);
    detector->detect(image,keypoints); 
    cv::KeyPointsFilter::retainBest(keypoints, totalNumber);

}

void ImageFrame::findFeaturesFASTAdaptive()
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

void ImageFrame::findFeaturesORBAdaptive()
{
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
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
                for (auto& key:tempkeys)
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

void ImageFrame::drawFeaturesWithLines(cv::Mat& outImage)
{
    cv::drawKeypoints(image, keypoints,outImage);
    for (size_t row = 0; row < rows; row++)
    {
        cv::line(outImage,cv::Point(row*image.cols/rows,0), cv::Point(row*image.cols/rows,image.rows),cv::Scalar(0,255,0,255));
        cv::line(outImage,cv::Point(0,row*image.rows/rows), cv::Point(image.cols,row*image.rows/rows),cv::Scalar(0,255,0,255));
        
    }
    
}

void RobustMatcher2::testFeatureExtraction()
{
    std::string imagePath = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/city.jpg";
    image = cv::imread(imagePath,cv::IMREAD_COLOR);
    assert(!image.empty() && "Could not read the image");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    clock_t fastStart = clock();
    leftImage.findFeaturesFAST();
    clock_t fastTotalTime = double(clock() - fastStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat fastImage;
    leftImage.drawFeaturesWithLines(fastImage);
    std::cout << "fast size : " << keypoints.size() << '\n';
    leftImage.keypoints.clear();
    cv::imshow("fast features", fastImage);
    clock_t fastGridStart = clock();
    leftImage.findFeaturesFASTAdaptive();
    clock_t fastGridTotalTime = double(clock() - fastGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat fastAdaptiveImage;
    leftImage.drawFeaturesWithLines(fastAdaptiveImage);
    std::cout << "fast grid size : " << keypoints.size() << '\n';
    leftImage.keypoints.clear();
    cv::imshow("fast features GRID", fastAdaptiveImage);
    clock_t ORBStart = clock();
    leftImage.findFeaturesORB();
    clock_t ORBTotalTime = double(clock() - ORBStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBImage;
    leftImage.drawFeaturesWithLines(ORBImage);
    std::cout << "ORB size : " << keypoints.size() << '\n';
    leftImage.keypoints.clear();
    cv::imshow("ORB features", ORBImage);
    clock_t ORBGridStart = clock();
    leftImage.findFeaturesORBAdaptive();
    clock_t ORBGridTotalTime = double(clock() - ORBGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBAdaptiveImage;
    leftImage.drawFeaturesWithLines(ORBAdaptiveImage);
    std::cout << "ORB grid size : " << keypoints.size() << '\n';
    leftImage.keypoints.clear();
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
    const int times = 328;
    int averageTime = 0;
    for (int frame = 0; frame < times; frame++)
    {
        bool withThread = true;
        start = clock();
        std::cout << "frame : " << frame << std::endl;
        if (withThread)
        {
            std::thread rightImageThread(&vio_slam::ImageFrame::findFeaturesOnImage, std::ref(rightImage), frame, "right", std::ref(rmap[1][0]), std::ref(rmap[1][1]));
            leftImage.findFeaturesOnImage(frame, "left", rmap[0][0], rmap[0][1]);
            rightImageThread.join();
        }
        else 
        {
            leftImage.findFeaturesOnImage(frame, "left", rmap[0][0], rmap[0][1]);
            rightImage.findFeaturesOnImage(frame, "right", rmap[1][0], rmap[1][1]);
        }
        detector->compute(leftImage.image, leftImage.keypoints,leftImage.desc);

        
        
        // getImage(leftImage.image, leftImage.realImage, frame, "left");
        // rectifyImage(leftImage.image,rmap[0][0],rmap[0][1]);
        // findFeaturesORBAdaptive(leftImage.image, leftImage.keypoints);
        detector->compute(rightImage.image, rightImage.keypoints,rightImage.desc);
        std::vector < cv::DMatch > matches;
        matchCrossRatio(leftImage, rightImage, matches);
        cv::Mat matchesImage;
        drawFeatureMatches(matches,leftImage, rightImage,matchesImage);
        cv::imshow("Matches", matchesImage);
        std::cout << "Matches size : " << matches.size() << std::endl;
        matches.clear();
        leftImage.keypoints.clear();
        rightImage.keypoints.clear();
        total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
        averageTime += total;
        std::cout << "-------------------------\n";
        std::cout << "\n Frame Processing Time  : " << total  << " milliseconds." << std::endl;
        std::cout << "-------------------------\n";
        cv::waitKey(1);
    }
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time should be : 66 milliseconds. (15fps so 1/15 = 66ms)" << std::endl;
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time  of " << times << " frames : " << averageTime/times  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
    
}

void RobustMatcher2::matchCrossRatio(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matches)
{
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
    std::vector < std::vector < cv::DMatch > > knnmatches1, knnmatches2;
    matcher.knnMatch(first.desc, second.desc,knnmatches1,2);
    matcher.knnMatch(second.desc, first.desc,knnmatches2,2);
    ratioTest(knnmatches1);
    ratioTest(knnmatches2);
    std::vector < cv::DMatch > matchesSym;
    symmetryTest(knnmatches1, knnmatches2,matchesSym);
    // matches = matchesSym;
    classIdCheck(first, second, matchesSym, matches);
}

void RobustMatcher2::classIdCheck(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesSym, std::vector < cv::DMatch >& matches)
{
    for (std::vector<cv::DMatch>::iterator matchIterator= matchesSym.begin(); matchIterator!= matchesSym.end(); ++matchIterator)
    {
        if ((*matchIterator).distance < 2*averageDistance)
        {
            if (first.keypoints[(*matchIterator).queryIdx].class_id % cols == 0)
            {
                if (first.keypoints[(*matchIterator).queryIdx].class_id == second.keypoints[(*matchIterator).trainIdx].class_id)
                {
                    matches.push_back(*matchIterator);
                }
            }
            else
            {
                if ((first.keypoints[(*matchIterator).queryIdx].class_id - second.keypoints[(*matchIterator).trainIdx].class_id < 2) && (first.keypoints[(*matchIterator).queryIdx].class_id >= second.keypoints[(*matchIterator).trainIdx].class_id))
                {
                    matches.push_back(*matchIterator);
                }
            }
        }
    }
}

void RobustMatcher2::symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches) 
{
  // for all matches image 1 -> image 2
    int count = 0;
    float dist = 0.0f;
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
            dist += (*matchIterator1)[0].distance;
            count ++;
            break; // next match in image 1 -> image 2
        }
        }
    }
    averageDistance = dist/count;
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

void ImageFrame::getImage(int frameNumber, const char* whichImage)
{
    std::string imagePath;
    if (frameNumber > 99)
    {
        imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/") + whichImage +std::string("/frame0") + std::to_string(frameNumber/100) + std::to_string((frameNumber%100 - frameNumber%10)/10) + std::to_string(frameNumber%10) + std::string(".jpg");
    }
    else if (frameNumber > 9)
    {
        imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/") + whichImage +std::string("/frame00") + std::to_string(frameNumber/10) + std::to_string(frameNumber%10) + std::string(".jpg");
    }
    else
    {
        imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/") + whichImage +std::string("/frame000") + std::to_string(frameNumber) + std::string(".jpg");
    }
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
    leftImage.rectifyImage(rmap[0][0],rmap[0][1]);
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

void ImageFrame::rectifyImage(cv::Mat& map1, cv::Mat& map2)
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
        cv::circle(outImage, firstImage.keypoints[m.queryIdx].pt,2,cv::Scalar(0,255,0));
        cv::line(outImage,firstImage.keypoints[m.queryIdx].pt, secondImage.keypoints[m.trainIdx].pt,cv::Scalar(0,0,255));
        cv::circle(outImage, secondImage.keypoints[m.trainIdx].pt,2,cv::Scalar(255,0,0));
    }

}

void RobustMatcher2::beginTest()
{
    testFeatureExtraction();
}

void ImageFrame::findFeaturesOnImage(int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2)
{
    getImage(frameNumber, whichImage);
    rectifyImage(map1, map2);
    findFeaturesORBAdaptive();
}

}