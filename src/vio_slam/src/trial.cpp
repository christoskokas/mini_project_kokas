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
                cv::FAST(patch,tempkeys,5,true);
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
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::HARRIS_SCORE,10,15);
    detector->compute(image, keypoints,desc);
    
}

void ImageFrame::findFeaturesORBAdaptive()
{
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::FAST_SCORE,10,15);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            cv::Mat patch = image.rowRange(row*imgSize.height, (row+1)*imgSize.height).colRange(col*imgSize.width, (col+1)*imgSize.width);
            std::vector< cv::KeyPoint > tempkeys;
            detector->detect(patch,tempkeys); 
            if(tempkeys.size() < numberPerCell)
            {
                detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::FAST_SCORE,10,10);
                detector->detect(patch,tempkeys); 
                detector = cv::ORB::create(numberPerCellFind,1.2f,8,0,0,2,cv::ORB::FAST_SCORE,10,15);
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
    detector->compute(image, keypoints,desc);
    
}

void ImageFrame::findFeaturesGoodFeatures()
{
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    int numberOfPoints = 200;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack(image,optPoints,numberOfPoints,qualityLevel,minDistance, cv::Mat(),blockSize,useHarrisDetector,k);
}

void ImageFrame::findDisparity(cv::Mat& otherImage, cv::Mat& disparity)
{
    int minDisparity = 0;
    int numDisparities = 32;
    int block = 11;
    int P1 = block * block * 8;
    int P2 = block * block * 32;
    float wlsLamda = 8000.0;
    float wlsSigma = 1.5;
    cv::Mat rightDisp, leftDisp, realDisp;
    cv::Mat leftRes,rightRes;
    cv::resize(image ,leftRes ,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
    cv::resize(otherImage,rightRes,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
    auto bm = cv::StereoBM::create(numDisparities,block);
    bm->compute(leftRes, rightRes, disparity);

    
    // bm->compute(leftRes, rightRes, leftDisp);
    // cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(bm);
    // cv::Ptr<cv::StereoMatcher> right = cv::ximgproc::createRightMatcher(bm);
    // right->compute(rightRes, leftRes,rightDisp);
    // wls_filter->setLambda(wlsLamda);
    // wls_filter->setSigmaColor(wlsSigma);
    // wls_filter->filter(leftDisp,image,disparity,rightDisp);



    // cv::ximgproc::getDisparityVis(realDisp,disparity);
    //   bm->setMinDisparity(0);
    //   bm->setPreFilterCap(25);
    //   bm->setPreFilterSize(5);
    //   bm->setTextureThreshold(10);
    //   bm->setUniquenessRatio(69);
    //   bm->setSpeckleRange(2);
    //   auto sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, block, P1, P2);
    //   sgbm->compute(image,otherImage,disparity);
}

void ImageFrame::opticalFlowRemoveOutliers(std::vector < cv::Point2f>& optPoints, std::vector < cv::Point2f>& prevOptPoints, cv::Mat& status)
{
    std::vector < cv::Point2f> inliersL, inlierspL;
    for(size_t i = 0; i < prevOptPoints.size(); i++) 
    {
        if (status.at<bool>(i))
        {
            inliersL.push_back(optPoints[i]);
            inlierspL.push_back(prevOptPoints[i]);
        }
    }
    optPoints = inliersL;
    prevOptPoints = inlierspL;
}

void ImageFrame::opticalFlow(ImageFrame& prevImage,cv::Mat& status, cv::Mat& optFlow)
{
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(prevImage.image, image, prevImage.optPoints, optPoints, status, err);
    // optFlow = image.clone();
    // opticalFlowRemoveOutliers(optPoints,prevImage.optPoints,status);
    // for(int j=0; j<prevImage.optPoints.size(); j++)
    // {
    //     cv::line(optFlow,prevImage.optPoints[j],optPoints[j],cv::Scalar(255,0,0,255));
    // }

    // cv::imshow("Optical Flow", optFlow);
}

void RobustMatcher2::drawOpticalFlow(ImageFrame& prevImage, ImageFrame& curImage, cv::Mat& outImage)
{
    outImage = curImage.image.clone();
    for(int j=0; j<prevImage.optPoints.size(); j++)
    {
        cv::line(outImage,prevImage.optPoints[j],curImage.optPoints[j],cv::Scalar(255,0,0,255));
    }
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

void ImageFrame::setImage(const sensor_msgs::ImageConstPtr& imageRef)
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

void RobustMatcher2::testFeatureExtraction()
{
    std::string imagePath = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/city.jpg";
    leftImage.image = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    assert(!leftImage.image.empty() && "Could not read the image");
    // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
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
    std::vector<cv::Point2f> corners;
    clock_t GoodFeaturesStart = clock();
    leftImage.findFeaturesGoodFeatures();
    clock_t GoodFeaturesTotalTime = double(clock() - GoodFeaturesStart) * 1000 / (double)CLOCKS_PER_SEC;
    int r = 4;
    cv::Mat outImage = leftImage.image.clone();
    for( size_t i = 0; i < leftImage.optPoints.size(); i++ )
        { cv::circle( outImage, leftImage.optPoints[i], r, cv::Scalar(255,0,0,255), -1, 8, 0 ); }
    cv::imshow( "good Features To Track", outImage );
    std::cout << "\nFast Features Time      : " << fastTotalTime        << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nFast Grid Features Time : " << fastGridTotalTime    << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nORB Features Time       : " << ORBTotalTime         << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nORB Grid Features Time  : " << ORBGridTotalTime     << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    std::cout << "\nGood Features Features Time  : " << GoodFeaturesTotalTime     << " milliseconds." << '\n';
    std::cout << "-------------------------\n";
    cv::waitKey(0);
}

void RobustMatcher2::testDisparityWithOpticalFlow()
{
    std::cout << "-------------------------\n";
    std::cout << "Disparity With Optical Flow \n";
    std::cout << "-------------------------\n";
    const int times = 200;
    // const int times = 658;
    bool firstImage = true;
    bool withThread = true;
    int averageTime = 0;
    for (int frame = 0; frame < times; frame++)
    {
        start = clock();
        leftImage.getImage(frame, "left");
        leftImage.rectifyImage(rmap[0][0], rmap[0][1]);
        rightImage.getImage(frame, "right");
        rightImage.rectifyImage(rmap[1][0], rmap[1][1]);
        if (firstImage)
        {
            prevLeftImage.image = leftImage.image.clone();
            firstImage = false;
            continue;
        }
        cv::Mat disparity, status;
        if (withThread)
        {
            std::thread disp(&vio_slam::ImageFrame::findDisparity,std::ref(leftImage),std::ref(rightImage.image),std::ref(disparity));
            if (prevLeftImage.optPoints.size()<10)
            {
                // New Keyframe
                prevLeftImage.findFeaturesGoodFeatures();
            }
            cv::Mat optFlow;
            leftImage.opticalFlow(prevLeftImage,status,optFlow);
            disp.join();
            std::cout << "number of tracked points : " << leftImage.optPoints.size() << std::endl;
            cv::Mat dispNorm;
            cv::Mat depthImage;
            cv::reprojectImageTo3D(disparity,depthImage,Q);
            cv::imshow("disparity",depthImage);
            // cv::normalize(disparity, dispNorm,0,1,cv::NORM_MINMAX, CV_32F);
            // cv::imshow("disparity",disparity);
        }
        else
        {
            leftImage.findDisparity(rightImage.image, disparity);
            // CHECK FOR NEW KEYFRAME
            if (prevLeftImage.optPoints.size()<10)
            {
                // New Keyframe
                prevLeftImage.findFeaturesGoodFeatures();
            }
            cv::Mat optFlow;
            leftImage.opticalFlow(prevLeftImage,status, optFlow);
        }

        // TODO REVERSE OPTICAL FLOW (Cross Check)

        // Because Images are rectified we can use directly the disparities


        
        //Calculate feature position
        prevLeftImage.optPoints = leftImage.optPoints;
        prevLeftImage.image = leftImage.image.clone();
        leftImage.optPoints.clear();
        total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
        averageTime += total;

        std::cout << "-------------------------\n";
        std::cout << "\n Frame Processing Time  : " << total  << " milliseconds." << std::endl;
        std::cout << "-------------------------\n";
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time should be : 66 milliseconds. (15fps so 1/15 = 66ms)" << std::endl;
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time  of " << times << " frames : " << averageTime/times  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}

void RobustMatcher2::triangulatePointsOpt(ImageFrame& first, ImageFrame& second, cv::Mat& points3D)
{
    cv::Mat Points4D(4,1,CV_32F);

    cv::triangulatePoints(P1, P2, first.optPoints,second.optPoints,Points4D);
    
    cv::convertPointsFromHomogeneous(Points4D.t(),points3D);
}

void RobustMatcher2::ceresSolver(cv::Mat& points3D, cv::Mat& prevPoints3D)
{
    ceres::Problem problem;
    ceres::LossFunction* lossfunction = NULL;
    for (size_t i = 0; i < points3D.rows; i++)
    {   
        if ((points3D.at<float>(i,2) < 40*zedcamera->mBaseline) && (prevPoints3D.at<float>(i,2) < 40*zedcamera->mBaseline))
        {
            float x = points3D.at<float>(i,0);
            float y = points3D.at<float>(i,1);
            float z = points3D.at<float>(i,2);
            float xp = prevPoints3D.at<float>(i,0);
            float yp = prevPoints3D.at<float>(i,1);
            float zp = prevPoints3D.at<float>(i,2);
            // std::cout << "PREVIOUS : " <<  xp << ' ' << yp  << " " << zp << '\n';
            // std::cout << "OBSERVED : " <<  x << ' ' << y  << " " << z << '\n';
            // std::cout << "x : " << x << " p3d : " << p3d.x() << '\n';
            Eigen::Vector3d p3d(x, y, z);
            Eigen::Vector3d pp3d(xp, yp, zp);
            ceres::CostFunction* costfunction = Reprojection3dError::Create(pp3d, p3d);
            problem.AddResidualBlock(costfunction, lossfunction, camera);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    options.trust_region_strategy_type = ceres::DOGLEG;
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

void RobustMatcher2::testFeatureMatchingWithOpticalFlow()
{
    std::cout << "-------------------------\n";
    std::cout << "Feature Matching With Optical Flow \n";
    std::cout << "-------------------------\n";
    const int waitKey = 1;
    // const int times = 100;
    const int times = 658;
    bool firstImage = true;
    bool withThread = true;
    int averageTime = 0;
    clock_t maxTime = 0;
    clock_t minTime = 100;
    for (int frame = 0; frame < times; frame++)
    {
        start = clock();
        leftImage.getImage(frame, "left");
        leftImage.rectifyImage(rmap[0][0], rmap[0][1]);
        rightImage.getImage(frame, "right");
        rightImage.rectifyImage(rmap[1][0], rmap[1][1]);
        if (firstImage)
        {
            prevLeftImage.image = leftImage.image.clone();
            prevRightImage.image = rightImage.image.clone();
            prevLeftImage.realImage = leftImage.realImage.clone();
            prevRightImage.realImage = rightImage.realImage.clone();
            firstImage = false;
            continue;
        }
        if (withThread)
        {
            if (prevLeftImage.optPoints.size()<20)
            {
                // New Keyframe
                prevLeftImage.findFeaturesGoodFeatures();
                cv::Mat optFlowLR, statusPrev;
                prevRightImage.opticalFlow(prevLeftImage, statusPrev, optFlowLR);
                std::cout << " prev left size : " << prevLeftImage.optPoints.size() << std::endl;
                std::cout << " prev right size : " << prevRightImage.optPoints.size() << std::endl;
                
                
                reduceVector(prevLeftImage.optPoints,statusPrev);
                reduceVector(prevRightImage.optPoints,statusPrev);


                cv::Mat inliers;
                cv::Mat fund = cv::findFundamentalMat(cv::Mat(prevLeftImage.optPoints), cv::Mat(prevRightImage.optPoints), inliers, cv::FM_RANSAC, 1, 0.99);
                reduceVector(prevLeftImage.optPoints,inliers);
                reduceVector(prevRightImage.optPoints,inliers);
                drawOpticalFlow(prevLeftImage,prevRightImage,optFlowLR);
                // cv::imshow("LR optical", optFlowLR);
                std::cout << " prev left size : " << prevLeftImage.optPoints.size() << std::endl;
                std::cout << " prev right size : " << prevRightImage.optPoints.size() << std::endl;
            }
            cv::Mat statusL, statusR;
            cv::Mat optFlow,optFlowR;
            leftImage.opticalFlow(prevLeftImage,statusL, optFlow);

            rightImage.opticalFlow(prevRightImage,statusR, optFlowR);

            cv::Mat status = statusL & statusR;
            std::cout << " left size : " << leftImage.optPoints.size() << std::endl;
            std::cout << " prev left size : " << prevLeftImage.optPoints.size() << std::endl;
            std::cout << " right size : " << rightImage.optPoints.size() << std::endl;
            std::cout << " prev right size : " << prevRightImage.optPoints.size() << std::endl;

            reduceVector(leftImage.optPoints,status);
            reduceVector(prevLeftImage.optPoints,status);
            reduceVector(rightImage.optPoints,status);
            reduceVector(prevRightImage.optPoints,status);

            std::cout << " left size : " << leftImage.optPoints.size() << std::endl;
            std::cout << " prev left size : " << prevLeftImage.optPoints.size() << std::endl;
            std::cout << " right size : " << rightImage.optPoints.size() << std::endl;
            std::cout << " prev right size : " << prevRightImage.optPoints.size() << std::endl;

            // Find Fund Matrix only from Left Image Points because the Right Camera Frame moves the same way as the Left Camera
            cv::Mat inliers;
            cv::Mat fund = cv::findFundamentalMat(cv::Mat(leftImage.optPoints), cv::Mat(prevLeftImage.optPoints), inliers, cv::FM_RANSAC, 1, 0.99);
            
            
            reduceVector(leftImage.optPoints,inliers);
            reduceVector(prevLeftImage.optPoints,inliers);
            reduceVector(rightImage.optPoints,inliers);
            reduceVector(prevRightImage.optPoints,inliers);

            std::cout << " AFTER RANSACCCCC left size : " << leftImage.optPoints.size() << std::endl;
            std::cout << " prev left size : " << prevLeftImage.optPoints.size() << std::endl;
            std::cout << " right size : " << rightImage.optPoints.size() << std::endl;
            std::cout << " prev right size : " << prevRightImage.optPoints.size() << std::endl;

            drawOpticalFlow(prevLeftImage,leftImage,optFlow);
            // cv::imshow("LpL optical", optFlow);

            drawOpticalFlow(prevRightImage,rightImage,optFlowR);
            // cv::imshow("RpR optical", optFlowR);
            cv::Mat PrevPoints3D, Points3D;
            triangulatePointsOpt(prevLeftImage,prevRightImage,PrevPoints3D);
            triangulatePointsOpt(leftImage,rightImage,Points3D);

            ceresSolver(Points3D,PrevPoints3D);
            publishPose();


            // std::cout << "points" << Points3D << std::endl;
            // for (size_t i = 0; i < Points3D.rows; i++)
            // {
            //     std::cout << "prev x : " << PrevPoints3D.at<float>(i,0) << " y : " << PrevPoints3D.at<float>(i,1) << " z : " << PrevPoints3D.at<float>(i,2) << std::endl;
            //     std::cout << "after x : " << Points3D.at<float>(i,0) << " y : " << Points3D.at<float>(i,1) << " z : " << Points3D.at<float>(i,2) << std::endl;
            // }
            
            // std::thread opt(&vio_slam::ImageFrame::opticalFlow,std::ref(prevRightImage),std::ref(prevLeftImage),std::ref(optFlowLR));
            // opt.join();
        }
        else
        {

            // CHECK FOR NEW KEYFRAME
            if (prevLeftImage.optPoints.size()<100 || prevRightImage.optPoints.size()<100)
            {
                // New Keyframe
                prevLeftImage.findFeaturesGoodFeatures();
                // prevRightImage.findFeaturesGoodFeatures();
                // prevLeftImage.pointsToKeyPoints();
                // prevRightImage.pointsToKeyPoints();
                // detector->compute(prevLeftImage.image,prevLeftImage.keypoints,prevLeftImage.desc);
                // detector->compute(prevRightImage.image,prevRightImage.keypoints,prevRightImage.desc);
                // std::vector< cv::DMatch > matches;
                // matchCrossRatio(prevLeftImage,prevRightImage,matches,true);
                // std::cout << "Matches size : " << matches.size() << std::endl;
                // cv::Mat matchesImage;
                // drawFeatureMatches(matches,prevLeftImage, prevRightImage,matchesImage);
                // cv::imshow("Matches", matchesImage);
                // prevLeftImage.keypoints.clear();
                // prevRightImage.keypoints.clear();

            }
            cv::Mat optFlow, status;
            leftImage.opticalFlow(prevLeftImage,status, optFlow);
            cv::imshow("LpL Optical", optFlow);

        }

        // TODO REVERSE OPTICAL FLOW (Cross Check)

        // TODO Rejct Outliers With Fundamental Matrix (left to prevleft has the same Fund Matrix with right to prevright)

        // Because Images are rectified we can use directly the disparities


        // for (size_t i = 0; i < leftImage.optPoints.size(); i++)
        // {
        //     std::cout << "left : " << leftImage.optPoints[i] << std::endl;
        //     std::cout << "right : " << rightImage.optPoints[i] << std::endl;
        // }
        
        
        //Calculate feature position
        prevLeftImage.optPoints = leftImage.optPoints;
        prevLeftImage.image = leftImage.image.clone();
        prevLeftImage.realImage = leftImage.realImage.clone();
        prevRightImage.optPoints = rightImage.optPoints;
        prevRightImage.image = rightImage.image.clone();
        prevRightImage.realImage = rightImage.realImage.clone();
        leftImage.optPoints.clear();
        rightImage.optPoints.clear();
        total = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
        averageTime += total;
        if (total > maxTime)
            maxTime = total;
        if (total < minTime)
            minTime = total;
        std::cout << "-------------------------\n";
        std::cout << "\n Frame Processing Time  : " << total  << " milliseconds." << std::endl;
        std::cout << "-------------------------\n";
        cv::waitKey(waitKey);
    }
    cv::destroyAllWindows();
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time should be : 66 milliseconds. (15fps so 1/15 = 66ms)" << std::endl;
    std::cout << "-------------------------\n";
    std::cout << "\n Average Processing Time  of " << times << " frames : " << averageTime/times  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
    std::cout << "\n Max Processing Time of Single Frame  : " << maxTime  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
    std::cout << "\n Min Processing Time of Single Frame  : " << minTime  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
}

void RobustMatcher2::testFeatureMatching()
{
    std::cout << "-------------------------\n";
    std::cout << "Feature Matching Trials \n";
    std::cout << "-------------------------\n";
    const int times = 200;
    // const int times = 100;
    int averageTime = 0;
    for (int frame = 0; frame < times; frame++)
    {
        bool withThread = false;
        start = clock();
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
            // detector->compute(leftImage.image, leftImage.keypoints,leftImage.desc);
            // detector->compute(rightImage.image, rightImage.keypoints,rightImage.desc);
            
        }

        // TODO reduce vectors

        std::vector < cv::DMatch > matches;
        matchCrossRatio(leftImage, rightImage, matches, true);
        std::cout << "LR Matches size : " << matches.size() << std::endl;
        cv::Mat matchesImage;
        drawFeatureMatches(matches,leftImage, rightImage,matchesImage);
        cv::imshow("Matches", matchesImage);

        // matches.clear();
        // if (!firstImage)
        // {
        //     matchCrossRatio(leftImage, prevLeftImage, matches, false);
            
        //     std::cout << "LpL Matches size : " << matches.size() << std::endl;

        // }
        // matches.clear();
        // prevLeftImage.clone(leftImage);
        firstImage = false;
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
    std::cout << "\n Average Processing Time  of " << times + 1 << " frames : " << averageTime/(times + 1)  << " milliseconds." << std::endl;
    std::cout << "-------------------------\n";
    
}

void RobustMatcher2::matchCrossRatio(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matches, bool LR)
{
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
    std::vector < std::vector < cv::DMatch > > knnmatches1, knnmatches2;
    matcher.knnMatch(first.desc, second.desc,knnmatches1,2);
    matcher.knnMatch(second.desc, first.desc,knnmatches2,2);
    ratioTest(knnmatches1);
    ratioTest(knnmatches2);
    std::vector < cv::DMatch > matchesSym, matchesId;
    symmetryTest(first, second, knnmatches1, knnmatches2,matchesSym);
    // matches = matchesSym;
    classIdCheck(first, second, matchesSym, matchesId, LR);
    removeMatchesDistance(first, second,matchesId, matches);
}

void RobustMatcher2::removeMatchesDistance(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesId, std::vector < cv::DMatch >& matches)
{
    for (std::vector<cv::DMatch>::iterator matchIterator= matchesId.begin(); matchIterator!= matchesId.end(); ++matchIterator)
    {
        if (getDistanceOfPoints(first, second,(*matchIterator)) < 1.3*averageDistance)
        {
            matches.push_back(*matchIterator);
        }
    }
}

void RobustMatcher2::classIdCheck(ImageFrame& first, ImageFrame& second, std::vector < cv::DMatch >& matchesSym, std::vector < cv::DMatch >& matches, bool LR)
{
    int count = 0;
    float dist = 0.0f;
    for (std::vector<cv::DMatch>::iterator matchIterator= matchesSym.begin(); matchIterator!= matchesSym.end(); ++matchIterator)
    {
        // std::cout << "averageDistance " << averageDistance << " get dist " << (*matchIterator).distance;
        bool matchadded = false;
        int firstClassId = first.keypoints[(*matchIterator).queryIdx].class_id;
        int secondClassId = second.keypoints[(*matchIterator).trainIdx].class_id;
        if (LR)
        {
            if ((first.keypoints[(*matchIterator).queryIdx].pt.y - second.keypoints[(*matchIterator).trainIdx].pt.y) < 3)
            {
                if (firstClassId % cols == 0)
                {
                    if (firstClassId == secondClassId)
                    {
                        matches.push_back(*matchIterator);
                        matchadded = true;
                    }
                }
                else
                {
                    if ((firstClassId - secondClassId < 2) && (firstClassId >= secondClassId))
                    {
                        matches.push_back(*matchIterator);
                        matchadded = true;
                    }
                }
            }
        }
        else
        {
            if (abs(firstClassId%cols - secondClassId%cols) < 2 && abs(firstClassId/cols - secondClassId/cols) < 2)
            {
                matches.push_back(*matchIterator);
                matchadded = true;
            }
        }
        if (matchadded)
        {
            dist += getDistanceOfPoints(first, second,(*matchIterator));
            count ++;
        }
    }
    averageDistance = dist/count;
}

void RobustMatcher2::symmetryTest(ImageFrame& first, ImageFrame& second, const std::vector<std::vector<cv::DMatch>>& matches1,const std::vector<std::vector<cv::DMatch>>& matches2,std::vector<cv::DMatch>& symMatches) 
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

void ImageFrame::clone(const ImageFrame& second)
{
    keypoints = second.keypoints;
    desc = second.desc.clone();
}

void ImageFrame::pointsToKeyPoints()
{
    std::vector<cv::Point2f>::iterator point;
    for (std::vector<cv::Point2f>::const_iterator point =optPoints.begin();point!=optPoints.end();++point)
    {
        keypoints.push_back(cv::KeyPoint(cv::Point2f((*point).x,(*point).y),1));
    }
}

void RobustMatcher2::reduceVector(std::vector<cv::Point2f> &v, cv::Mat& status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status.at<bool>(i))
            v[j++] = v[i];
    v.resize(j);
}

void RobustMatcher2::ImagesCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
{
    trial.setImage(rIm);
    std::cout << "LOOOOOOOOOOOOOOOOOL\n";
}

void RobustMatcher2::publishPose()
{
nav_msgs::Odometry position;
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

position.header.frame_id = "whatever";
position.header.stamp = ros::Time::now();
posePublisher.publish(position);
std::cout << "pose Published \n";
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

float RobustMatcher2::getDistanceOfPoints(ImageFrame& first, ImageFrame& second, const cv::DMatch& match)
{
    float x1 = first.keypoints[match.queryIdx].pt.x;
    float y1 = first.keypoints[match.queryIdx].pt.y;
    float x2 = second.keypoints[match.trainIdx].pt.x;
    float y2 = second.keypoints[match.trainIdx].pt.y;
    return sqrt(pow(y2-y1,2) + pow(x2-x1,2));
}

void RobustMatcher2::beginTest()
{
    // testFeatureExtraction();
    // testDisparityWithOpticalFlow();
    // testFeatureMatching();
    testFeatureMatchingWithOpticalFlow();
}

void ImageFrame::findFeaturesOnImage(int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2)
{
    getImage(frameNumber, whichImage);
    rectifyImage(map1, map2);
    findFeaturesORBAdaptive();
}

}