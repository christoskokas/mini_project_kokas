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
                    class_id.push_back(row*cols + col);
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
                    class_id.push_back(row*cols + col);
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
    int numberOfPoints = 400;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack(image,optPoints,numberOfPoints,qualityLevel,minDistance, cv::Mat(),blockSize,useHarrisDetector,k);
}

void ImageFrame::findFeaturesGoodFeaturesGrid()
{
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    int totalNumberOfPoints = 400;
    bool useHarrisDetector = false;
    double k = 0.04;
    int numberOfPoints = totalNumberOfPoints/(rows*cols);
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            cv::Mat patch = image.rowRange(row*imgSize.height, (row+1)*imgSize.height).colRange(col*imgSize.width, (col+1)*imgSize.width);
            std::vector< cv::Point2f > tempkeys;
            cv::goodFeaturesToTrack(patch,tempkeys,numberOfPoints,qualityLevel,minDistance, cv::Mat(),blockSize,useHarrisDetector,k);
            if(!tempkeys.empty())
            {
                for (auto& key:tempkeys)
                {
                    key.x +=col*imgSize.width;
                    key.y +=row*imgSize.height;
                    class_id.push_back(row*cols + col);
                    optPoints.push_back(key);
                }
            }
        }
        
    }

}

void ImageFrame::findFeaturesGoodFeaturesWithPast()
{
    cv::Mat mask = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    for (size_t i = 0; i < optPoints.size(); i++)
    {
        mask.at<uchar>(optPoints[i]) = 0;
    }
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    int numberOfPoints = 400;
    bool useHarrisDetector = false;
    double k = 0.04;
    std::vector < cv::Point2f > temp;
    cv::goodFeaturesToTrack(image,temp,numberOfPoints,qualityLevel,minDistance, mask,blockSize,useHarrisDetector,k);
    for (size_t i = 0; i < temp.size(); i++)
    {
        optPoints.push_back(temp[i]);
    }
    
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

void RobustMatcher2::opticalFlowRemoveOutliers(ImageFrame& first, ImageFrame& second, cv::Mat& status, bool LR)
{
    findAverageDistanceOfPoints(first,second);

    for (size_t i = 0; i < first.optPoints.size(); i++)
    {
        if (LR)
        {
            if (getDistanceOfPointsOptical(first.optPoints[i],second.optPoints[i])> 1.2*averageDistance || abs(first.optPoints[i].y - second.optPoints[i].y) > 1.0f)
            {
                status.at<bool>(i) = 0;
            }
        }
        else
        {
            if (getDistanceOfPointsOptical(first.optPoints[i],second.optPoints[i])> 1.2*averageDistance && abs(getAngleOfPoints(first.optPoints[i],second.optPoints[i]) - averageAngle[first.class_id[i]]) > 1.57079632679f)
            {
                status.at<bool>(i) = 0;
            }
        }
    }
    
}

void ImageFrame::opticalFlow(ImageFrame& prevImage,cv::Mat& status, cv::Mat& optFlow)
{
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(prevImage.image, image, prevImage.optPoints, optPoints, status, err,cv::Size(21,21),3,cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 60, (0.01000000000000000021)),cv::OPTFLOW_USE_INITIAL_FLOW);
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
    Logging("fast size", keypoints.size(),1);
    leftImage.keypoints.clear();
    cv::imshow("fast features", fastImage);
    clock_t fastGridStart = clock();
    leftImage.findFeaturesFASTAdaptive();
    clock_t fastGridTotalTime = double(clock() - fastGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat fastAdaptiveImage;
    leftImage.drawFeaturesWithLines(fastAdaptiveImage);
    Logging("fast grid size", keypoints.size(),1);
    leftImage.keypoints.clear();
    cv::imshow("fast features GRID", fastAdaptiveImage);
    clock_t ORBStart = clock();
    leftImage.findFeaturesORB();
    clock_t ORBTotalTime = double(clock() - ORBStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBImage;
    leftImage.drawFeaturesWithLines(ORBImage);
    Logging("ORB size", keypoints.size(),1);
    leftImage.keypoints.clear();
    cv::imshow("ORB features", ORBImage);
    clock_t ORBGridStart = clock();
    leftImage.findFeaturesORBAdaptive();
    clock_t ORBGridTotalTime = double(clock() - ORBGridStart) * 1000 / (double)CLOCKS_PER_SEC;
    cv::Mat ORBAdaptiveImage;
    leftImage.drawFeaturesWithLines(ORBAdaptiveImage);
    Logging("ORB grid size", keypoints.size(),1);
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
    Logging("FAST time ----",fastTotalTime,1);
    Logging("FAST Grid time ----",fastTotalTime,1);
    Logging("ORB time ----",fastTotalTime,1);
    Logging("ORB Grid time ----",fastTotalTime,1);
    Logging("Good features time ----",fastTotalTime,1);
    cv::waitKey(0);
}

void RobustMatcher2::testDisparityWithOpticalFlow()
{
    Logging("Disparity With Optical Flow","-------------------------",1);
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
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
}

void RobustMatcher2::triangulatePointsOpt(ImageFrame& first, ImageFrame& second, cv::Mat& points3D)
{
    cv::Mat Points4D(4,first.optPoints.size(),CV_32F);

    cv::triangulatePoints(P1, P2, first.optPoints,second.optPoints,Points4D);

    cv::convertPointsFromHomogeneous(Points4D.t(),points3D);

    // cv::Mat trial;
    // cv::convertPointsFromHomogeneous(Points4D.t(),trial);
    // cv::Mat cameraTransform(4,4,CV_32FC1);
    // Eigen::Matrix4f tr = previousT.cast <float> ();
    // cv::eigen2cv(tr,cameraTransform);
    // cv::perspectiveTransform(trial,points3D,cameraTransform);
}

void RobustMatcher2::triangulatePointsOptWithProjection(ImageFrame& first, ImageFrame& second, cv::Mat& points3D)
{
    cv::Mat Points4D(4,first.optPoints.size(),CV_32F);

    cv::triangulatePoints(P1, P2, first.optPoints,second.optPoints,Points4D);
    for (size_t i = 0; i < Points4D.cols; i++)
    {
        Points4D.at<float>(0,i) = Points4D.at<float>(0,i)/Points4D.at<float>(3,i);
        Points4D.at<float>(1,i) = Points4D.at<float>(1,i)/Points4D.at<float>(3,i);
        Points4D.at<float>(2,i) = Points4D.at<float>(2,i)/Points4D.at<float>(3,i);
        Points4D.at<float>(3,i) = Points4D.at<float>(3,i)/Points4D.at<float>(3,i);
    }
    cv::Mat cameraTransform(4,4,CV_32FC1);
    Eigen::Matrix4f tr = previousT.cast <float> ();
    cv::eigen2cv(tr,cameraTransform);
    cv::Mat pointsProj = cameraTransform * Points4D;
    points3D = pointsProj.colRange(cv::Range::all()).rowRange(cv::Range(0,3)).clone();
    points3D = points3D.t();
    
}

void RobustMatcher2::ceresSolver(cv::Mat& points3D, cv::Mat& prevPoints3D)
{
    // double camera[6] = {0, 1, 2, 0, 0, 0};
    ceres::Problem problem;
    ceres::LossFunction* lossfunction = NULL;
    for (size_t i = 0; i < points3D.rows; i++)
    {   
        if ((abs(points3D.at<float>(i,2)) < 40*zedcamera->mBaseline) && (abs(prevPoints3D.at<float>(i,2)) < 40*zedcamera->mBaseline))
        {
            double x = static_cast<double>(points3D.at<float>(i,0));
            double y = static_cast<double>(points3D.at<float>(i,1));
            double z = static_cast<double>(points3D.at<float>(i,2));
            double xp = static_cast<double>(prevPoints3D.at<float>(i,0));
            double yp = static_cast<double>(prevPoints3D.at<float>(i,1));
            double zp = static_cast<double>(prevPoints3D.at<float>(i,2));
            // Logging("Previous",cv::Point3d(xp,yp,zp),0);
            // Logging("Observed",cv::Point3d(x,y,z),0);
            Eigen::Vector3d p3d(x, y, z);
            Eigen::Vector3d pp3d(xp, yp, zp);
            ceres::CostFunction* costfunction = Reprojection3dError::Create(p3d, pp3d);
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

    double quat[4];
    ceres::AngleAxisToQuaternion(camera, quat);
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    Eigen::Isometry3d Transform(q.matrix());
    Transform.pretranslate(Eigen::Vector3d(camera[3], camera[4], camera[5]));
    T = Transform.matrix();
    
}

void RobustMatcher2::ceresSolverPnp(cv::Mat& points3D, cv::Mat& prevPoints3D)
{
    // double camera[6] = {0, 1, 2, 0, 0, 0};
    ceres::Problem problem;
    ceres::LossFunction* lossfunction = NULL;
    for (size_t i = 0; i < points3D.rows; i++)
    {   
        if ((points3D.at<float>(i,2) < 40*zedcamera->mBaseline) && (prevPoints3D.at<float>(i,2) < 40*zedcamera->mBaseline))
        {
            double x = points3D.at<float>(i,0);
            double y = points3D.at<float>(i,1);
            double z = points3D.at<float>(i,2);
            double xp = prevPoints3D.at<float>(i,0);
            double yp = prevPoints3D.at<float>(i,1);
            double zp = prevPoints3D.at<float>(i,2);
            // Logging("Previous",cv::Point3d(xp,yp,zp),0);
            // Logging("Observed",cv::Point3d(x,y,z),0);
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

    double quat[4];
    ceres::AngleAxisToQuaternion(camera, quat);
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    Eigen::Isometry3d Transform(q.matrix());
    Transform.pretranslate(Eigen::Vector3d(camera[3], camera[4], camera[5]));
    T = Transform.matrix();
    
}

void RobustMatcher2::testFeatureMatchingWithOpticalFlow()
{
    Logging("Feature Matching With Optical Flow","-------------------------",1);
    // const int waitKey = 0;
    // const int times = 100;
    const int waitKey = 1;
    const int times = 658;
    bool firstImage = true;
    bool withThread = true;
    int averageTime = 0;
    clock_t maxTime = 0;
    clock_t minTime = 100;
    int count = 0;
    leftImage.rows = 2;
    leftImage.cols = 2;
    rightImage.rows = 2;
    rightImage.cols = 2;
    prevLeftImage.rows = 2;
    prevLeftImage.cols = 2;
    prevRightImage.rows = 2;
    prevRightImage.cols = 2;
    for (int frame = 0; frame < times; frame++)
    {
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
            // if (count == 1)
            // {
            //     count = 0;
            //     continue;
            // }
            count = 1;
            if (prevLeftImage.optPoints.size()<100)
            {
                // New Keyframe
                
                // prevLeftImage.optPoints.clear();
                // prevLeftImage.keypoints.clear();
                // prevLeftImage.findFeaturesFASTAdaptive();
                // for (size_t i = 0; i < prevLeftImage.keypoints.size(); i++)
                // {
                //     prevLeftImage.optPoints.push_back(cv::Point2f(prevLeftImage.keypoints[i].pt.x,prevLeftImage.keypoints[i].pt.y));
                // }

                prevLeftImage.findFeaturesGoodFeatures();

                
                // cv::cornerSubPix(prevLeftImage.image,prevLeftImage.optPoints,cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ));
                

                cv::Mat optFlowLR, statusPrev,statusPrevc;
                predictRightImagePoints(prevLeftImage, prevRightImage);
                prevRightImage.opticalFlow(prevLeftImage, statusPrev, optFlowLR);
                opticalFlowRemoveOutliers(prevLeftImage,prevRightImage,statusPrev, true);
                
                
                reduceVector(prevLeftImage.optPoints,statusPrev);
                reduceVector(prevRightImage.optPoints,statusPrev);


                cv::Mat inliers;
                cv::Mat fund = cv::findFundamentalMat(cv::Mat(prevLeftImage.optPoints), cv::Mat(prevRightImage.optPoints), inliers, cv::FM_RANSAC, 1, 0.99);
                reduceVector(prevLeftImage.optPoints,inliers);
                reduceVector(prevRightImage.optPoints,inliers);
                drawOpticalFlow(prevLeftImage,prevRightImage,optFlowLR);
                cv::imshow("LR optical", optFlowLR);
            }
            cv::Mat statusL, statusR;
            cv::Mat optFlow,optFlowR;

            leftImage.optPoints = prevLeftImage.optPoints;
            leftImage.opticalFlow(prevLeftImage,statusL, optFlow);
            opticalFlowRemoveOutliers(leftImage,prevLeftImage,statusL, false);
            

            rightImage.optPoints = prevRightImage.optPoints;
            rightImage.opticalFlow(prevRightImage,statusR, optFlowR);
            opticalFlowRemoveOutliers(rightImage,prevRightImage,statusR, false);

            cv::Mat status = statusL & statusR;

            reduceVector(leftImage.optPoints,status);
            reduceVector(prevLeftImage.optPoints,status);
            reduceVector(rightImage.optPoints,status);
            reduceVector(prevRightImage.optPoints,status);


            // Find Fund Matrix only from Left Image Points because the Right Camera Frame moves the same way as the Left Camera
            cv::Mat inliers, inliersR;
            cv::Mat fund = cv::findFundamentalMat(cv::Mat(leftImage.optPoints), cv::Mat(prevLeftImage.optPoints), inliers, cv::FM_RANSAC, 1, 0.99);

            fund = cv::findFundamentalMat(cv::Mat(rightImage.optPoints), cv::Mat(prevRightImage.optPoints), inliersR, cv::FM_RANSAC, 1, 0.99);
            
            inliers = inliers & inliersR;

            reduceVector(leftImage.optPoints,inliers);
            reduceVector(prevLeftImage.optPoints,inliers);
            reduceVector(rightImage.optPoints,inliers);
            reduceVector(prevRightImage.optPoints,inliers);


            drawOpticalFlow(prevLeftImage,leftImage,optFlow);
            cv::imshow("LpL optical", optFlow);

            drawOpticalFlow(prevRightImage,rightImage,optFlowR);
            cv::imshow("RpR optical", optFlowR);
            cv::Mat PrevPoints3D, Points3D;


            triangulatePointsOpt(prevLeftImage,prevRightImage,PrevPoints3D);
            triangulatePointsOpt(leftImage,rightImage,Points3D);
            ceresSolver(Points3D,PrevPoints3D);
            publishPose();


            // triangulatePointsOptWithProjection(prevLeftImage,prevRightImage,PrevPoints3D);
            // triangulatePointsOptWithProjection(leftImage,rightImage,Points3D);
            // ceresSolver(Points3D,PrevPoints3D);
            // publishPose();


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
        cv::waitKey(waitKey);
    }
    cv::destroyAllWindows();
}

void RobustMatcher2::testOpticalFlowWithPairs()
{
    Logging("Feature Matching With Optical Flow Pairs","-------------------------",1);
    // const int waitKey = 0;
    // const int times = 200;
    const int waitKey = 1;
    const int times = 658;
    bool firstImage = true;
    bool withThread = true;
    int averageTime = 0;
    clock_t maxTime = 0;
    clock_t minTime = 100;
    leftImage.rows = 2;
    leftImage.cols = 2;
    rightImage.rows = 2;
    rightImage.cols = 2;
    prevLeftImage.rows = 2;
    prevLeftImage.cols = 2;
    prevRightImage.rows = 2;
    prevRightImage.cols = 2;
    for (int frame = 0; frame < times; frame++)
    {
        start = clock();
        leftImage.getImage(frame, "left");
        rightImage.getImage(frame, "right");
        if (!zedcamera->rectified)
        {
            leftImage.rectifyImage(rmap[0][0], rmap[0][1]);
            rightImage.rectifyImage(rmap[1][0], rmap[1][1]);
        }
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
            if (prevLeftImage.optPoints.size()<50)
            {
                // New Keyframe
                
                // prevLeftImage.optPoints.clear();
                // prevLeftImage.keypoints.clear();
                // prevLeftImage.findFeaturesORBAdaptive();
                // for (size_t i = 0; i < prevLeftImage.keypoints.size(); i++)
                // {
                //     prevLeftImage.optPoints.push_back(cv::Point2f(prevLeftImage.keypoints[i].pt.x,prevLeftImage.keypoints[i].pt.y));
                // }
                prevLeftImage.findFeaturesGoodFeaturesGrid();

                int prevSize = pointsTimes.size();
                for (size_t i = 0; i < (prevLeftImage.optPoints.size()-prevSize); i++)
                {
                    pointsTimes.push_back(0);
                }
                
                
                // cv::cornerSubPix(prevLeftImage.image,prevLeftImage.optPoints,cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ));
                
                cv::Mat optFlowLR, statusPrev,statusPrevc;
                predictRightImagePoints(prevLeftImage, prevRightImage);
                prevRightImage.class_id = prevLeftImage.class_id;
                prevRightImage.opticalFlow(prevLeftImage, statusPrev, optFlowLR);
                opticalFlowRemoveOutliers(prevLeftImage,prevRightImage,statusPrev, true);
                
                
                reduceVector(prevLeftImage.optPoints,statusPrev);
                reduceVector(prevRightImage.optPoints,statusPrev);
                reduceVectorInt(pointsTimes,statusPrev);
                reduceVectorInt(prevLeftImage.class_id,statusPrev);
                reduceVectorInt(prevRightImage.class_id,statusPrev);



                cv::Mat inliers;
                cv::Mat fund = cv::findFundamentalMat(cv::Mat(prevLeftImage.optPoints), cv::Mat(prevRightImage.optPoints), inliers, cv::FM_RANSAC, 1, 0.99);
                reduceVector(prevLeftImage.optPoints,inliers);
                reduceVector(prevRightImage.optPoints,inliers);
                reduceVectorInt(pointsTimes,inliers);
                reduceVectorInt(prevLeftImage.class_id,inliers);
                reduceVectorInt(prevRightImage.class_id,inliers);
                drawOpticalFlow(prevLeftImage,prevRightImage,optFlowLR);
                cv::imshow("LR optical", optFlowLR);
            }
            cv::Mat statusL, statusR;
            cv::Mat optFlow,optFlowR;

            leftImage.optPoints = prevLeftImage.optPoints;
            leftImage.class_id = prevLeftImage.class_id;
            leftImage.opticalFlow(prevLeftImage,statusL, optFlow);
            opticalFlowRemoveOutliers(leftImage,prevLeftImage,statusL, false);
            

            rightImage.optPoints = prevRightImage.optPoints;
            rightImage.class_id = prevRightImage.class_id;
            rightImage.opticalFlow(prevRightImage,statusR, optFlowR);
            opticalFlowRemoveOutliers(rightImage,prevRightImage,statusR, false);

            cv::Mat status = statusL & statusR;

            reduceVector(leftImage.optPoints,status);
            reduceVector(prevLeftImage.optPoints,status);
            reduceVector(rightImage.optPoints,status);
            reduceVector(prevRightImage.optPoints,status);
            reduceVectorInt(pointsTimes,status);
            reduceVectorInt(leftImage.class_id,status);
            reduceVectorInt(prevLeftImage.class_id,status);
            reduceVectorInt(rightImage.class_id,status);
            reduceVectorInt(prevRightImage.class_id,status);


            // Find Fund Matrix only from Left Image Points because the Right Camera Frame moves the same way as the Left Camera
            cv::Mat inliers, inliersR;
            cv::Mat fund = cv::findFundamentalMat(cv::Mat(leftImage.optPoints), cv::Mat(prevLeftImage.optPoints), inliers, cv::FM_RANSAC, 3, 0.99);

            fund = cv::findFundamentalMat(cv::Mat(rightImage.optPoints), cv::Mat(prevRightImage.optPoints), inliersR, cv::FM_RANSAC, 3, 0.99);
            
            inliers = inliers & inliersR;

            reduceVector(leftImage.optPoints,inliers);
            reduceVector(prevLeftImage.optPoints,inliers);
            reduceVector(rightImage.optPoints,inliers);
            reduceVector(prevRightImage.optPoints,inliers);
            reduceVectorInt(pointsTimes,inliers);
            reduceVectorInt(leftImage.class_id,inliers);
            reduceVectorInt(prevLeftImage.class_id,inliers);
            reduceVectorInt(rightImage.class_id,inliers);
            reduceVectorInt(prevRightImage.class_id,inliers);

            for (size_t i = 0; i < pointsTimes.size(); i++)
            {
                pointsTimes[i] ++;
            }
            


            drawOpticalFlow(prevLeftImage,leftImage,optFlow);
            cv::imshow("LpL optical", optFlow);

            drawOpticalFlow(prevRightImage,rightImage,optFlowR);
            cv::imshow("RpR optical", optFlowR);
            cv::Mat PrevPoints3D, Points3D;

            // int count = 0;
            // for (size_t i = 0; i < pointsTimes.size(); i++)
            // {
            //     if (pointsTimes[i] % 2 == 0 && pointsTimes[i] > 0)
            //     {
            //         count ++;
            //         if (count > 20)
            //         {
            //             break;
            //         }
            //     }
            // }
            

            // if (count > 20)
            // {
            triangulatePointsOpt(prevLeftImage,prevRightImage,PrevPoints3D);
            triangulatePointsOpt(leftImage,rightImage,Points3D);
            ceresSolver(Points3D,PrevPoints3D);
            publishPose();
            // }
            

            prevLeftImage.optPoints = leftImage.optPoints;
            prevLeftImage.class_id = leftImage.class_id;
            prevLeftImage.image = leftImage.image.clone();
            prevLeftImage.realImage = leftImage.realImage.clone();
            prevRightImage.optPoints = rightImage.optPoints;
            prevRightImage.class_id = rightImage.class_id;
            prevRightImage.image = rightImage.image.clone();
            prevRightImage.realImage = rightImage.realImage.clone();
            leftImage.optPoints.clear();
            rightImage.optPoints.clear();
            leftImage.class_id.clear();
            rightImage.class_id.clear();


            // triangulatePointsOptWithProjection(prevLeftImage,prevRightImage,PrevPoints3D);
            // triangulatePointsOptWithProjection(leftImage,rightImage,Points3D);
            // ceresSolver(Points3D,PrevPoints3D);
            // publishPose();


            
            // std::thread opt(&vio_slam::ImageFrame::opticalFlow,std::ref(prevRightImage),std::ref(prevLeftImage),std::ref(optFlowLR));
            // opt.join();
        }

        // TODO REVERSE OPTICAL FLOW (Cross Check)

        // TODO Rejct Outliers With Fundamental Matrix (left to prevleft has the same Fund Matrix with right to prevright)

        // Because Images are rectified we can use directly the disparities


        // for (size_t i = 0; i < leftImage.optPoints.size(); i++)
        // {
        // }
        
        
        //Calculate feature position
        
        cv::waitKey(waitKey);
    }
    cv::destroyAllWindows();
}

void RobustMatcher2::removeLeftRightOutliers(ImageFrame& left, ImageFrame& right, cv::Mat& status)
{
    for (size_t i = 0; i < left.optPoints.size(); i++)
    {
        if (abs(left.optPoints[i].y - right.optPoints[i].y) > 2)
            status.row(i) = 0;
    }
    
}

void RobustMatcher2::predictRightImagePoints(ImageFrame& left, ImageFrame& right)
{
    right.optPoints.clear();
    for (size_t i = 0; i < left.optPoints.size(); i++)
    {
        right.optPoints.push_back(cv::Point2f(left.optPoints[i].x - 10,left.optPoints[i].y));
    }
}

void RobustMatcher2::testFeatureMatching()
{
    Logging("Feature Matching Trials \n","-------------------------",1);
    const int times = 200;
    // const int times = 100;
    int averageTime = 0;
    for (int frame = 0; frame < times; frame++)
    {
        bool withThread = false;
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
        Logging("LR Matches size", matches.size(),1);
        cv::Mat matchesImage;
        drawFeatureMatches(matches,leftImage, rightImage,matchesImage);
        cv::imshow("Matches", matchesImage);

        // matches.clear();
        // if (!firstImage)
        // {
        //     matchCrossRatio(leftImage, prevLeftImage, matches, false);
            

        // }
        // matches.clear();
        // prevLeftImage.clone(leftImage);
        firstImage = false;
        leftImage.keypoints.clear();
        rightImage.keypoints.clear();

        cv::waitKey(1);
    }
    
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
    std::string first;
    std::string second, format;
    if (!kitti)
    {
        first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/";
        second = "/frame";
        format = ".jpg";
    }
    else
    {
        first = "/home/christos/Downloads/data_odometry_gray/dataset/sequences/00/";
        second = "/00";
        format = ".png";
    }

    if (frameNumber > 999)
    {
        imagePath = first + whichImage + second + std::to_string(frameNumber/1000) + std::to_string((frameNumber%1000 - frameNumber%100)/100) + std::to_string((frameNumber%100 - frameNumber%10)/10) + std::to_string(frameNumber%10) + format;
    }
    if (frameNumber > 99)
    {
        imagePath = first + whichImage + second + "0" + std::to_string(frameNumber/100) + std::to_string((frameNumber%100 - frameNumber%10)/10) + std::to_string(frameNumber%10) + format;
    }
    else if (frameNumber > 9)
    {
        imagePath = first + whichImage + second + "00" + std::to_string(frameNumber/10) + std::to_string(frameNumber%10) + format;
    }
    else
    {
        imagePath = first + whichImage + second + "000" + std::to_string(frameNumber) + format;
    }
    image = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    realImage = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void RobustMatcher2::testImageRectify()
{
    ProcessTime first("Rectify");
    Logging("Image Rectify Testing \n","-------------------------",1);
    std::string imagePath = std::string("/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/left/frame0000.jpg");
    leftImage.image = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    cv::imshow("left Image", leftImage.image);
    leftImage.rectifyImage(rmap[0][0],rmap[0][1]);
    cv::imshow("left Image Rectified", leftImage.image);
    first.totalTime();
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
    Logging("P1",P1,1);
    Logging("P2",P2,1);
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
        if (status.at<uchar>(i))
            v[j++] = v[i];
    v.resize(j);
}

void RobustMatcher2::reduceVectorInt(std::vector<int> &v, cv::Mat& status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status.at<uchar>(i))
            v[j++] = v[i];
    v.resize(j);
}

// void RobustMatcher2::ImagesCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm)
// {
//     trialL.setImage(lIm);
//     trialR.setImage(rIm);
// }

void RobustMatcher2::publishPose()
{
nav_msgs::Odometry position;
Eigen::Matrix3d Rot;
previousT = previousT * T;
zedcamera->cameraPose.pose = previousT;
zedcamera->cameraPose.poseTranspose = previousT.transpose();
Logging zed("Zed Camera Pose", zedcamera->cameraPose.pose,0);
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

void RobustMatcher2::drawFeatureMatchesStereo(const std::vector<cv::DMatch>& matches, const cv::Mat& image, const std::vector <cv::KeyPoint>& leftKeys, const std::vector <cv::KeyPoint>& rightKeys, cv::Mat& outImage)
{
    outImage = image.clone();
    for (auto m:matches)
    {
        cv::circle(outImage, leftKeys[m.queryIdx].pt,2,cv::Scalar(0,255,0));
        cv::line(outImage,leftKeys[m.queryIdx].pt, rightKeys[m.trainIdx].pt,cv::Scalar(0,0,255));
        cv::circle(outImage, rightKeys[m.trainIdx].pt,2,cv::Scalar(255,0,0));
    }

}

void RobustMatcher2::drawFeatureMatchesStereoSub(const std::vector<cv::DMatch>& matches, const cv::Mat& image, const std::vector <cv::Point2f>& leftKeys, const std::vector <cv::Point2f>& rightKeys, cv::Mat& outImage)
{
    outImage = image.clone();
    for (auto m:matches)
    {
        cv::circle(outImage, leftKeys[m.queryIdx],2,cv::Scalar(0,255,0));
        cv::line(outImage,leftKeys[m.queryIdx], rightKeys[m.trainIdx],cv::Scalar(0,0,255));
        cv::circle(outImage, rightKeys[m.trainIdx],2,cv::Scalar(255,0,0));
    }

}

float RobustMatcher2::getAngleOfPoints(cv::Point2f& first, cv::Point2f& second)
{
    return atan2(second.y - first.y,second.x - first.x);
}

float RobustMatcher2::getDistanceOfPointsOptical(cv::Point2f& first, cv::Point2f& second)
{
    float x1 = first.x;
    float y1 = first.y;
    float x2 = second.x;
    float y2 = second.y;
    return sqrt(pow(y2-y1,2) + pow(x2-x1,2));
}

void RobustMatcher2::findAverageDistanceOfPoints(ImageFrame& first, ImageFrame& second)
{
    float dist = 0.0f;
    float angle[4] {0.0f};
    int count[4] {0};
    for (size_t i = 0; i < first.optPoints.size(); i++)
    {
        dist += getDistanceOfPointsOptical(first.optPoints[i],second.optPoints[i]);
        angle[first.class_id[i]] += getAngleOfPoints(first.optPoints[i],second.optPoints[i]);
        count[first.class_id[i]] += 1;
    }
    averageDistance = dist/first.optPoints.size();
    for (size_t i = 0; i < sizeof(count)/sizeof(count[0]); i++)
    {
        if (count[i] != 0)
        {
            averageAngle[i] = angle[i]/count[i];
        }
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
    // testImageRectify();
    // testFeatureExtraction();
    // testDisparityWithOpticalFlow();
    // testFeatureMatching();
    // testFeatureMatchingWithOpticalFlow();
    // testOpticalFlowWithPairs();
    // testFeatureExtractorClassWithCallback();
    testFeatureExtractorClass();

}

void RobustMatcher2::testFeatureExtractorClassWithCallback()
{
    FeatureExtractor trial;
    // FeatureExtractor trial(FeatureExtractor::FeatureChoice::ORB,1000,8,1.2f,10, 20, 6, true);
    int i {0};
    ros::Time prevStamp;
    const int times {600};
    ProcessTime orb("orb");
    while(i < times)
    {
        if (trialL.header.stamp != prevStamp)
        {
            prevStamp = trialL.header.stamp;
            // leftImage.getImage(i, "left");
            // rightImage.getImage(i, "right");
            trialL.rectifyImage(rmap[0][0],rmap[0][1]);
            trialR.rectifyImage(rmap[1][0],rmap[1][1]);
            std::vector <cv::KeyPoint> fastKeys, rightKeys;
            cv::Mat lDesc, rDesc;
            // ProcessTime their("their");
            // for (int i =0;i < 8;i++)
            // {
            //     fastKeys.clear();
            //     cv::FAST(leftImage.image,fastKeys,20,true);
            // }
            // their.totalTime();
            // cv::Mat outImage2;
            // cv::drawKeypoints(leftImage.image, fastKeys,outImage2);
            // cv::imshow("fast",outImage2);

            // ProcessTime mine("mine");
            // for (int i =0;i < 14;i++)
            // {
            // fastKeys.clear();
            // rightKeys.clear();
            orb.start = clock();
            trial.findORB(trialL.image, fastKeys, lDesc);
            trial.findORB(trialR.image, rightKeys, rDesc);
            orb.totalTime();
            // }
            // mine.totalTime();
            i++;
            cv::Mat outImage, outImageR;
            cv::drawKeypoints(trialL.image, fastKeys,outImage);
            cv::imshow("left",outImage);
            cv::drawKeypoints(trialR.image, rightKeys,outImageR);
            cv::imshow("right",outImageR);
            cv::waitKey(1);
        }
    }
    orb.averageTimeOverTimes(times);
    
}

void RobustMatcher2::testFeatureExtractorClass()
{
    FeatureExtractor trial;
    FeatureMatcher matcher(zedcamera->mHeight, trial.getGridRows(), trial.getGridCols());
    // FeatureExtractor trial(FeatureExtractor::FeatureChoice::ORB,1000,8,1.2f,10, 20, 6, true);
    int i {0};
    const int times {600};
    Timer all("all");
    
    while(i < times)
    {
        
        leftImage.getImage(i, "left");
        rightImage.getImage(i, "right");
        leftImage.rectifyImage(rmap[0][0],rmap[0][1]);
        rightImage.rectifyImage(rmap[1][0],rmap[1][1]);
        std::vector <cv::KeyPoint> leftKeys, rightKeys;
        cv::Mat lDesc, rDesc;
        std::vector <cv::DMatch> matches;

        {

        Timer extr("Feature Extraction Took");

        // trial.findORB(leftImage.image, leftKeys, lDesc);
        // trial.findORB(rightImage.image, rightKeys, rDesc);
        trial.findFAST(leftImage.image, leftKeys, lDesc);
        trial.findFAST(rightImage.image, rightKeys, rDesc);

        Timer matchTimer("Feature Matching Took");
        matcher.stereoMatch(leftImage.image, rightImage.image, leftKeys, rightKeys,lDesc, rDesc, matches);
        }
        // trial.findORBWithCV(rightImage.image, rightKeys);
        // trial.findORBWithCV(leftImage.image, fastKeys);

        // std::thread left(&vio_slam::FeatureExtractor::findORB, std::ref(trial),std::ref(rightImage.image),std::ref(rightKeys), std::ref(rDesc));
        // trial.findORB(leftImage.image, fastKeys, lDesc);
        // left.join();

        // auto d = std::async(std::launch::async, &vio_slam::FeatureExtractor::findORB,std::ref(trial), std::ref(rightImage.image),std::ref(rightKeys), std::ref(rDesc));
        // d.wait();
        // auto s = std::async(std::launch::async, &vio_slam::FeatureExtractor::findORB,std::ref(trial), std::ref(leftImage.image),std::ref(fastKeys), std::ref(lDesc));
        // s.wait();



        i++;
        cv::Mat outImage, outImageR;
        cv::drawKeypoints(leftImage.image, leftKeys,outImage);
        cv::imshow("left",outImage);
        cv::drawKeypoints(rightImage.image, rightKeys,outImageR);
        cv::imshow("right",outImageR);
        cv::Mat outImageMatches;
        drawFeatureMatchesStereo(matches,leftImage.realImage, leftKeys,rightKeys,outImageMatches);
        cv::imshow("Matches",outImageMatches);
        cv::waitKey(1);

    }
    

    
}

void RobustMatcher2::calcP1P2()
{
    if (!kitti)
    {
        cv::Mat a1, a2;
        cv::hconcat(cv::Mat::eye(cv::Size(3,3),CV_64FC1),cv::Mat::zeros(cv::Size(1,3),CV_64FC1),a1);
        cv::hconcat(zedcamera->sensorsRotate,zedcamera->sensorsTranslate,a2);
        
        P1 = zedcamera->cameraLeft.cameraMatrix * a1;
        P2 = zedcamera->cameraRight.cameraMatrix * a2;

        Logging("P1",P1,1);
        Logging("P2",P2,1);
    }
    else
    {
        P1 = (cv::Mat_<double>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
        P2 = (cv::Mat_<double>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
        Logging("P1",P1,1);
        Logging("P2",P2,1);
    }
}

void ImageFrame::findFeaturesOnImage(int frameNumber, const char* whichImage, cv::Mat& map1, cv::Mat& map2)
{
    getImage(frameNumber, whichImage);
    rectifyImage(map1, map2);
    findFeaturesORBAdaptive();
}

} // namespace vio_slam