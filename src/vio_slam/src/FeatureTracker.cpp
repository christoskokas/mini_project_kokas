#include "FeatureTracker.h"

namespace vio_slam
{

void ImageData::setImage(const int frameNumber, const char* whichImage)
{
    std::string imagePath;
    std::string first;
    std::string second, format;
#if KITTI_DATASET
    first = "/home/christos/Downloads/data_odometry_gray/dataset/sequences/00/";
    second = "/00";
    format = ".png";
#else
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/";
    second = "/frame";
    format = ".jpg";
#endif

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
    im = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    rIm = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void ImageData::rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2)
{
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
}

FeatureData::FeatureData(Zed_Camera* zedPtr) : fx(zedPtr->cameraLeft.fx), fy(zedPtr->cameraLeft.fy), cx(zedPtr->cameraLeft.cx), cy(zedPtr->cameraLeft.cy)
{

}

void FeatureData::compute3DPoints(SubPixelPoints& prePnts, const int keyNumb)
{
    const size_t end{prePnts.left.size()};

    const size_t start{prePnts.points3D.size()};

    prePnts.points3D.reserve(end);
    prePnts.kfn.reserve(end);
    for (size_t i = start; i < end; i++)
    {   

        const double zp = (double)prePnts.depth[i];
        const double xp = (double)(((double)prePnts.left[i].x-cx)*zp/fx);
        const double yp = (double)(((double)prePnts.left[i].y-cy)*zp/fy);

        prePnts.points3D.emplace_back(xp,yp,zp);
        prePnts.kfn.emplace_back(keyNumb);
        
    }
}

FeatureTracker::FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr) : zedPtr(_zedPtr), fm(zedPtr, zedPtr->mHeight, fe.getGridRows(), fe.getGridCols()), pE(zedPtr), fd(zedPtr)
{
    rmap[0][0] = _rmap[0][0];
    rmap[0][1] = _rmap[0][1];
    rmap[1][0] = _rmap[1][0];
    rmap[1][1] = _rmap[1][1];
}

void FeatureTracker::stereoFeatures(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    fe.extractFeatures(leftIm, rightIm, desc, keys);
    fm.computeStereoMatches(leftIm, rightIm, desc, matches, pnts, keys);
    std::vector<uchar> inliers;
    cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 3, 0.99);

    pnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
    Logging("matches size", matches.size(),1);
#if MATCHESIM
    drawMatches(lIm.rIm, pnts, matches);
#endif
}

void FeatureTracker::initializeTracking()
{
    startTime = std::chrono::high_resolution_clock::now();
    setLRImages(0);
    std::vector<cv::DMatch> matches;
    stereoFeatures(lIm.im, rIm.im, matches,pnts);
    cv::Mat rot = (cv::Mat_<double>(3,3) << 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
    uStereo = pnts.left.size();
    uMono = uStereo;
    pE.setPrevR(rot);
    cv::Mat tr = (cv::Mat_<double>(3,1) << 0.0,0.0,0.0);
    pE.setPrevT(tr);
    setPreInit();
    fd.compute3DPoints(prePnts, keyNumb);
    uStereo = prePnts.points3D.size();
    keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
    keyNumb++;
    addFeatures = checkFeaturesArea(prePnts);
}

void FeatureTracker::beginTracking(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
        float dt {calcDt()};
        setLRImages(frame);
        if (addFeatures || uStereo < mnSize)
        {
            zedPtr->addKeyFrame = true;
            updateKeys(frame);
            fd.compute3DPoints(prePnts, keyNumb);
            keyframes.emplace_back(zedPtr->cameraPose.pose,prePnts.points3D,keyNumb);
            keyNumb ++;
            
        }
        opticalFlow();



        getSolvePnPPose();

        setPre();

        addFeatures = checkFeaturesArea(prePnts);
    }
}

bool FeatureTracker::checkFeaturesArea(const SubPixelPoints& prePnts)
{
    const size_t end{prePnts.left.size()};
    const int sep {3};
    std::vector<int> gridCount;
    gridCount.resize(sep * sep);
    const int wid {(int)zedPtr->mWidth/sep + 1};
    const int hig {(int)zedPtr->mHeight/sep + 1};
    for (size_t i{0};i < end; i++)
    {
        const int w {(int)prePnts.left[i].x/wid};
        const int h {(int)prePnts.left[i].y/hig};
        gridCount[(int)(h + sep*w)] += 1;
    }
    const int mnK {5};
    const int mnG {4};
    const size_t endgr {gridCount.size()};
    int count {0};
    for (size_t i{0}; i < endgr; i ++ )
    {
        if ( gridCount[i] > mnK)
            count ++;
    }
    if ( count > (sep * sep - mnG))
        return false;
    else
        return true;
}

void FeatureTracker::getEssentialPose()
{
    cv::Mat Rvec(3,3,CV_64F), tvec(3,1,CV_64F);
    std::vector <uchar> inliers;
    std::vector<cv::Point2f> p, pp;
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    
    
    cv::undistortPoints(pnts.left,p,zedPtr->cameraLeft.cameraMatrix, dist);
    cv::undistortPoints(prePnts.left,pp,zedPtr->cameraLeft.cameraMatrix, dist);
    cv::Mat E = cv::findEssentialMat(prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC,0.99,0.1, inliers);
    if (!inliers.empty())
    {
        prePnts.reduce<uchar>(inliers);
        pnts.reduce<uchar>(inliers);
        reduceVectorTemp<cv::Point2f,uchar>(p,inliers);
        reduceVectorTemp<cv::Point2f,uchar>(pp,inliers);
    }
    uStereo = prePnts.left.size();
    if (uStereo > 10)
    {
        cv::Mat R1,R2,t;
        cv::decomposeEssentialMat(E, R1, R2,t);
        if (cv::norm(prevR,R1) > cv::norm(prevR,R2))
            Rvec = R2;
        else
            Rvec = R1;
        prevR = Rvec.clone();
        tvec = -t/10;
        convertToEigen(Rvec,tvec,poseEstFrame);
        Logging("R1",R1,3);
        Logging("R2",R2,3);
        publishPose();

    }
    
}

void FeatureTracker::getSolvePnPPose()
{

    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;
    p3D.reserve(end);
    p2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        if (checkProjection3D(point,prePnts.kfn[i]))
        {
            inliers[i] = true;
            p3D.emplace_back(point);
            p2D.emplace_back(pnts.points2D[i]);
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
    std::vector<cv::Point2d> outp2D;
    cv::projectPoints(p3D,cv::Mat::eye(3,3, CV_64F),cv::Mat::zeros(3,1, CV_64F),zedPtr->cameraLeft.cameraMatrix,cv::Mat::zeros(5,1, CV_64F),outp2D);
    inliers.clear();
    const size_t endproj{p3D.size()};
    inliers.resize(endproj);
    const int wid {zedPtr->mWidth - 1};
    const int hig {zedPtr->mHeight - 1};
    for (size_t i{0};i < endproj; i++)
    {
        if (!(outp2D[i].x > wid || outp2D[i].x < 0 || outp2D[i].y > hig || outp2D[i].y < 0))
            inliers[i] = true;
    }

    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
    reduceVectorTemp<cv::Point2d,bool>(outp2D,inliers);
    reduceVectorTemp<cv::Point2d,bool>(p2D,inliers);
    reduceVectorTemp<cv::Point3d,bool>(p3D,inliers);

    std::vector<uchar> check;
    cv::findFundamentalMat(outp2D, p2D, check, cv::FM_RANSAC, 3, 0.99);

    prePnts.reduce<uchar>(check);
    pnts.reduce<uchar>(check);
    reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    reduceVectorTemp<cv::Point3d,uchar>(p3D,check);

    uStereo = p3D.size();
    if (uStereo > 10)
    {
        cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);
         cv::solvePnP(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,false);
        //  cv::solvePnPRansac(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,false,100,2.0f);
        pE.convertToEigenMat(Rvec, tvec, poseEstFrame);
        publishPose();

    }
#if PROJECTIM
    draw2D3D(pLIm.rIm,p3D, p2D);
#endif
}

void FeatureTracker::opticalFlow()
{
    std::vector<float> err, err1;
    std::vector <uchar>  inliers;
    cv::calcOpticalFlowPyrLK(pLIm.im, lIm.im, prePnts.left, pnts.left, inliers, err,cv::Size(21,21),3, criteria);

    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);
    reduceVectorTemp<float,uchar>(err,inliers);

    const float minErrValue {20.0f};

    prePnts.reduceWithValue<float>(err, minErrValue);
    pnts.reduceWithValue<float>(err, minErrValue);

    // inliers.clear();
    // cv::findFundamentalMat(pnts.left, prePnts.left, inliers, cv::FM_RANSAC, 3, 0.99);


    // prePnts.reduce<uchar>(inliers);
    // pnts.reduce<uchar>(inliers);

    const size_t end{pnts.left.size()};
    for (size_t i{0};i < end;i++)
        pnts.points2D.emplace_back((double)pnts.left[i].x,(double)pnts.left[i].y);

#if OPTICALIM
    drawOptical(lIm.rIm,prePnts.left, pnts.left);
#endif

}

void FeatureTracker::updateKeys(const int frame)
{
    std::vector<cv::DMatch> matches;
    stereoFeatures(pLIm.im, pRIm.im, matches,pnts);
    prePnts.addLeft(pnts);
    pnts.clear();
}

float FeatureTracker::calcDt()
{
    endTime =  std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    startTime = std::chrono::high_resolution_clock::now();
    return duration.count();
}

void FeatureTracker::setLRImages(const int frameNumber)
{
    lIm.setImage(frameNumber,"left");
    rIm.setImage(frameNumber,"right");
    if (!zedPtr->rectified)
        rectifyLRImages();
}

void FeatureTracker::setLImage(const int frameNumber)
{
    lIm.setImage(frameNumber,"left");
    if (!zedPtr->rectified)
        rectifyLImage();
    
}

void FeatureTracker::setPreLImage()
{
    pLIm.im = lIm.im.clone();
    pLIm.rIm = lIm.rIm.clone();
}

void FeatureTracker::setPreRImage()
{
    pRIm.im = rIm.im.clone();
    pRIm.rIm = rIm.rIm.clone();
}

void FeatureTracker::setPre()
{
    setPreLImage();
    setPreRImage();
    prePnts.left = pnts.left;
    clearPre();
}

void FeatureTracker::setPreInit()
{
    setPreLImage();
    setPreRImage();
    prePnts.clone(pnts);
    clearPre();
}

void FeatureTracker::clearPre()
{
    pnts.clear();
}

cv::Mat FeatureTracker::getLImage()
{
    return lIm.im;
}

cv::Mat FeatureTracker::getRImage()
{
    return rIm.im;
}

cv::Mat FeatureTracker::getPLImage()
{
    return pLIm.im;
}

cv::Mat FeatureTracker::getPRImage()
{
    return pRIm.im;
}

void FeatureTracker::rectifyLRImages()
{
    lIm.rectifyImage(lIm.im, rmap[0][0], rmap[0][1]);
    lIm.rectifyImage(lIm.rIm, rmap[0][0], rmap[0][1]);
    rIm.rectifyImage(rIm.im, rmap[1][0], rmap[1][1]);
    rIm.rectifyImage(rIm.rIm, rmap[1][0], rmap[1][1]);
}

void FeatureTracker::rectifyLImage()
{
    lIm.rectifyImage(lIm.im, rmap[0][0], rmap[0][1]);
    lIm.rectifyImage(lIm.rIm, rmap[0][0], rmap[0][1]);
}

void FeatureTracker::drawMatches(const cv::Mat& lIm, const SubPixelPoints& pnts, const std::vector<cv::DMatch> matches)
{
    cv::Mat outIm = lIm.clone();
    for (auto m:matches)
    {
        cv::circle(outIm, pnts.left[m.queryIdx],2,cv::Scalar(0,255,0));
        cv::line(outIm, pnts.left[m.queryIdx], pnts.right[m.trainIdx],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts.right[m.trainIdx],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Matches", outIm);
    cv::waitKey(waitImMat);
}

void FeatureTracker::drawOptical(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts)
{
    cv::Mat outIm = im.clone();
    const size_t end {prePnts.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, prePnts[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, prePnts[i], pnts[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Optical", outIm);
    cv::waitKey(waitImOpt);
}

void FeatureTracker::draw2D3D(const cv::Mat& im, const std::vector<cv::Point3d>& p3D, const std::vector<cv::Point2d>& p2D)
{
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    cv::Mat R = (cv::Mat_<double>(3,3) << 1,0,0,0,1,0,0,0,1);
    cv::Mat t = (cv::Mat_<double>(1,3) << 0,0,0);
    std::vector<cv::Point2d> out;
    cv::projectPoints(p3D,R,t,zedPtr->cameraLeft.cameraMatrix,dist,out);

    cv::Mat outIm = im.clone();
    const size_t end {out.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, out[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, out[i], p2D[i],cv::Scalar(0,0,255));
        cv::circle(outIm, p2D[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Project", outIm);
    cv::waitKey(waitImPro);

}

bool FeatureTracker::checkProjection3D(cv::Point3d& point3D, const int keyFrameNumb)
{
    

    Eigen::Vector4d point(point3D.x, point3D.y, point3D.z,1);
    point = zedPtr->cameraPose.poseInverse * keyframes[keyFrameNumb].getPose() * point;
    point3D.x = point(0);
    point3D.y = point(1);
    point3D.z = point(2);
    const double &pointX = point(0);
    const double &pointY = point(1);
    const double &pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;

    const double invZ = 1.0f/pointZ;
    const double fx = zedPtr->cameraLeft.fx;
    const double fy = zedPtr->cameraLeft.fy;
    const double cx = zedPtr->cameraLeft.cx;
    const double cy = zedPtr->cameraLeft.cy;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};


    const int min {0};
    const int maxW {zedPtr->mWidth};
    const int maxH {zedPtr->mHeight};

    if (u < min || u > maxW)
        return false;
    if (v < min || v > maxH)
        return false;

    // const double k1 = zedptr->cameraLeft.distCoeffs.at<double>(0,0);
    // const double k2 = zedptr->cameraLeft.distCoeffs.at<double>(0,1);
    // const double p1 = zedptr->cameraLeft.distCoeffs.at<double>(0,2);
    // const double p2 = zedptr->cameraLeft.distCoeffs.at<double>(0,3);
    // const double k3 = zedptr->cameraLeft.distCoeffs.at<double>(0,4);

    const double k1 {0};
    const double k2 {0};
    const double p1 {0};
    const double p2 {0};
    const double k3 {0};

    double u_distort, v_distort;

    double x = (u - cx) * invfx;
    double y = (v - cy) * invfy;
    
    double r2 = x * x + y * y;

    // Radial distorsion
    double x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    double y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;



    return true;

}

void FeatureTracker::convertToEigen(cv::Mat& Rvec, cv::Mat& tvec, Eigen::Matrix4d& tr)
{
    Eigen::Matrix3d Reig;
    Eigen::Vector3d teig;
    cv::cv2eigen(Rvec.t(),Reig);
    cv::cv2eigen(-tvec,teig);

    tr.setIdentity();
    tr.block<3,3>(0,0) = Reig;
    tr.block<3,1>(0,3) = teig;
}

void FeatureTracker::publishPose()
{
poseEst = poseEst * poseEstFrame;
zedPtr->cameraPose.setPose(poseEst);
zedPtr->cameraPose.setInvPose(poseEst.inverse());

Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
}

} // namespace vio_slam