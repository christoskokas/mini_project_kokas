#include "FeatureTracker.h"

namespace vio_slam
{

void ImageData::setImage(const int frameNumber, const char* whichImage)
{
    std::string imagePath;
    std::string first;
    std::string second, format;
    std::string t = whichImage;
#if KITTI_DATASET
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/kitti_00/";
    second = "/00";
    format = ".png";
#elif ZED_DATASET
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/zed_exp/";
    second = "/" + t + "00";
    format = ".png";
#else
    first = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/images/";
    second = "/frame";
    format = ".jpg";
#endif

    if (frameNumber > 999)
    {
        imagePath = first + t + second + std::to_string(frameNumber/(int)(pow(10,3))%10) + std::to_string(frameNumber/(int)(pow(10,2))%10) + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
        int i{};
    }
    else if (frameNumber > 99)
    {
        imagePath = first + t + second + "0" + std::to_string(frameNumber/(int)(pow(10,2))%10) + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
    }
    else if (frameNumber > 9)
    {
        imagePath = first + t + second + "00" + std::to_string(frameNumber/(int)(pow(10,1))%10) + std::to_string(frameNumber%10) + format;
    }
    else
    {
        imagePath = first + t + second + "000" + std::to_string(frameNumber) + format;
    }
    im = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
    rIm = cv::imread(imagePath,cv::IMREAD_COLOR);
}

void ImageData::rectifyImage(cv::Mat& image, const cv::Mat& map1, const cv::Mat& map2)
{
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
}

FeatureData::FeatureData(Zed_Camera* _zedPtr) : zedPtr(_zedPtr), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

void FeatureData::compute3DPoints(SubPixelPoints& prePnts, const int keyNumb)
{
    const size_t end{prePnts.left.size()};

    const size_t start{prePnts.points3D.size()};

    prePnts.points3D.reserve(end);
    for (size_t i = start; i < end; i++)
    {   

        const double zp = (double)prePnts.depth[i];
        const double xp = (double)(((double)prePnts.left[i].x-cx)*zp/fx);
        const double yp = (double)(((double)prePnts.left[i].y-cy)*zp/fy);
        Eigen::Vector4d p4d(xp,yp,zp,1);
        p4d = zedPtr->cameraPose.pose * p4d;
        prePnts.points3D.emplace_back(p4d(0),p4d(1),p4d(2));
        
    }
}

FeatureTracker::FeatureTracker(cv::Mat _rmap[2][2], Zed_Camera* _zedPtr) : zedPtr(_zedPtr), fm(zedPtr, zedPtr->mHeight, fe.getGridRows(), fe.getGridCols()), pE(zedPtr), fd(zedPtr), dt(zedPtr->mFps), lkal(dt), datafile(filepath)
{
    rmap[0][0] = _rmap[0][0];
    rmap[0][1] = _rmap[0][1];
    rmap[1][0] = _rmap[1][0];
    rmap[1][1] = _rmap[1][1];
}

void FeatureTracker::setMask(const SubPixelPoints& prePnts, cv::Mat& mask)
{
    const int rad {4};
    mask = cv::Mat(zedPtr->mHeight, zedPtr->mWidth, CV_8UC1, cv::Scalar(255));

    std::vector<cv::Point2f>::const_iterator it, end{prePnts.left.end()};
    for (it = prePnts.left.begin();it != end; it++)
    {
        if (mask.at<uchar>(*it) == 255)
        {
            cv::circle(mask, *it, rad, 0, cv::FILLED);
        }
    }

}

void FeatureTracker::setPopVec(const SubPixelPoints& prePnts, std::vector<int>& pop)
{
    const int gRows {fe.getGridRows()};
    const int gCols {fe.getGridCols()};
    pop.resize(gRows * gCols);
    const int wid {(int)zedPtr->mWidth/gCols + 1};
    const int hig {(int)zedPtr->mHeight/gRows + 1};
    std::vector<cv::Point2f>::const_iterator it, end(prePnts.left.end());
    for (it = prePnts.left.begin(); it != end; it ++)
    {
        const int w {(int)it->x/wid};
        const int h {(int)it->y/hig};
        pop[(int)(w + h*gCols)] += 1;
    }
}

void FeatureTracker::stereoFeaturesPop(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    std::vector<int> pop;
    setPopVec(prePnts, pop);
    fe.extractFeaturesPop(leftIm, rightIm, desc, keys, pop);
    fm.computeStereoMatches(leftIm, rightIm, desc, matches, pnts, keys);
    std::vector<uchar> inliers;
    if ( pnts.left.size() >  6)
    {
        cv::findFundamentalMat(pnts.left, pnts.right, inliers, cv::FM_RANSAC, 1, 0.99);

        pnts.reduce<uchar>(inliers);
        reduceVectorTemp<cv::DMatch,uchar>(matches, inliers);
    }
    Logging("matches size", matches.size(),1);

#if KEYSIM
    drawKeys(pLIm.rIm, keys.left);
    drawKeys(pRIm.rIm, keys.right);
#endif


#if MATCHESIM
    drawMatches(lIm.rIm, pnts, matches);
#endif
}

void FeatureTracker::stereoFeaturesMask(cv::Mat& leftIm, cv::Mat& rightIm, std::vector<cv::DMatch>& matches, SubPixelPoints& pnts, const SubPixelPoints& prePnts)
{
    StereoDescriptors desc;
    StereoKeypoints keys;
    cv::Mat mask;
    setMask(prePnts, mask);
    fe.extractFeaturesMask(leftIm, rightIm, desc, keys, mask);
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
    // addFeatures = checkFeaturesArea(prePnts);
}

void FeatureTracker::beginTracking(const int frames)
{
    for (int32_t frame {1}; frame < frames; frame++)
    {
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

        // Logging("addf", addFeatures,3);
        Logging("ustereo", uStereo,3);

        // getSolvePnPPoseWithEss();

        getPoseCeres();

        setPre();

        addFeatures = checkFeaturesAreaCont(prePnts);
    }
    datafile.close();
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
    const int mnK {1};
    const int mnG {7};
    const size_t endgr {gridCount.size()};
    int count {0};
    for (size_t i{0}; i < endgr; i ++ )
    {
        if ( gridCount[i] > mnK)
            count ++;
    }
    if ( count < mnG)
        return true;
    else
        return false;
}

bool FeatureTracker::checkFeaturesAreaCont(const SubPixelPoints& prePnts)
{
    static int skip = 0;
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
    const int mnK {3};
    const int mnmxG {7};
    const int mnG {3};
    const size_t endgr {gridCount.size()};
    int count {0};
    for (size_t i{0}; i < endgr; i ++ )
    {
        if ( gridCount[i] > mnK)
            count ++;
    }
    if ( count < mnmxG)
        skip++;
    else if (count < mnG)
        return true;
    else
        skip = 0;
    Logging("skip", skip,3);
    Logging("count", count,3);
    if ( skip > 2 || skip == 0)
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
    std::vector<cv::Point2d> outp2D;
    p3D.reserve(end);
    p2D.reserve(end);
    outp2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            if (prePnts.useable[i])
            {
                inliers[i] = true;
                p3D.emplace_back(point);
                p2D.emplace_back(pnts.points2D[i]);
                outp2D.emplace_back(p2dtemp);
            }
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
    // cv::projectPoints(p3D,cv::Mat::eye(3,3, CV_64F),cv::Mat::zeros(3,1, CV_64F),zedPtr->cameraLeft.cameraMatrix,cv::Mat::zeros(5,1, CV_64F),outp2D);
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
    cv::findFundamentalMat(outp2D, p2D, check, cv::FM_RANSAC, 1, 0.99);

    prePnts.reduce<uchar>(check);
    pnts.reduce<uchar>(check);
    reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    reduceVectorTemp<cv::Point3d,uchar>(p3D,check);

    uStereo = p3D.size();
    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);
    if (uStereo > 10)
    {
        //  cv::solvePnP(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true);
        check.clear();
        cv::solvePnPRansac(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true,100,2.0f, 0.999, check);

    }

    // prePnts.reduce<uchar>(check);
    // pnts.reduce<uchar>(check);
    // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point3d,uchar>(p3D,check);
    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    Logging("norm",cv::norm(tvec,pTvec),3);
    Logging("normr",cv::norm(Rvec,pRvec),3);
    if (cv::norm(tvec,pTvec) + cv::norm(Rvec,pRvec) > 1)
    {
        tvec = pTvec;
        Rvec = pRvec;
    }

    if (p3D.size() > mnInKal)
    {
        lkal.fillMeasurements(measurements, tvec, Rvec);
    }
    else
    {
        Logging("less than 50","",3);
    }

    pTvec = tvec;
    pRvec = Rvec;

    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    Logging("measurements",measurements,3);
    Logging("rot",rotation_estimated,3);
    Logging("tra",translation_estimated,3);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();
#if PROJECTIM
    draw2D3D(pLIm.rIm, outp2D, p2D);
#endif
}

void FeatureTracker::getSolvePnPPoseWithEss()
{

    cv::Mat dist = (cv::Mat_<double>(1,5) << 0,0,0,0,0);
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    std::vector<cv::Point3d> p3D;
    std::vector<cv::Point2d> p2D;
    std::vector<cv::Point2d> pp2Dess;
    std::vector<cv::Point2d> p3Dp2D;
    std::vector<cv::Point2d> outp2D;
    p3D.reserve(end);
    p3Dp2D.reserve(end);
    p2D.reserve(end);
    outp2D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            outp2D.emplace_back(pnts.left[i]);
            pp2Dess.emplace_back(prePnts.left[i]);
            if (prePnts.useable[i])
            {

                p3D.emplace_back(point);
                p3Dp2D.emplace_back(p2dtemp);
                p2D.emplace_back(pnts.left[i]);
            }
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);


    // inliers.clear();
    // const size_t endproj{p3D.size()};
    // inliers.resize(endproj);
    // const int wid {zedPtr->mWidth - 1};
    // const int hig {zedPtr->mHeight - 1};
    // for (size_t i{0};i < endproj; i++)
    // {
    //     if (!(outp2D[i].x > wid || outp2D[i].x < 0 || outp2D[i].y > hig || outp2D[i].y < 0))
    //         inliers[i] = true;
    // }

    // prePnts.reduce<bool>(inliers);
    // pnts.reduce<bool>(inliers);
    // reduceVectorTemp<cv::Point2d,bool>(outp2D,inliers);
    // reduceVectorTemp<cv::Point2d,bool>(p2D,inliers);
    // reduceVectorTemp<cv::Point2d,bool>(pp2Dess,inliers);
    // reduceVectorTemp<cv::Point3d,bool>(p3D,inliers);

    // std::vector<uchar> check;
    // cv::findFundamentalMat(p3Dp2D, p2D, check, cv::FM_RANSAC, 1, 0.999);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p3Dp2D,check);
    // reduceVectorTemp<cv::Point3d,uchar>(p3D,check);
    // prePnts.reduce<uchar>(check);
    // pnts.reduce<uchar>(check);
    // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(p2D,check);
    // reduceVectorTemp<cv::Point2d,uchar>(pp2Dess,check);
    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);

    // cv::Mat E = cv::findEssentialMat(pp2Dess,outp2D,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC, 0.999,1.0f);
    // cv::Mat R1es,R2es,tes;
    // cv::decomposeEssentialMat(E,R1es, R2es, tes);
    // cv::Rodrigues(R1es,R1es);
    // cv::Rodrigues(R2es,R2es);
    // const double norm1 {cv::norm(Rvec,R1es)};
    // const double norm2 {cv::norm(Rvec,R2es)};

    // if (norm1 > norm2)
    //     Rvec = R2es;
    // else
    //     Rvec = R1es;

    uStereo = p3D.size();
    if (uStereo > 10)
    {
        //  cv::solvePnP(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true);
        std::vector<int>idxs;
        cv::solvePnPRansac(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec,true,100,2.0f, 0.999, idxs);

        reduceVectorInliersTemp<cv::Point2d,int>(p2D,idxs);
        reduceVectorInliersTemp<cv::Point2d,int>(p3Dp2D,idxs);
        reduceVectorInliersTemp<cv::Point3d,int>(p3D,idxs);
        // cv::solvePnPRefineLM(p3D, p2D,zedPtr->cameraLeft.cameraMatrix, dist,Rvec,tvec);
        // prePnts.reduce<uchar>(check);
        // pnts.reduce<uchar>(check);
        // reduceVectorTemp<cv::Point2d,uchar>(outp2D,check);
        // reduceVectorTemp<cv::Point2d,uchar>(pp2Dess,check);

    }

    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    if (cv::norm(tvec,pTvec) + cv::norm(Rvec,pRvec) > 1)
    {
        tvec = pTvec;
        Rvec = pRvec;
    }

    if (p3D.size() > mnInKal)
    {
        lkal.fillMeasurements(measurements, tvec, Rvec);
    }
    else
    {
        Logging("less than ",mnInKal,3);
    }

    pTvec = tvec;
    pRvec = Rvec;

    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();
#if PROJECTIM
    draw2D3D(pLIm.rIm,p3Dp2D, p2D);
#endif
}

void FeatureTracker::getPoseCeres()
{

    std::vector<cv::Point3d> p3D;

    get3dPointsforPoseAll(p3D);

    cv::Mat Rvec = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1, CV_64F);

    essForMonoPose(Rvec, tvec,p3D);

    if (p3D.size() > 10)
    {
        pnpRansac(Rvec, tvec, p3D);
    }
    // uStereo = p3D.size();
    poseEstKal(Rvec, tvec, p3D.size());

}

void FeatureTracker::compute2Dfrom3D(std::vector<cv::Point3d>& p3D, std::vector<cv::Point2d>& p2D, std::vector<cv::Point2d>& pn2D)
{
    const size_t end {p3D.size()};

    p2D.reserve(end);
    pn2D.reserve(end);

    for (size_t i{0}; i < end ; i ++)
    {
        const double px {p3D[i].x};
        const double py {p3D[i].y};
        const double pz {p3D[i].z};

        const double invZ = 1.0f/pz;
        const double fx = zedPtr->cameraLeft.fx;
        const double fy = zedPtr->cameraLeft.fy;
        const double cx = zedPtr->cameraLeft.cx;
        const double cy = zedPtr->cameraLeft.cy;

        p2D.emplace_back(fx*px*invZ + cx, fy*py*invZ + cy);
        pn2D.emplace_back((double)pnts.left[i].x, (double)pnts.left[i].y);
    }

}

void FeatureTracker::essForMonoPose(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D)
{
    std::vector<uchar> inliers;
    cv::Mat E = cv::findEssentialMat(prePnts.left, pnts.left,zedPtr->cameraLeft.cameraMatrix,cv::FM_RANSAC, 0.999,2.0f, inliers);
    cv::Mat R1es,R2es,tes;
    cv::decomposeEssentialMat(E,R1es, R2es, tes);
    cv::Rodrigues(R1es,R1es);
    cv::Rodrigues(R2es,R2es);
    const double norm1 {cv::norm(Rvec,R1es)};
    const double norm2 {cv::norm(Rvec,R2es)};

    if (norm1 > norm2)
        Rvec = R2es;
    else
        Rvec = R1es;

    pnts.reduce<uchar>(inliers);
    prePnts.reduce<uchar>(inliers);
    reduceVectorTemp<cv::Point3d,uchar>(p3D, inliers);

#if POINTSIM
    drawPoints(lIm.rIm,prePnts.left, pnts.left, "essential");
#endif

}

void FeatureTracker::pnpRansac(cv::Mat& Rvec, cv::Mat& tvec, std::vector<cv::Point3d>& p3D)
{
    // std::vector<cv::Point3d> p3Ddepth;
    // std::vector<cv::Point2d> p2Ddepth;
    // const size_t end {p3D.size()};
    // p3Ddepth.reserve(end);
    // for (size_t i{0};i < end; i ++)
    // {
    //     if ( prePnts.useable[i] )
    //     {
    //         p3Ddepth.emplace_back(p3D[i]);
    //         p2Ddepth.emplace_back((double)pnts.left[i].x,(double)pnts.left[i].y);
    //     }
    // }

    // cv::solvePnP(p3Ddepth, p2Ddepth,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true);

    std::vector<int>idxs;
    cv::solvePnPRansac(p3D, pnts.left,zedPtr->cameraLeft.cameraMatrix, cv::Mat::zeros(5,1,CV_64F),Rvec,tvec,true,100, 8.0f, 0.999, idxs);

    // prePnts.reduceWithInliers<int>(idxs);
    // pnts.reduceWithInliers<int>(idxs);
    reduceVectorInliersTemp<cv::Point3d,int>(p3D,idxs);
    reduceVectorInliersTemp<cv::Point3d,int>(prePnts.points3D,idxs);
    reduceVectorInliersTemp<bool,int>(prePnts.useable,idxs);
    reduceVectorInliersTemp<cv::Point2f,int>(prePnts.left,idxs);
    reduceVectorInliersTemp<float,int>(prePnts.depth,idxs);
    reduceVectorInliersTemp<cv::Point2f,int>(pnts.left,idxs);
    // reduceVectorInliersTemp<cv::Point2d,int>(p2Ddepth,idxs);
    // reduceVectorInliersTemp<cv::Point3d,int>(p3Ddepth,idxs);

    uStereo = p3D.size();

#if PROJECTIM
    std::vector<cv::Point2d> p2D, pn2D;

    compute2Dfrom3D(p3D, p2D, pn2D);

    draw2D3D(pLIm.rIm, p2D, pn2D);
#endif

}

void FeatureTracker::poseEstKal(cv::Mat& Rvec, cv::Mat& tvec, const size_t p3dsize)
{
    cv::Mat measurements = cv::Mat::zeros(6,1, CV_64F);

    if (cv::norm(tvec,pTvec) + cv::norm(Rvec,pRvec) > 1)
    {
        tvec = pTvec;
        Rvec = pRvec;
    }

    if ( p3dsize > mnInKal)
    {
        lkal.fillMeasurements(measurements, tvec, Rvec);
    }
    else
    {
        Logging("less than ",mnInKal,3);
    }

    pTvec = tvec;
    pRvec = Rvec;

    cv::Mat translation_estimated(3, 1, CV_64F);
    cv::Mat rotation_estimated(3, 3, CV_64F);

    lkal.updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
    pE.convertToEigenMat(rotation_estimated, translation_estimated, poseEstFrame);
    publishPose();
}

void FeatureTracker::get3dPointsforPose(std::vector<cv::Point3d>& p3D)
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    p3D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            if (prePnts.useable[i])
                p3D.emplace_back(point);
            else
                p3D.emplace_back(cv::Point3d(prePnts.left[i].x, prePnts.left[i].y, 0.0f));
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
}

void FeatureTracker::get3dPointsforPoseAll(std::vector<cv::Point3d>& p3D)
{
    std::vector<bool> inliers;
    const size_t end {prePnts.points3D.size()};
    inliers.resize(end);
    p3D.reserve(end);
    for (size_t i {0};i < end;i++)
    {
        cv::Point3d point = prePnts.points3D[i];
        cv::Point2d p2dtemp;
        if (checkProjection3D(point,p2dtemp))
        {
            inliers[i] = true;
            p3D.emplace_back(point);
        }
    }
    prePnts.reduce<bool>(inliers);
    pnts.reduce<bool>(inliers);
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

    cv::findFundamentalMat(prePnts.left, pnts.left, inliers, cv::FM_RANSAC, 1, 0.9999);


    prePnts.reduce<uchar>(inliers);
    pnts.reduce<uchar>(inliers);

    const size_t end{pnts.left.size()};
    std::vector<bool> check;
    check.resize(end);
    for (size_t i{0};i < end;i++)
    {
        if (!(pnts.left[i].x > zedPtr->mWidth || pnts.left[i].x < 0 || pnts.left[i].y > zedPtr->mHeight || pnts.left[i].y < 0))
            check[i] = true;
    }

    prePnts.reduce<bool>(check);
    pnts.reduce<bool>(check);

#if OPTICALIM
    drawOptical(lIm.rIm,prePnts.left, pnts.left);
#endif
}

void FeatureTracker::updateKeys(const int frame)
{
    std::vector<cv::DMatch> matches;
    stereoFeaturesPop(pLIm.im, pRIm.im, matches,pnts, prePnts);
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

void FeatureTracker::drawKeys(cv::Mat& im, std::vector<cv::KeyPoint>& keys)
{
    cv::Mat outIm = im.clone();
    for (auto& key:keys)
    {
        cv::circle(outIm, key.pt,2,cv::Scalar(0,255,0));

    }
    cv::imshow("keys", outIm);
    cv::waitKey(waitImKey);
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

void FeatureTracker::drawPoints(const cv::Mat& im, const std::vector<cv::Point2f>& prePnts,const std::vector<cv::Point2f>& pnts, const char* str)
{
    cv::Mat outIm = im.clone();
    const size_t end {prePnts.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, prePnts[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, prePnts[i], pnts[i],cv::Scalar(0,0,255));
        cv::circle(outIm, pnts[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow(str, outIm);
    cv::waitKey(waitImOpt);
}

void FeatureTracker::draw2D3D(const cv::Mat& im, const std::vector<cv::Point2d>& p2Dfp3D, const std::vector<cv::Point2d>& p2D)
{
    cv::Mat outIm = im.clone();
    const size_t end {p2Dfp3D.size()};
    for (size_t i{0};i < end; i ++ )
    {
        cv::circle(outIm, p2Dfp3D[i],2,cv::Scalar(0,255,0));
        cv::line(outIm, p2Dfp3D[i], p2D[i],cv::Scalar(0,0,255));
        cv::circle(outIm, p2D[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("Project", outIm);
    cv::waitKey(waitImPro);

}

bool FeatureTracker::checkProjection3D(cv::Point3d& point3D, cv::Point2d& point2d)
{
    
    // Logging("key",keyFrameNumb,3);
    Eigen::Vector4d point(point3D.x, point3D.y, point3D.z,1);
    // Logging("point",point,3);
    point = zedPtr->cameraPose.poseInverse * point;
    // Logging("point",point,3);
    // Logging("zedPtr",zedPtr->cameraPose.poseInverse,3);
    // Logging("getPose",keyframes[keyFrameNumb].getPose(),3);
    point3D.x = point(0);
    point3D.y = point(1);
    point3D.z = point(2);
    const double pointX = point(0);
    const double pointY = point(1);
    const double pointZ = point(2);

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


    // u = u_distort;
    // v = v_distort;

    point2d = cv::Point2d(u,v);
    

    if (u > maxW || u < 0 || v > maxH || v < 0)
        return false;

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
#if SAVEODOMETRYDATA
    saveData();
#endif
    Logging zed("Zed Camera Pose", zedPtr->cameraPose.pose,3);
}

void FeatureTracker::saveData()
{
    Eigen::Matrix4d mat = zedPtr->cameraPose.pose.transpose();
    for (int32_t i{0}; i < 12; i ++)
    {
        if ( i == 0 )
            datafile << mat(i);
        else
            datafile << " " << mat(i);
    }
    datafile << '\n';
}

} // namespace vio_slam