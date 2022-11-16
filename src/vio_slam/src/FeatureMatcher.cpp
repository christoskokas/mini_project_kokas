#include "FeatureMatcher.h"

namespace vio_slam
{

FeatureMatcher::FeatureMatcher(const Zed_Camera* _zed, const int _imageHeight, const int _gridRows, const int _gridCols, const int _stereoYSpan) : zedptr(_zed), imageHeight(_imageHeight), gridRows(_gridRows), gridCols(_gridCols), stereoYSpan(_stereoYSpan), mnDisp(floor((float)_zed->cameraLeft.fx/40))
{

}

void FeatureMatcher::computeOpticalFlow(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, SubPixelPoints& prevPoints, SubPixelPoints& newPoints)
{
    cv::Mat err;
    std::vector <uchar> status;
    cv::calcOpticalFlowPyrLK(prevLeftIm, leftIm, prevPoints.left, newPoints.left, status, err,cv::Size(21,21),3, criteria);

    reduceVectorTemp<int,uchar>(prevPoints.indexes3D,status);
    prevPoints.reduce<uchar>(status);
    newPoints.reduce<uchar>(status);

    // reduceVectorTemp<int,uchar>(prevPoints.indexes2D,status);
    // reduceVectorTemp<cv::Point2f,uchar>(prevPoints.left,status);
    // reduceVectorTemp<cv::Point2f,uchar>(points.left,status);
    // checkVectorTemp<uchar>(prevPoints.useable,status);


    // cv::calcOpticalFlowPyrLK(prevRightIm, rightIm, prevPoints.right, newPoints.right, status, err,cv::Size(21,21),3, criteria);

    // prevPoints.reduce<uchar>(status);
    // newPoints.reduce<uchar>(status);

}

void FeatureMatcher::computeOpticalFlowWithSliding(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, SubPixelPoints& prevPoints, SubPixelPoints& newPoints)
{

    std::vector<bool> status = slidingWindowOpticalFlow(prevLeftIm, leftIm, prevPoints.left,newPoints.left);

    // reduceVectorTemp<cv::Point2f,bool>(prevPoints.left,status);
    reduceVectorTemp<int,bool>(prevPoints.indexes3D,status);
    reduceVectorTemp<int,bool>(prevPoints.indexes2D,status);

}

std::vector<bool> FeatureMatcher::slidingWindowOpticalLR(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::Point2f>& leftPoints, std::vector<cv::Point2f>& rightPoints)
{
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {5};

    std::vector<bool> goodDist;
    const size_t size {leftPoints.size()};
    goodDist.reserve(size);
    // newMatches.reserve(matches.size());
    for (size_t i = 0; i != size; i++)
    {
        int bestDist {INT_MAX};
        int bestX {windowMovementX + 1};
        const int lKeyX {cvRound(leftPoints[i].x)};
        const int lKeyY {cvRound(leftPoints[i].y)};

        std::vector < float > allDists;
        allDists.reserve(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        if ((lRowStart < 0) || (lRowEnd >= leftImage.rows) || (lColStart < 0) || (lColEnd >= leftImage.cols))
            continue;

        const cv::Mat lWin = leftImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
            const int rKeyX {cvRound(rightPoints[i].x)};
            const int rKeyY {cvRound(rightPoints[i].y)};

            const int rRowStart {rKeyY - windowRadius};
            const int rRowEnd {rKeyY + windowRadius + 1};
            const int rColStart {rKeyX + xMov - windowRadius};
            const int rColEnd {rKeyX + xMov + windowRadius + 1};

            if ((rRowStart < 0) || (rRowEnd >= rightImage.rows) || (rColStart < 0) || (rColEnd >= rightImage.cols))
            {
                continue;
            }

            const cv::Mat rWin = rightImage.rowRange(rRowStart,rRowEnd).colRange(rColStart, rColEnd);

            float dist = cv::norm(lWin,rWin, cv::NORM_L1);
            if (bestDist > dist)
            {
                bestX = xMov;
                bestDist = dist;
            }
            allDists.emplace_back(dist);

        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == (windowMovementX + 1)))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];

        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);

        goodDist.push_back(true);


        rightPoints[i].x += bestX + delta;
    }
    return goodDist;
}

std::vector<bool> FeatureMatcher::slidingWindowOpticalFlow(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {7};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {15};
    const int windowMovementY {15};

    std::vector<bool> goodDist;
    goodDist.reserve(prevPoints.size());
    // newMatches.reserve(matches.size());
    int count {0};
    std::vector<cv::Point2f>::const_iterator it, end(prevPoints.end());
    newPoints.resize(prevPoints.size());
    for (it = prevPoints.begin(); it != end; it++, count++)
    {
        int bestDist {INT_MAX};
        int bestX {windowMovementX + 1};
        int bestY {windowMovementY + 1};
        const int lKeyX {cvRound(it->x)};
        const int lKeyY {cvRound(it->y)};

        std::vector < std::vector<float> > allDists;
        allDists.resize(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        bool out {false};

        if ((lRowStart < 0) || (lRowEnd >= prevImage.rows) || (lColStart < 0) || (lColEnd >= prevImage.cols))
            continue;

        const cv::Mat lWin = prevImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
            for (int32_t yMov {-windowMovementY}; yMov < windowMovementY + 1; yMov++)
            {
                const int rKeyX {lKeyX};
                const int rKeyY {lKeyY};

                const int rRowStart {rKeyY + yMov - windowRadius};
                const int rRowEnd {rKeyY + yMov + windowRadius + 1};
                const int rColStart {rKeyX + xMov - windowRadius};
                const int rColEnd {rKeyX + xMov + windowRadius + 1};

                if ((rRowStart < 0) || (rRowEnd >= prevImage.rows) || (rColStart < 0) || (rColEnd >= prevImage.cols))
                {
                    out = true;
                    break;
                }

                const cv::Mat rWin = image.rowRange(rRowStart,rRowEnd).colRange(rColStart, rColEnd);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestY = yMov;
                    bestDist = dist;
                }
                if (yMov == -windowMovementY)
                    allDists[xMov + windowMovementX].reserve(2*windowMovementY);
                allDists[xMov + windowMovementX].emplace_back(dist);
            }
            if (out)
                break;
        }
        if (out)
        {
            goodDist.push_back(false);
            continue;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == (windowMovementX + 1)) || (bestY == -windowMovementY) || (bestY == windowMovementY) || (bestY == (windowMovementY + 1)))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1x = allDists[windowMovementX + bestX-1][bestY + windowMovementY];
        const float dist2x = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3x = allDists[windowMovementX + bestX+1][bestY + windowMovementY];

        const float deltaX = (dist1x-dist3x)/(2.0f*(dist1x+dist3x-2.0f*dist2x));

        const float dist1y = allDists[windowMovementX + bestX][bestY + windowMovementY - 1];
        const float dist2y = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3y = allDists[windowMovementX + bestX][bestY + windowMovementY + 1];

        const float deltaY = (dist1y-dist3y)/(2.0f*(dist1y+dist3y-2.0f*dist2y));

        if (deltaX > 1 || deltaX < -1 || deltaY > 1 || deltaY < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);

        goodDist.push_back(true);


        // newPoints.push_back(cv::Point2f(it->x + bestX + deltaX, it->y + bestY + deltaY));

        newPoints[count].x = it->x + bestX + deltaX;
        newPoints[count].y = it->y + bestY + deltaY;

    }
    return goodDist;
}

std::vector<bool> FeatureMatcher::slidingWindowOptical(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {4};
    const int windowMovementY {4};

    std::vector<bool> goodDist;
    goodDist.reserve(prevPoints.size());
    // newMatches.reserve(matches.size());
    int count {0};
    std::vector<cv::Point2f>::const_iterator it, end(prevPoints.end());
    for (it = prevPoints.begin(); it != end; it++, count++)
    {
        int bestDist {INT_MAX};
        int bestX {windowMovementX + 1};
        int bestY {windowMovementY + 1};
        const int lKeyX {cvRound(it->x)};
        const int lKeyY {cvRound(it->y)};

        std::vector < std::vector<float> > allDists;
        allDists.resize(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        bool out {false};

        if ((lRowStart < 0) || (lRowEnd >= prevImage.rows) || (lColStart < 0) || (lColEnd >= prevImage.cols))
            continue;

        const cv::Mat lWin = prevImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
            for (int32_t yMov {-windowMovementY}; yMov < windowMovementY + 1; yMov++)
            {
                const int rKeyX {cvRound(newPoints[count].x)};
                const int rKeyY {cvRound(newPoints[count].y)};

                const int rRowStart {rKeyY + yMov - windowRadius};
                const int rRowEnd {rKeyY + yMov + windowRadius + 1};
                const int rColStart {rKeyX + xMov - windowRadius};
                const int rColEnd {rKeyX + xMov + windowRadius + 1};

                if ((rRowStart < 0) || (rRowEnd >= prevImage.rows) || (rColStart < 0) || (rColEnd >= prevImage.cols))
                {
                    out = true;
                    break;
                }

                const cv::Mat rWin = image.rowRange(rRowStart,rRowEnd).colRange(rColStart, rColEnd);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestY = yMov;
                    bestDist = dist;
                }
                if (yMov == -windowMovementY)
                    allDists[xMov + windowMovementX].reserve(2*windowMovementY);
                allDists[xMov + windowMovementX].emplace_back(dist);
            }
            if (out)
                break;
        }
        if (out)
        {
            goodDist.push_back(false);
            continue;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == (windowMovementX + 1)) || (bestY == -windowMovementY) || (bestY == windowMovementY) || (bestY == (windowMovementY + 1)))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1x = allDists[windowMovementX + bestX-1][bestY + windowMovementY];
        const float dist2x = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3x = allDists[windowMovementX + bestX+1][bestY + windowMovementY];

        const float deltaX = (dist1x-dist3x)/(2.0f*(dist1x+dist3x-2.0f*dist2x));

        const float dist1y = allDists[windowMovementX + bestX][bestY + windowMovementY - 1];
        const float dist2y = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3y = allDists[windowMovementX + bestX][bestY + windowMovementY + 1];

        const float deltaY = (dist1y-dist3y)/(2.0f*(dist1y+dist3y-2.0f*dist2y));

        if (deltaX > 1 || deltaX < -1 || deltaY > 1 || deltaY < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);

        goodDist.push_back(true);


        newPoints[count].x += bestX + deltaX;
        newPoints[count].y += bestY + deltaY;

    }
    return goodDist;
}

std::vector<bool> FeatureMatcher::slidingWindowOpticalBackUp(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {3};
    const int windowMovementY {3};

    std::vector<bool> goodDist;
    goodDist.reserve(prevPoints.size());
    // newMatches.reserve(matches.size());
    int count {0};
    std::vector<cv::Point2f>::const_iterator it, end(prevPoints.end());
    for (it = prevPoints.begin(); it != end; it++, count++)
    {
        int bestDist {INT_MAX};
        int bestX {windowMovementX + 1};
        int bestY {windowMovementY + 1};
        const int lKeyX {cvRound(it->x)};
        const int lKeyY {cvRound(it->y)};

        std::vector < std::vector<float> > allDists;
        allDists.resize(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        bool out {false};

        if ((lRowStart < 0) || (lRowEnd >= prevImage.rows) || (lColStart < 0) || (lColEnd >= prevImage.cols))
            continue;

        const cv::Mat lWin = prevImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
            for (int32_t yMov {-windowMovementY}; yMov < windowMovementY + 1; yMov++)
            {
                const int rKeyX {cvRound(newPoints[count].x)};
                const int rKeyY {cvRound(newPoints[count].y)};

                const int rRowStart {rKeyY + yMov - windowRadius};
                const int rRowEnd {rKeyY + yMov + windowRadius + 1};
                const int rColStart {rKeyX + xMov - windowRadius};
                const int rColEnd {rKeyX + xMov + windowRadius + 1};

                if ((rRowStart < 0) || (rRowEnd >= prevImage.rows) || (rColStart < 0) || (rColEnd >= prevImage.cols))
                {
                    out = true;
                    break;
                }

                const cv::Mat rWin = image.rowRange(rRowStart,rRowEnd).colRange(rColStart, rColEnd);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestY = yMov;
                    bestDist = dist;
                }
                if (yMov == -windowMovementY)
                    allDists[xMov + windowMovementX].reserve(2*windowMovementY);
                allDists[xMov + windowMovementX].emplace_back(dist);
            }
            if (out)
                break;
        }
        if (out)
        {
            goodDist.push_back(false);
            continue;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == (windowMovementX + 1)) || (bestY == -windowMovementY) || (bestY == windowMovementY) || (bestY == (windowMovementY + 1)) || bestDist > 1000)
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1x = allDists[windowMovementX + bestX-1][bestY + windowMovementY];
        const float dist2x = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3x = allDists[windowMovementX + bestX+1][bestY + windowMovementY];

        const float deltaX = (dist1x-dist3x)/(2.0f*(dist1x+dist3x-2.0f*dist2x));

        const float dist1y = allDists[windowMovementX + bestX][bestY + windowMovementY - 1];
        const float dist2y = allDists[windowMovementX + bestX][bestY + windowMovementY];
        const float dist3y = allDists[windowMovementX + bestX][bestY + windowMovementY + 1];

        const float deltaY = (dist1y-dist3y)/(2.0f*(dist1y+dist3y-2.0f*dist2y));

        if (deltaX > 1 || deltaX < -1 || deltaY > 1 || deltaY < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);

        goodDist.push_back(true);


        newPoints[count].x += bestX + deltaX;
        newPoints[count].y += bestY + deltaY;

    }
    return goodDist;
}

bool FeatureMatcher::checkProjection(Eigen::Vector4d& point, cv::Point2d& kp)
{
    
    point = zedptr->cameraPose.poseInverse * point;
    const double &pointX = point(0);
    const double &pointY = point(1);
    const double &pointZ = point(2);

    if (pointZ <= 0.0f)
        return false;

    const double invZ = 1.0f/pointZ;
    const double fx = zedptr->cameraLeft.fx;
    const double fy = zedptr->cameraLeft.fy;
    const double cx = zedptr->cameraLeft.cx;
    const double cy = zedptr->cameraLeft.cy;

    const double invfx = 1.0f/fx;
    const double invfy = 1.0f/fy;


    double u {fx*pointX*invZ + cx};
    double v {fy*pointY*invZ + cy};


    const int min {0};
    const int maxW {zedptr->mWidth};
    const int maxH {zedptr->mHeight};

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

    kp = cv::Point2d(u, v);

    return true;

}

void FeatureMatcher::removeWithFund(SubPixelPoints& prevPoints, SubPixelPoints& points)
{
    std::vector<uchar> inliers;
    cv::findFundamentalMat(prevPoints.left, points.left, inliers, cv::FM_RANSAC, 0.5, 0.99);

    prevPoints.reduce<uchar>(inliers);
    points.reduce<uchar>(inliers);
    reduceVectorTemp<int,uchar>(prevPoints.indexes3D,inliers);

    // reduceVectorTemp<cv::Point2f,uchar>(prevPoints.left,inliers);
    
    // reduceVectorTemp<int,uchar>(prevPoints.indexes2D,inliers);

    // reduceVectorTemp<cv::Point2f,uchar>(prevPoints.left,status);
    // reduceVectorTemp<cv::Point2f,uchar>(points.left,status);
    // checkVectorTemp<uchar>(prevPoints.useable,inliers);
    points.useable = prevPoints.useable;


    // cv::findFundamentalMat(prevPoints.right, points.right, inliers, cv::FM_RANSAC, 1, 0.99);

    // prevPoints.reduce<uchar>(inliers);
    // points.reduce<uchar>(inliers);
}

void FeatureMatcher::computeRightPoints(const SubPixelPoints& prevPoints, SubPixelPoints& points)
{
    const size_t size {prevPoints.left.size()};
    points.right.resize(size);
    for (size_t i {0};i < size;i++)
    {
        const float difX {prevPoints.left[i].x - points.left[i].x};
        const float difY {prevPoints.left[i].y - points.left[i].y};
        points.right[i].x = prevPoints.right[i].x - difX;
        points.right[i].y = prevPoints.right[i].y - difY;
    }

    // calculate depth


}

void FeatureMatcher::outlierRejection(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, const cv::Mat& rightIm, SubPixelPoints& prevPoints, SubPixelPoints& points)
{
    // std::vector<bool> optCheck = slidingWindowOptical(prevLeftIm, leftIm, prevPoints.left, points.left);

    // prevPoints.reduce<bool>(optCheck);
    // points.reduce<bool>(optCheck);
    removeWithFund(prevPoints, points);

    // computeRightPoints(prevPoints, points);

    // std::vector<bool> optCheckR = slidingWindowOpticalLR(leftIm, rightIm, points.left, points.right);

    // prevPoints.reduce<bool>(optCheckR);
    // points.reduce<bool>(optCheckR);

    // cv::cornerSubPix(rightIm,points.right,cv::Size(5,5),cv::Size(-1,-1),criteria);
    // cv::cornerSubPix(leftIm,points.left,cv::Size(5,5),cv::Size(-1,-1),criteria);


    // computeDepth(prevPoints, points);
}

void FeatureMatcher::computeDepth(SubPixelPoints& prevPoints, SubPixelPoints& points)
{
    const size_t size {points.left.size()};
    std::vector<bool>check;
    check.resize(size);
    points.depth.resize(size);
    points.useable.resize(size);
    for (size_t i {0};i < size;i++)
    {
        // Compute Depth
        const float disparity {points.left[i].x - points.right[i].x};

        if (disparity > 0.0f)
        {
            check[i] = true;
            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false, depth is unusable
            if (depth < zedptr->mBaseline * 40)
            {
                points.depth[i] = depth;
                points.useable[i] = true;
                continue;
            }
            points.depth[i] = 0.0f;
            points.useable[i] = false;

        }
        else
        {
            points.depth[i] = 0.0f;
            points.useable[i] = false;
        }
    }

    points.reduce<bool>(check);
    prevPoints.reduce<bool>(check);
}

std::vector<bool> FeatureMatcher::inlierDetection(std::vector < cv::Point3d>& first, std::vector < cv::Point3d>& second, std::vector <cv::Point2d>& toReduce)
{

    const size_t end = second.size();
    std::vector<std::pair<int,int>>sums;
    sums.reserve(end);

    std::vector <bool>inliers(end,false);


    const float maxDist {0.5f};

    for (size_t i {0};i < second.size();i++)
    {

        std::vector <bool>indxes(end,false);
        int count {0};
        for (size_t j {1};j < second.size() - 1;j++)
        {
            indxes[i] = true;
            if (abs(computeDistanceOf3DPoints(first[i],first[j]) - computeDistanceOf3DPoints(second[i],second[j])) < maxDist)
            {
                indxes[j] = true;
                count ++;
            }
        }
        if (count > second.size()/2)
        {
            inliers = indxes;
            break;
        }



        // if (abs(computeDistanceOf3DPoints(first[i],first[i+1]) - computeDistanceOf3DPoints(second[i],second[i+1])) < maxDist)
        // {
        //     inliers[i] = true;
        //     inliers[i+1] = true;
        //     i += 2;
        // }
        // else 
        // {
        //     if (abs(computeDistanceOf3DPoints(first[i],first[i+2]) - computeDistanceOf3DPoints(second[i],second[i+2])) < maxDist)
        //     {
        //         inliers[i] = true;
        //         inliers[i+2] = true;
        //     }
        //     else if (abs(computeDistanceOf3DPoints(first[i+1],first[i+2]) - computeDistanceOf3DPoints(second[i+1],second[i+2])) < maxDist)
        //     {
        //         inliers[i+1] = true;
        //         inliers[i+2] = true;
        //     }
        //     i += 3;
        // }
    }
    return inliers;
}

double FeatureMatcher::computeDistanceOf3DPoints(cv::Point3d& first, cv::Point3d& second)
{
    return (pow(first.x-second.x,2) + pow(first.y-second.y,2) + pow(first.z-second.z,2));
}

double FeatureMatcher::computeDistanceThreshold(const cv::Point3d& p1, const cv::Point3d& p2, const double L1, const double L2)
{

    const double DL1 {computeDL(p1,p2,L1)};
    const double DL2 {computeDL(p2,p1,L2)};

    return 3*sqrt(pow(DL1,2) + pow(DL2,2));

}

double FeatureMatcher::computeDL(const cv::Point3d& p1,const cv::Point3d& p2, const double L)
{
    const float focal {2.12};
    const float De {0.2f};

    const double bs {(double)zedptr->mBaseline};
    const double A {pow((p1.x-p2.x)*(bs-p1.x)-(p1.y-p2.y)*p1.y-(p1.z-p2.z)*p1.z,2)};
    const double B {pow((p1.x-p2.x)*p1.x+(p1.y-p2.y)*p1.y+(p1.z-p2.z)*p1.z,2)};
    const double C {pow(bs*(p1.y-p2.y),2)/2};
    const double D {pow((p1.x-p2.x)*(bs-p2.x)-(p1.y-p2.y)*p2.y-(p1.z-p2.z)*p2.z,2)};
    const double E {pow((p1.x-p2.x)*p2.x+(p1.y-p2.y)*p2.y+(p1.z-p2.z)*p2.z,2)};
    const double F {C};

    const double DL {De*sqrt(pow(p1.z,2)*(A+B+C) + pow(p2.z,2)*(D+E+F))/(L*focal*bs)};
    return DL;

}

std::vector<bool> FeatureMatcher::getMaxClique( const std::vector<cv::Point3d>& ptsA, const std::vector<cv::Point3d>& ptsB )
{

    assert( ptsA.size() == ptsB.size() );

    const int32_t numMatches = ptsA.size();

    cv::Mat  consistencyMatrix    = cv::Mat::zeros( numMatches, numMatches, CV_8U );
    uint8_t* consistencyMatrixPtr = consistencyMatrix.ptr<uint8_t>();

    //TODO: Make this threshold adaptive to quadratic depth error
    std::vector<int> nodeDegrees( numMatches, 0 );

    for( int i = 0; i < numMatches; ++i )
    {
        for( int j = i+1, index = (i*numMatches)+j; j < numMatches; ++j, ++index )
        {

            const double L1 {cv::norm( ptsA[i] - ptsA[j] )};
            const double L2 {cv::norm( ptsB[i] - ptsB[j] )};

            const double distThresh {computeDistanceThreshold(ptsA[i], ptsB[i],L1,L2)};

            const double absDistance = abs( L1 - L2 );

            if( absDistance <= distThresh )
            {
                consistencyMatrixPtr[index] = 1;
                ++nodeDegrees[i];
                ++nodeDegrees[j];
            }
        }
    }

    // Fnd the largest set of mutually consistent matches:
    //
    // This is equivalent to ﬁnding the maximum clique on a graph with
    // adjacency matrix W. Since the maximum clique problem is known to
    // be NP-complete, we use the following sub-optimal algorithm:
    //
    // 1) Initialize the clique to contain the match with the largest
    //    number of consistent matches (i.e., choose the node with the
    //    maximum degree).
    // 2) Find the set of matches compatible with all the matches already
    //    in the clique.
    // 3) Add the match with the largest number consistent matches.
    //
    // Repeat (2) and (3) until the set of compatible matches is empty.

    const int maxNodeIndex = std::distance( nodeDegrees.begin(),
                                            std::max_element( nodeDegrees.begin(), nodeDegrees.end() ) );

    // We need to make this matrix not just upper triangular and so we
    // must 'complete' the Consistency Matrix:
    consistencyMatrix += consistencyMatrix.t();

    std::vector<int> candidates = consistencyMatrix.row( maxNodeIndex );

    std::vector<int> candidatesIndices;
    candidatesIndices.reserve( nodeDegrees[ maxNodeIndex ] );

    for( int i = 0; i < numMatches; ++i )
    {
        if( candidates[i] > 0 )
        {
            candidatesIndices.push_back( i );
        }
    }

    std::vector<bool> maxClique;
    maxClique.resize( nodeDegrees[ maxNodeIndex ] );
    maxClique[maxNodeIndex] = true;

    while( !candidatesIndices.empty() )
    {
        // Find the candidate with largest 'consistent' degree:
        int maxIndex  = -1;
        int maxDegree = -1;
        for( int i = 0; i < candidatesIndices.size(); ++i )
        {
            const int degree = cv::countNonZero( consistencyMatrix.row( candidatesIndices[i] ) );

            if( degree > maxDegree )
            {
                maxIndex  = candidatesIndices[i];
                maxDegree = degree;
            }
        }

        maxClique[maxIndex] = true;
        // maxClique.push_back( maxIndex );

        // New clique addition at consistencyMatrix(maxIndex,maxIndex) is now and
        // always zero, so it'll be erased:
        candidatesIndices.erase( std::remove_if( candidatesIndices.begin(), candidatesIndices.end(), 
                                                [=]( const int index )
                                                {
                                                    return consistencyMatrixPtr[ index*numMatches + maxIndex ] == 0;
                                                } ),
                                std::end( candidatesIndices ) );

    }


    return maxClique;
}

void FeatureMatcher::addUcharVectors(std::vector <uchar>& first, std::vector <uchar>& second)
{
    const size_t size {first.size()};
    for (size_t i {0};i < size;i++)
        first[i] = (first[i] & second[i]);
}

void FeatureMatcher::computeStereoMatches(const cv::Mat& leftImage, const cv::Mat& rightImage, const StereoDescriptors& desc, std::vector <cv::DMatch>& matches, SubPixelPoints& points, StereoKeypoints& keypoints)
{
    std::vector<std::vector < int > > indexes;
    std::vector<cv::DMatch> tempMatches;
    destributeRightKeys(keypoints.right, indexes);
    matchPoints(desc, indexes,tempMatches,points, keypoints);

    slidingWindowOptimization(leftImage, rightImage, matches, tempMatches,points);

}

void FeatureMatcher::computeStereoMatchesClose(const cv::Mat& leftImage, const cv::Mat& rightImage, const StereoDescriptors& desc, std::vector <cv::DMatch>& matches, SubPixelPoints& points, StereoKeypoints& keypoints)
{
    std::vector<std::vector < int > > indexes;
    std::vector<cv::DMatch> tempMatches;
    destributeRightKeys(keypoints.right, indexes);
    matchPoints(desc, indexes,tempMatches,points, keypoints);

    slidingWindowOptimizationClose(leftImage, rightImage, matches, tempMatches,points);

}

void FeatureMatcher::stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches, SubPixelPoints& points)
{
    // Timer("stereo match");
    std::vector<std::vector < int > > indexes;
    
    destributeRightKeys(rightKeys, indexes);

    std::vector<cv::DMatch> tempMatches;
    matchKeys(leftKeys, rightKeys, indexes, leftDesc, rightDesc, tempMatches);

    slidingWindowOpt(leftImage, rightImage, matches, tempMatches, leftKeys, rightKeys, points);

    // Logging("matches size",matches.size(),2);
}

void FeatureMatcher::matchPoints(const StereoDescriptors& desc, const std::vector<std::vector < int > >& indexes, std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points, StereoKeypoints& keypoints)
{
    // Timer("Matching took");
    points.left.reserve(keypoints.left.size());
    points.right.reserve(keypoints.right.size());
    std::vector<int> matchedDist(keypoints.right.size(),256);
    std::vector<int> matchedLIdx(keypoints.right.size(),-1);
    int leftRow {0};
    tempMatches.reserve(keypoints.left.size());
    int matchesCount {0};

    const float minZ = zedptr->mBaseline;
    const float minD = 0;
    const float maxD = zedptr->mBaseline * zedptr->cameraLeft.fx/minZ;

    std::vector < cv::KeyPoint >::const_iterator it,end(keypoints.left.end());
    for (it = keypoints.left.begin(); it != end; it++, leftRow++)
    {
        const int yKey = cvRound(it->pt.y);
        const float uL {it->pt.y};

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = 256;
        int bestIdx = -1;
        // int secDist = 256;
        // int secIdx = -1;
        const int lGrid {it->class_id};

        // If the idx has already been checked because they are repeated in indexes vector
        // std::vector <int> checkedIdxes;
        // checkedIdxes.reserve(500);

        for (int32_t iKey = yKey - stereoYSpan; iKey < yKey + stereoYSpan + 1; iKey ++)
        {
            int count {0};
            const size_t endCount {indexes[iKey].size()};
            if (endCount == 0)
                continue;
            for (size_t allIdx {0};allIdx < endCount; allIdx++)
            {
                const int idx {indexes[iKey][allIdx]};
                {
                    // const size_t endidxes {checkedIdxes.size()};
                    // bool found {false};
                    // for (size_t checkIdx {0}; checkIdx < endidxes; checkIdx++)
                    //     if (idx == checkedIdxes[checkIdx])
                    //     {
                    //         found = true;
                    //         break;
                    //     }
                    // if (found)
                    //     continue;
                    // checkedIdxes.push_back(idx);
                    const float uR {keypoints.right[idx].pt.y};
                    if(!(uR>=minU && uR<=maxU))
                        continue;

                    // int maxDif {0};
                    // if ( gridCols >= 10)
                    //     maxDif = floor(gridCols/10);
                    // else
                    //     maxDif = 1;
                    // const int rGrid {keypoints.right[idx].class_id};
                    // const int difGrid {lGrid - rGrid};
                    // const int leftSide {lGrid%(gridCols + 1)};
                    // const int rightSide {rGrid%(gridCols + 1)};
                    // if (((leftSide - rightSide > maxDif) || (leftSide < rightSide)))
                    //     continue;
                }

                int dist {DescriptorDistance(desc.left.row(leftRow),desc.right.row(idx))};

                if (bestDist > dist)
                {
                    // secDist = bestDist;
                    // secIdx = bestIdx;
                    bestDist = dist;
                    bestIdx = idx;
                    continue;
                }
                // if (secDist > dist)
                // {
                //     secDist = dist;
                //     secIdx = idx;
                // }
            }
        }
        if (bestDist > thDist)
            continue;
            // if (bestDist < 0.8f*secDist)
            // {
        if (matchedDist[bestIdx] != 256)
        {
            if (bestDist < matchedDist[bestIdx])
            {
                points.left[matchedLIdx[bestIdx]] = it->pt;
                points.right[matchedLIdx[bestIdx]] = keypoints.right[bestIdx].pt;
                tempMatches[matchedLIdx[bestIdx]] = cv::DMatch(matchedLIdx[bestIdx],matchedLIdx[bestIdx],bestDist);
                matchedDist[bestIdx] = bestDist;
                // matchedKeys.lIdx[bestIdx] = matchesCount;

            }
            continue;
        }
        else
        {
            matchedLIdx[bestIdx] = matchesCount;
            matchedDist[bestIdx] = bestDist;
            points.left.emplace_back(it->pt);
            points.right.emplace_back(keypoints.right[bestIdx].pt);
            tempMatches.emplace_back(matchedLIdx[bestIdx],matchedLIdx[bestIdx],bestDist);
            matchesCount ++;
        }

            // }
    }
}

void FeatureMatcher::matchKeys(std::vector < cv::KeyPoint >& leftKeys, std::vector < cv::KeyPoint >& rightKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& tempMatches)
{
    // Timer("Matching took");
    MatchedKeysDist matchedKeys(rightKeys.size(),256, -1);
    int leftRow {0};
    tempMatches.reserve(leftKeys.size());
    int matchesCount {0};
    std::vector < cv::KeyPoint >::const_iterator it,end(leftKeys.end());
    for (it = leftKeys.begin(); it != end; it++, leftRow++)
    {
        const int yKey = cvRound(it->pt.y);

        int bestDist = 256;
        int bestIdx = -1;
        int secDist = 256;
        int secIdx = -1;
        const int lGrid {it->class_id};

        // If the idx has already been checked because they are repeated in indexes vector
        std::vector <int> checkedIdxes;
        checkedIdxes.reserve(500);

        for (int32_t iKey = yKey - stereoYSpan; iKey < yKey + stereoYSpan + 1; iKey ++)
        {
            int count {0};
            const size_t endCount {indexes[iKey].size()};
            if (endCount == 0)
                continue;
            for (size_t allIdx {0};allIdx < endCount; allIdx++)
            {
                const int idx {indexes[iKey][allIdx]};
                {
                    const size_t endidxes {checkedIdxes.size()};
                    bool found {false};
                    for (size_t checkIdx {0}; checkIdx < endidxes; checkIdx++)
                        if (idx == checkedIdxes[checkIdx])
                        {
                            found = true;
                            break;
                        }
                    if (found)
                        continue;
                    checkedIdxes.push_back(idx);
                    
                    const int rGrid {rightKeys[idx].class_id};
                    const int difGrid {lGrid - rGrid};
                    const int leftSide {lGrid%gridCols};
                    if (!((difGrid <= 1) && lGrid >= rGrid && leftSide != 0 ))
                        continue;
                }

                int dist {DescriptorDistance(leftDesc.row(leftRow),rightDesc.row(idx))};

                if (bestDist > dist)
                {
                    secDist = bestDist;
                    secIdx = bestIdx;
                    bestDist = dist;
                    bestIdx = idx;
                    continue;
                }
                if (secDist > dist)
                {
                    secDist = dist;
                    secIdx = idx;
                }
            }
        }
        if (bestDist > 100)
            continue;
        if (bestIdx != -1)
        {
            if (bestDist < 0.8f*secDist)
            {
                if (matchedKeys.dist[bestIdx] != 256)
                {
                    if (bestDist < matchedKeys.dist[bestIdx])
                    {
                        tempMatches[matchedKeys.lIdx[bestIdx]] = cv::DMatch(leftRow,bestIdx,bestDist);
                        matchedKeys.dist[bestIdx] = bestDist;
                        // matchedKeys.lIdx[bestIdx] = matchesCount;

                    }
                    continue;
                }
                else
                {
                    matchedKeys.lIdx[bestIdx] = matchesCount;
                    matchedKeys.dist[bestIdx] = bestDist;
                    tempMatches.emplace_back(leftRow,bestIdx,bestDist);
                    matchesCount ++;
                }

            }
        }
    }
}

void FeatureMatcher::slidingWindowOpt(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, SubPixelPoints& points)
{
    // Timer("sliding Widow took");
    // Because we use FAST to detect Features the sliding window will get a window of side 7 pixels (3 pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {3};

    const int distThreshold {1500};

    std::vector<bool> goodDist;
    goodDist.reserve(tempMatches.size());
    matches.reserve(tempMatches.size());
    points.left.reserve(tempMatches.size());
    points.right.reserve(tempMatches.size());
    // newMatches.reserve(matches.size());
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++)
    {
        int bestDist {INT_MAX};
        int bestX {-1};
        int bestY {-1};
        const int lKeyX {(int)leftKeys[it->queryIdx].pt.x};
        const int lKeyY {(int)leftKeys[it->queryIdx].pt.y};

        std::vector < float > allDists;
        allDists.reserve(2*windowMovementX);

        const cv::Mat lWin = leftImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(lKeyX - windowRadius, lKeyX + windowRadius + 1);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
                const int rKeyX {(int)rightKeys[it->trainIdx].pt.x};
                const int rKeyY {(int)rightKeys[it->trainIdx].pt.y};
                const cv::Mat rWin = rightImage.rowRange(rKeyY - windowRadius, rKeyY + windowRadius + 1).colRange(rKeyX + xMov - windowRadius, rKeyX + xMov + windowRadius + 1);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestDist = dist;
                }
                allDists.emplace_back(dist);
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == -1))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];

        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);

        goodDist.push_back(true);


        rightKeys[it->trainIdx].pt.x += bestX + delta;
    }
    }


    {
    int count {0};
    int matchesCount {0};
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++, count++ )
        if (goodDist[count])
        {
            points.left.emplace_back(leftKeys[it->queryIdx].pt);
            points.right.emplace_back(rightKeys[it->trainIdx].pt);
            matches.emplace_back(matchesCount, matchesCount, it->distance);
            matchesCount += 1;
        }
    }
    Logging("matches size", matches.size(),2);
    // cv::cornerSubPix(leftImage, points.left,cv::Size(3,3),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,0.001));
    // cv::cornerSubPix(rightImage, points.right,cv::Size(3,3),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,0.001));
    
    // count = 0;
    // std::vector<cv::DMatch>::const_iterator it, end(matches.end());
    // for (it = matches.begin(); it != end; it++, count++ )
    // {
    //     leftKeys[it->queryIdx].pt = points.left[count];
    //     rightKeys[it->trainIdx].pt = points.right[count];
    //     Logging("left",leftKeys[it->queryIdx].pt,2);
    //     Logging("right",rightKeys[it->trainIdx].pt,2);
    // }

    // std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    // for (it = tempMatches.begin(); it != end; it++, count++ )
    //     if (goodDist[count])
    //         matches.emplace_back(it->queryIdx, it->trainIdx, it->distance);

}

void FeatureMatcher::slidingWindowOptimization(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {3};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {5};

    std::vector<bool> goodDist;
    std::vector < std::pair<int,int> > allBestDists;
    const size_t tsize {tempMatches.size()};
    allBestDists.reserve(tsize);
    goodDist.reserve(tsize);
    matches.reserve(tsize);
    points.depth.reserve(tsize);
    points.useable.reserve(tsize);
    // newMatches.reserve(matches.size());
    int matchesCount {0};
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++)
    {
        int bestDist {INT_MAX};
        int bestX {0};
        const float lKeyX {round(points.left[it->queryIdx].x)};
        const float lKeyY {round(points.left[it->queryIdx].y)};

        std::vector < float > allDists;
        allDists.resize(2*windowMovementX + 1);

        const cv::Mat lWin = leftImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(lKeyX - windowRadius, lKeyX + windowRadius + 1);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
                const float rKeyX {round(points.right[it->trainIdx].x)};
                // const float rKeyY {round(points.right[it->trainIdx].y)};
                const cv::Mat rWin = rightImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(rKeyX + xMov - windowRadius, rKeyX + xMov + windowRadius + 1);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestDist = dist;
                }
                allDists[xMov + windowMovementX] = dist;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];

        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);
        allBestDists.emplace_back(std::make_pair(matchesCount,bestDist));
        matchesCount++;
        

        points.right[it->trainIdx].y = points.left[it->queryIdx].y;
        points.right[it->trainIdx].x = round(points.right[it->trainIdx].x) +  bestX + delta;

        // calculate depth
        const float disparity {points.left[it->queryIdx].x - points.right[it->trainIdx].x};
        if (disparity > 0.0f && disparity < zedptr->cameraLeft.fx)
        {
            goodDist.push_back(true);

            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false depth is unusable
            points.depth.emplace_back(depth);
            if (depth < zedptr->mBaseline * closeNumber)
            {
                points.useable.emplace_back(true);
                continue;
            }
            points.useable.emplace_back(false);

            // Logging("depth",depth,2);
        }
        else
        {
            goodDist.push_back(false);
        }
    }
    }


    {
    reduceVectorTemp<cv::Point2f,bool>(points.left,goodDist);
    reduceVectorTemp<cv::Point2f,bool>(points.right,goodDist);
    int count {0};
    int matchesCount {0};
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++, count++ )
        if (goodDist[count])
        {
            matches.emplace_back(matchesCount, matchesCount, it->distance);
            matchesCount += 1;
        }
    }
    // Logging("matches size", matches.size(),2);
}

void FeatureMatcher::slWinGF(const cv::Mat& leftImage, const cv::Mat& rightImage, SubPixelPoints& points)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {5};

    std::vector<bool> goodDist;
    std::vector < std::pair<int,int> > allBestDists;
    const size_t tsize {points.left.size()};
    allBestDists.reserve(tsize);
    goodDist.reserve(tsize);
    points.depth.reserve(tsize);
    points.useable.reserve(tsize);
    // newMatches.reserve(matches.size());
    int matchesCount {0};
    {
    for (size_t iP {0}; iP < tsize; iP ++)
    {
        int bestDist {INT_MAX};
        int bestX {0};
        const int lKeyX {(int)points.left[iP].x};
        const int lKeyY {(int)points.left[iP].y};
        const int rKeyX {(int)points.right[iP].x};
        const int rKeyY {(int)points.right[iP].y};

        std::vector < float > allDists;
        allDists.resize(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        if ((lRowStart < 0) || (lRowEnd >= leftImage.rows) || (lColStart < 0) || (lColEnd >= leftImage.cols))
            continue;

        const cv::Mat lWin = leftImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);

        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
                const int rRowStart {rKeyY - windowRadius};
                const int rRowEnd {rKeyY + windowRadius + 1};
                const int rColStart {rKeyX + xMov - windowRadius};
                const int rColEnd {rKeyX + xMov + windowRadius + 1};

                if ((rRowStart < 0) || (rRowEnd >= rightImage.rows) || (rColStart < 0) || (rColEnd >= rightImage.cols))
                    continue;

                const cv::Mat rWin = rightImage.rowRange(rRowStart, rRowEnd).colRange(rKeyX + xMov - windowRadius, rColEnd);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestDist = dist;
                }
                allDists[xMov + windowMovementX] = dist;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];
        // if (dist1 == 0.0f || dist2 == 0.0f || dist3 == 0.0f);
        // {
        //     goodDist.push_back(false);
        //     continue;
        // }
        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);
        allBestDists.emplace_back(std::make_pair(matchesCount,bestDist));
        matchesCount++;
        


        points.right[iP].x += bestX + delta;

        // calculate depth
        const float disparity {points.left[iP].x - points.right[iP].x};
        if (disparity > 0.0f)
        {
            goodDist.push_back(true);

            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false depth is unusable
            points.depth.emplace_back(depth);
            if (depth < zedptr->mBaseline * 40)
            {
                points.useable.emplace_back(true);
                continue;
            }
            points.useable.emplace_back(false);

            // Logging("depth",depth,2);
        }
        else
        {
            goodDist.push_back(false);
        }
    }
    }

    reduceVectorTemp<cv::Point2f,bool>(points.left,goodDist);
    reduceVectorTemp<cv::Point2f,bool>(points.right,goodDist);

    // Logging("matches size", matches.size(),2);
}

void FeatureMatcher::slidingWindowOptimizationClose(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, SubPixelPoints& points)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {3};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {5};

    std::vector<bool> goodDist;
    const size_t tsize {tempMatches.size()};
    goodDist.reserve(tsize);
    matches.reserve(tsize);
    points.depth.reserve(tsize);
    points.useable.reserve(tsize);
    // newMatches.reserve(matches.size());
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++)
    {
        int bestDist {INT_MAX};
        int bestX {0};
        const float lKeyX {round(points.left[it->queryIdx].x)};
        const float lKeyY {round(points.left[it->queryIdx].y)};

        std::vector < float > allDists;
        allDists.resize(2*windowMovementX + 1);

        const cv::Mat lWin = leftImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(lKeyX - windowRadius, lKeyX + windowRadius + 1);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
                const float rKeyX {round(points.right[it->trainIdx].x)};
                // const int rKeyY {(int)points.right[it->trainIdx].y};
                const cv::Mat rWin = rightImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(rKeyX + xMov - windowRadius, rKeyX + xMov + windowRadius + 1);

                float dist = cv::norm(lWin,rWin, cv::NORM_L1);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestDist = dist;
                }
                allDists[xMov + windowMovementX] = dist;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX))
        {
            goodDist.push_back(false);
            continue;
        }
        // const int bestDistIdx {bestX + windowMovementX};
        // float delta = (2*allDists[bestDistIdx] - allDists[bestDistIdx-1] - allDists[bestDistIdx+1])/(2*(allDists[bestDistIdx-1] - allDists[bestDistIdx+1]));

        // Linear Interpolation for sub pixel accuracy
        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];

        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
        {
            goodDist.push_back(false);
            continue;
        }
        // Logging("delta ",delta,2);
        


        points.right[it->trainIdx].y = points.left[it->queryIdx].y;
        points.right[it->trainIdx].x = round(points.right[it->trainIdx].x) + bestX + delta;

        // calculate depth
        const float disparity {points.left[it->queryIdx].x - points.right[it->trainIdx].x};
        if (disparity > 0.0f && disparity < zedptr->cameraLeft.fx)
        {

            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false depth is unusable
            if (depth < zedptr->mBaseline * closeNumber)
            {
                points.depth.emplace_back(depth);
                points.useable.emplace_back(true);
                goodDist.push_back(true);
                continue;
            }
        }
        goodDist.push_back(false);
    }
    }


    {
    reduceVectorTemp<cv::Point2f,bool>(points.left,goodDist);
    reduceVectorTemp<cv::Point2f,bool>(points.right,goodDist);
    int count {0};
    int matchesCount {0};
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++, count++ )
        if (goodDist[count])
        {
            matches.emplace_back(matchesCount, matchesCount, it->distance);
            matchesCount += 1;
        }
    }
    // Logging("matches size", matches.size(),2);
}

void FeatureMatcher::checkDepthChange(const cv::Mat& leftImage, const cv::Mat& rightImage, SubPixelPoints& points)
{
    // Timer("sliding Window took");
    // Because we use FAST to detect Features the sliding window will get a window of side 11 pixels (5 (3 + 2offset) pixels radius + 1 which is the feature)
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {5};


    const size_t end{points.left.size()};
    for (size_t i{0}; i < end; i ++)
    {
        if ( points.useable[i] )
            continue;
        const int pDisp {cvRound(((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/points.depth[i])};
        const int xStart {- windowMovementX};
        const int xEnd {windowMovementX + 1};
        int bestDist {INT_MAX};
        int bestX {0};
        const int lKeyX {(int)points.left[i].x};
        const int lKeyY {(int)points.left[i].y};
        const int rKeyX {(int)points.left[i].x - pDisp};
        const int rKeyY {(int)points.left[i].y};

        std::vector < float > allDists;
        allDists.resize(2*windowMovementX + 1);

        const int lRowStart {lKeyY - windowRadius};
        const int lRowEnd {lKeyY + windowRadius + 1};
        const int lColStart {lKeyX - windowRadius};
        const int lColEnd {lKeyX + windowRadius + 1};

        if ((lRowStart < 0) || (lRowEnd >= leftImage.rows) || (lColStart < 0) || (lColEnd >= leftImage.cols))
            continue;

        const cv::Mat lWin = leftImage.rowRange(lRowStart, lRowEnd).colRange(lColStart, lColEnd);
        for (int32_t xMov {xStart}; xMov <xEnd; xMov++)
        {

            const int rRowStart {rKeyY - windowRadius};
            const int rRowEnd {rKeyY + windowRadius + 1};
            const int rColStart {rKeyX + xMov - windowRadius};
            const int rColEnd {rKeyX + xMov + windowRadius + 1};

            if ((rRowStart < 0) || (rRowEnd >= rightImage.rows) || (rColStart < 0) || (rColEnd >= rightImage.cols))
                continue;

            const cv::Mat rWin = rightImage.rowRange(rRowStart, rRowEnd).colRange(rColStart, rColEnd);

            float dist = cv::norm(lWin,rWin, cv::NORM_L1);
            if (bestDist > dist)
            {
                bestX = xMov;
                bestDist = dist;
            }
            allDists[xMov + windowMovementX] = dist;

        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX))
            continue;

        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];
        // if ( dist1 == 0.0f || dist2 == 0.0f || dist3 == 0.0f )
        //     continue;
        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
            continue;
        // Logging("delta ",delta,2);

        const float pR = rKeyX + bestX + delta;

        // calculate depth
        const float disparity {points.left[i].x - pR};
        if (disparity > 0.0f)
        {
            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false depth is unusable
            points.depth[i] = depth;
            if (depth < zedptr->mBaseline * 40)
            {
                const double fx {zedptr->cameraLeft.fx};
                const double fy {zedptr->cameraLeft.fy};
                const double cx {zedptr->cameraLeft.cx};
                const double cy {zedptr->cameraLeft.cy};

                const double zp = (double)points.depth[i];
                const double xp = (double)(((double)points.left[i].x-cx)*zp/fx);
                const double yp = (double)(((double)points.left[i].y-cy)*zp/fy);
                Eigen::Vector4d p4d(xp,yp,zp,1);
                p4d = zedptr->cameraPose.pose * p4d;
                points.points3D[i] = cv::Point3d(p4d(0),p4d(1),p4d(2));

                points.useable[i] = true;
                continue;
            }
        }
    }
}

int FeatureMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void FeatureMatcher::destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes)
{

    // Timer("distribute keys took");

    indexes.resize(imageHeight);

    for (int32_t i = 0; i < imageHeight; i++)
        indexes[i].reserve(200);

    std::vector<cv::KeyPoint>::const_iterator it,end(rightKeys.end());
    int count {0};
    for (it = rightKeys.begin(); it != end; it++, count ++)
    {
        const int yKey = cvRound((*it).pt.y);

        for (int32_t pos = yKey - stereoYSpan; pos < yKey + stereoYSpan + 1; pos++)
            indexes[pos].emplace_back(count);
    }

}

} // namespace vio_slam