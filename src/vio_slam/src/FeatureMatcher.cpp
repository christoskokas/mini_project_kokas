#include "FeatureMatcher.h"

namespace vio_slam
{

FeatureMatcher::FeatureMatcher(const Zed_Camera* _zed, const int _imageHeight, const int _gridRows, const int _gridCols, const int _stereoYSpan) : zedptr(_zed), imageHeight(_imageHeight), gridRows(_gridRows), gridCols(_gridCols), stereoYSpan(_stereoYSpan)
{

}

void FeatureMatcher::computeOpticalFlow(const cv::Mat& prevLeftIm, const cv::Mat& leftIm, const std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints, std::vector <uchar>& status)
{
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(prevLeftIm, leftIm, prevPoints, newPoints, status, err,cv::Size(21,21),3,cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 100, (0.00100000000000000021)));
}

void FeatureMatcher::slidingWindowOptical(const cv::Mat& prevImage, const cv::Mat& image, std::vector<cv::Point2f>& prevPoints, std::vector<cv::Point2f>& newPoints)
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
    {
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

        if ((lRowStart < 0) || (lRowEnd > prevImage.rows) || (lColStart < 0) || (lColEnd > prevImage.cols))
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

                if ((rRowStart < 0) || (rRowEnd > prevImage.rows) || (rColStart < 0) || (rColEnd > prevImage.cols))
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
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestX == (windowMovementX + 1)) || (bestY == -windowMovementY) || (bestY == windowMovementY) || (bestY == (windowMovementY + 1)) || bestDist > 1500)
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
    }

    reduceVectorTemp<cv::Point2f,bool>(prevPoints,goodDist);
    reduceVectorTemp<cv::Point2f,bool>(newPoints,goodDist);

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
    std::vector < cv::KeyPoint >::const_iterator it,end(keypoints.left.end());
    for (it = keypoints.left.begin(); it != end; it++, leftRow++)
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
                    
                    const int rGrid {keypoints.right[idx].class_id};
                    const int difGrid {lGrid - rGrid};
                    const int leftSide {lGrid%gridCols};
                    if (!((difGrid <= 1) && lGrid >= rGrid && leftSide != 0 ))
                        continue;
                }

                int dist {DescriptorDistance(desc.left.row(leftRow),desc.right.row(idx))};

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

            }
        }
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
    const int windowRadius {5};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {3};

    std::vector<bool> goodDist;
    goodDist.reserve(tempMatches.size());
    matches.reserve(tempMatches.size());
    points.depth.reserve(tempMatches.size());
    points.useable.reserve(tempMatches.size());
    // newMatches.reserve(matches.size());
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++)
    {
        int bestDist {INT_MAX};
        int bestX {windowMovementX + 1};
        const int lKeyX {(int)points.left[it->queryIdx].x};
        const int lKeyY {(int)points.left[it->queryIdx].y};

        std::vector < float > allDists;
        allDists.reserve(2*windowMovementX);

        const cv::Mat lWin = leftImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(lKeyX - windowRadius, lKeyX + windowRadius + 1);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
                const int rKeyX {(int)points.right[it->trainIdx].x};
                const int rKeyY {(int)points.right[it->trainIdx].y};
                const cv::Mat rWin = rightImage.rowRange(rKeyY - windowRadius, rKeyY + windowRadius + 1).colRange(rKeyX + xMov - windowRadius, rKeyX + xMov + windowRadius + 1);

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


        points.right[it->trainIdx].x += bestX + delta;

        // calculate depth
        const float disparity {points.left[it->queryIdx].x - points.right[it->trainIdx].x};
        if (disparity > 0.0f)
        {
            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            // if false depth is unusable
            if (depth < zedptr->mBaseline * 40)
            {
                points.useable.emplace_back(true);
                points.depth.emplace_back(depth);
                continue;
            }
            points.depth.emplace_back(0.0f);
            points.useable.emplace_back(false);

            // Logging("depth",depth,2);
        }
        else
        {
            points.depth.emplace_back(0.0f);
            points.useable.emplace_back(false);
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
    Logging("matches size", matches.size(),2);
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