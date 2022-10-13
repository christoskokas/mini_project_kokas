#include "FeatureMatcher.h"

namespace vio_slam
{

FeatureMatcher::FeatureMatcher(const int _imageHeight, const int _gridRows, const int _gridCols, const int _stereoYSpan) : imageHeight(_imageHeight), gridRows(_gridRows), gridCols(_gridCols), stereoYSpan(_stereoYSpan)
{

}

void FeatureMatcher::stereoMatch(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches)
{
    // Timer("stereo match");
    std::vector<std::vector < int > > indexes;
    
    destributeRightKeys(rightKeys, indexes);

    std::vector<cv::DMatch> tempMatches;
    matchKeys(leftKeys, rightKeys, indexes, leftDesc, rightDesc, tempMatches);

    slidingWindowOpt(leftImage, rightImage, matches, tempMatches, leftKeys, rightKeys);

    // Logging("matches size",matches.size(),2);
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
        if (bestDist > 60)
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

void FeatureMatcher::slidingWindowOpt(const cv::Mat& leftImage, const cv::Mat& rightImage, std::vector <cv::DMatch>& matches, const std::vector <cv::DMatch>& tempMatches, std::vector<cv::KeyPoint>& leftKeys, std::vector<cv::KeyPoint>& rightKeys)
{
    // Timer("sliding Widow took");
    // Because we use FAST to detect Features the sliding window will get a window of side 7 pixels (3 pixels radius + 1 which is the feature)
    const int windowLength {7};
    const int windowRadius {3};
    // Because of the EdgeThreshold used around the image we dont need to check out of bounds

    const int windowMovementX {3};
    const int windowMovementY {stereoYSpan};

    std::vector<bool> goodDist;
    goodDist.reserve(tempMatches.size());
    matches.reserve(tempMatches.size());
    // newMatches.reserve(matches.size());
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++)
    {
        double bestDist {250.0};
        int bestX {-1};
        int bestY {-1};
        const int lKeyX {(int)leftKeys[it->queryIdx].pt.x};
        const int lKeyY {(int)leftKeys[it->queryIdx].pt.y};
        const cv::Mat lWin = leftImage.rowRange(lKeyY - windowRadius, lKeyY + windowRadius + 1).colRange(lKeyX - windowRadius, lKeyX + windowRadius + 1);
        for (int32_t xMov {-windowMovementX}; xMov < windowMovementX + 1; xMov++)
        {
            for (int32_t yMov {-windowMovementY}; yMov < windowMovementY + 1; yMov++)
            {
                

                const int rKeyX {(int)rightKeys[it->trainIdx].pt.x};
                const int rKeyY {(int)rightKeys[it->trainIdx].pt.y};
                if (abs(lKeyY - rKeyY) > 1)
                    continue;
                const cv::Mat rWin = rightImage.rowRange(rKeyY + yMov - windowRadius, rKeyY + yMov + windowRadius + 1).colRange(rKeyX + xMov - windowRadius, rKeyX + xMov + windowRadius + 1);

                double dist = cv::norm(lWin,rWin);
                if (bestDist > dist)
                {
                    bestX = xMov;
                    bestY = yMov;
                    bestDist = dist;
                }
            }
        }
        // Logging("bestdist ",bestDist,2);
        if ((bestX == -windowMovementX) || (bestX == windowMovementX) || (bestY == windowMovementY) || (bestY == windowMovementY) || (bestX == -1) || (bestDist >= 80))
        {
            goodDist.push_back(false);
            continue;
        }
        goodDist.push_back(true);

        rightKeys[it->trainIdx].pt.x += bestX;
        rightKeys[it->trainIdx].pt.y += bestY;
    }
    }


    int count {0};
    {
    std::vector<cv::DMatch>::const_iterator it, end(tempMatches.end());
    for (it = tempMatches.begin(); it != end; it++, count++ )
        if (goodDist[count])
            matches.emplace_back(it->queryIdx, it->trainIdx, it->distance);
    }
    Logging("matches size", matches.size(),2);
    // cv::cornerSubPix(leftImage, points.left,cv::Size(5,5),cv::Size(1,1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,0.001));
    // cv::cornerSubPix(rightImage, points.right,cv::Size(5,5),cv::Size(1,1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,0.001));
    
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