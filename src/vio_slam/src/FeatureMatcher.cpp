#include "FeatureMatcher.h"

namespace vio_slam
{

FeatureMatcher::FeatureMatcher(const int _imageHeight, const int _stereoYSpan) : imageHeight(_imageHeight), stereoYSpan(_stereoYSpan)
{

}

void FeatureMatcher::stereoMatch(const std::vector<cv::KeyPoint>& leftKeys, const std::vector<cv::KeyPoint>& rightKeys, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches)
{
    
    std::vector<std::vector < int > > indexes;
    
    destributeRightKeys(rightKeys, indexes);
    
    std::vector < int > bestIndexes;
    matchKeys(leftKeys, indexes, leftDesc, rightDesc, matches);

}

void FeatureMatcher::matchKeys(const std::vector < cv::KeyPoint >& leftKeys, const std::vector<std::vector < int > >& indexes, const cv::Mat& leftDesc, const cv::Mat& rightDesc, std::vector <cv::DMatch>& matches)
{
    int leftRow {0};
    std::vector < int > bestIndexes;
    bestIndexes.reserve(leftKeys.size());
    matches.reserve(leftKeys.size());
    std::vector < cv::KeyPoint >::const_iterator it,end(leftKeys.end());
    for (it = leftKeys.begin(); it != end; it++, leftRow++)
    {
        const int yKey = cvRound(it->pt.y);

        int bestDist = 256;
        int bestIdx = -1;

        for (int32_t iKey = yKey - stereoYSpan; iKey < yKey + stereoYSpan; iKey ++)
        {
            int count {0};
            const size_t endCount {indexes[iKey].size()};
            for (size_t allIdx {0};allIdx < endCount; allIdx++)
            {
                const int idx {indexes[iKey][allIdx]};
                int dist {DescriptorDistance(leftDesc.row(leftRow),rightDesc.row(idx))};

                if (bestDist > dist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
        }
        bestIndexes.emplace_back(bestIdx);
        matches.emplace_back(leftRow,bestIdx,bestDist);
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

    indexes.resize(imageHeight);

    for (int32_t i = 0; i < imageHeight; i++)
        indexes[i].reserve(200);

    std::vector<cv::KeyPoint>::const_iterator it,end(rightKeys.end());
    int count {0};
    for (it = rightKeys.begin(); it != end; it++, count ++)
    {
        const int yKey = cvRound((*it).pt.y);

        for (int32_t pos = yKey - stereoYSpan; pos < yKey + stereoYSpan; pos++)
            indexes[pos].emplace_back(count);
    }

}

} // namespace vio_slam