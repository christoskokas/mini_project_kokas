#include "Map.h"

namespace vio_slam
{

MapPoint::MapPoint(Eigen::Vector4d& p, const int _kdx, const int _idx) : wp(p), kdx(_kdx), idx(_idx)
{
    
}

bool MapPoint::GetInFrame()
{
    return inFrame;
}

bool MapPoint::GetIsOutlier()
{
    return isOutlier;
}

void MapPoint::SetInFrame(bool infr)
{
    inFrame = infr;
}

void MapPoint::SetIsOutlier(bool isOut)
{
    isOutlier = isOut;
}

Eigen::Vector4d MapPoint::getWordPose()
{
    return wp;
}

void Map::addMapPoint(Eigen::Vector4d& p)
{
    MapPoint* mp = new MapPoint(p, kIdx, pIdx);
    mapPoints.insert(std::pair<unsigned long, MapPoint*>(pIdx, mp));
    pIdx++;
}

void Map::addKeyFrame(Eigen::Matrix4d _pose, const int _numb)
{
    KeyFrame* kF = new KeyFrame(_pose, _numb);
    keyFrames.insert(std::pair<unsigned long, KeyFrame*>(kIdx, kF));
    kIdx ++;
}

} // namespace vio_slam