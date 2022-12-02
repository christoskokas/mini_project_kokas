#include "Map.h"

namespace vio_slam
{

MapPoint::MapPoint(Eigen::Vector4d& p, const cv::Mat& _desc, cv::KeyPoint& obsK, bool _close, const int _kdx, const int _idx) : close(_close), wp(p), kdx(_kdx), idx(_idx)
{
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
    obs.push_back(obsK);
    desc.push_back(_desc);
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

Eigen::Vector4d MapPoint::getWordPose4d()
{
    return wp;
}

Eigen::Vector3d MapPoint::getWordPose3d()
{
    return wp3d;
}



void MapPoint::setWordPose4d(Eigen::Vector4d& p)
{
    wp = p;
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
}

void MapPoint::addTCnt()
{
    trackCnt++;
}

void MapPoint::updateMapPoint(Eigen::Vector4d& p, const cv::Mat& _desc, cv::KeyPoint& _obs)
{
    setWordPose4d(p);
    addTCnt();
    obs.push_back(_obs);
    desc.push_back(_desc);

}

void Map::addMapPoint(Eigen::Vector4d& p, const cv::Mat& _desc, cv::KeyPoint& obsK, bool _useable)
{
    MapPoint* mp = new MapPoint(p, _desc, obsK, _useable, kIdx, pIdx);
    mapPoints.insert(std::pair<unsigned long, MapPoint*>(pIdx, mp));
    pIdx++;
}

void Map::addKeyFrame(Eigen::Matrix4d _pose)
{
    KeyFrame* kF = new KeyFrame(_pose, kIdx);
    keyFrames.insert(std::pair<unsigned long, KeyFrame*>(kIdx, kF));
    kIdx ++;
}

} // namespace vio_slam