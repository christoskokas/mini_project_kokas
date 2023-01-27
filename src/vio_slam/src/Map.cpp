#include "Map.h"

namespace vio_slam
{

MapPoint::MapPoint(const Eigen::Vector4d& p, const cv::Mat& _desc, const cv::KeyPoint& obsK, const bool _close, const unsigned long _kdx, const unsigned long _idx) : close(_close), wp(p), kdx(_kdx), idx(_idx)
{
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
    obs.push_back(obsK);
    desc.push_back(_desc);
}

MapPoint::MapPoint(const unsigned long _idx, const unsigned long _kdx) : idx(_idx), kdx(_kdx)
{}

int MapPoint::predictScale(float dist)
{
    float dif = maxScaleDist/dist;
    // std::cout << "max" << maxScaleDist << std::endl;
    // std::cout << "prev scaleL" << lastObsL.octave << std::endl;
    // std::cout << "prev scaleR" << lastObsR.octave << std::endl;
    int scale = cvCeil(log(dif)/lastObsKF->logScale);
    if ( scale < 0 )
        scale = 0;
    else if ( scale >= lastObsKF->nScaleLev )
        scale = lastObsKF->nScaleLev - 1;
    // std::cout << "scale" << scale << std::endl;
    return scale;
}

void MapPoint::update(KeyFrame* kF)
{
    lastObsKF = kF;
    Eigen::Vector3d pos = wp3d;
    pos = pos - kF->pose.pose.block<3,1>(0,3);
    const float dist = pos.norm();
    const std::pair<int, int>& idxs = kFMatches[kF];
    int level;
    if ( idxs.first >= 0 )
        level = lastObsL.octave;
    else
        level = lastObsR.octave;

    const float scaleF = kF->scaleFactor[level];
    const int maxLevels = kF->nScaleLev;


    maxScaleDist = dist * scaleF;
    minScaleDist = maxScaleDist * kF->scaleFactor[maxLevels - 1];

    
    // const int level
}

void MapPoint::copyMp(MapPoint* mp, const Zed_Camera* zedPtr)
{
    // Eigen::Vector3d p3d = mp->getWordPose3d();
    // Logging("before pos", wp3d,3);
    setWordPose3d(mp->getWordPose3d());
    // Logging("after pos", wp3d,3);
    desc.push_back(mp->desc.clone());
    // kFWithFIdx = mp->kFWithFIdx;
    std::vector<KeyFrame*> toerase;
    toerase.reserve(kFWithFIdx.size());
    std::unordered_map<KeyFrame*, size_t>::const_iterator it, end(kFWithFIdx.end());
    for ( it = kFWithFIdx.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        size_t keyPos = it->second;
        if ( mp->kFWithFIdx.find(kF) == mp->kFWithFIdx.end() /* || kF->localMapPoints[keyPos] != mp */ )
        {
            kF->eraseMPConnection(keyPos);
            toerase.push_back(kF);
            // eraseKFConnection(kF);
        }

    }

    for (size_t i{0}, end{toerase.size()}; i < end; i++)
    {
            eraseKFConnection(toerase[i]);
    }
    std::unordered_map<KeyFrame*, size_t>::const_iterator itn, endn(mp->kFWithFIdx.end());
    for ( itn = mp->kFWithFIdx.begin(); itn != endn; itn++)
    {
        KeyFrame* kFcand = itn->first;
        size_t keyPos = itn->second;
        kFcand->localMapPoints[keyPos] = mp;
        kFcand->unMatchedF[keyPos] = mp->kdx;
        TrackedKeys& tKeys = kFcand->keys;
        if ( kFWithFIdx.find(kFcand) == kFWithFIdx.end() )
        {
            kFWithFIdx.insert((*itn));
        }
        if ( tKeys.estimatedDepth[keyPos] <= 0 )
            continue;
        Eigen::Vector4d pCam = kFcand->pose.getInvPose() * wp;
        tKeys.estimatedDepth[keyPos] = pCam(2);
        if ( pCam(2) <= zedPtr->mBaseline * 40 )
            tKeys.close[keyPos] = true;
    }
    // kFWithFIdx = mp->kFWithFIdx;
    // inFrame = mp->inFrame;
    // close = mp->close;
    trackCnt = mp->trackCnt;
    // obs = mp->obs;
}

void MapPoint::calcDescriptor()
{
        // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;

    std::unordered_map<KeyFrame*,std::pair<int,int>> observations = kFMatches;

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(std::unordered_map<KeyFrame*,std::pair<int,int>> ::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        std::pair<int,int> indexes = mit -> second;
        int leftIndex = indexes.first, rightIndex = indexes.second;

        if(leftIndex != -1){
            vDescriptors.push_back(pKF->keys.Desc.row(leftIndex));
        }
        if(rightIndex != -1){
            vDescriptors.push_back(pKF->keys.rightDesc.row(rightIndex));
        }
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = FeatureMatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        std::vector<int> vDists(Distances[i],Distances[i]+N);
        std::sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        desc = vDescriptors[BestIdx].clone();
    }
}

void MapPoint::updatePos(const Eigen::Matrix4d& camPoseInv, const Zed_Camera* zedPtr)
{
    // Eigen::Vector4d p = wp;
    // p = camPoseInv * p;

    // const double invZ = 1.0/p(2);
    // const double fx {zedPtr->cameraLeft.fx};
    // const double fy {zedPtr->cameraLeft.fy};
    // const double cx {zedPtr->cameraLeft.cx};
    // const double cy {zedPtr->cameraLeft.cy};

    // const double u {fx*p(0)*invZ + cx};
    // const double v {fy*p(1)*invZ + cy};
    // obs[0].pt = cv::Point2f((float)u, (float)v);
    std::vector<KeyFrame*> toerase;
    toerase.reserve(kFWithFIdx.size());
    std::unordered_map<KeyFrame*, size_t>::iterator it;
    std::unordered_map<KeyFrame*, size_t>::const_iterator end(kFWithFIdx.end());
    for ( it = kFWithFIdx.begin(); it != end; it++)
    {
        KeyFrame* kFcand = it->first;
        const size_t keyPos = it->second;
        TrackedKeys& tKeys = kFcand->keys;
        if ( tKeys.estimatedDepth[keyPos] <= 0 )
            continue;
        Eigen::Vector4d pCam = kFcand->pose.getInvPose() * wp;
        if (pCam(2) <= 0 )
        {
            tKeys.estimatedDepth[keyPos] = -1;
            kFcand->localMapPoints[keyPos] = nullptr;
            tKeys.rightIdxs[keyPos] = -1;
            toerase.emplace_back(kFcand);
            continue;
        }
        tKeys.estimatedDepth[keyPos] = pCam(2);
        if ( pCam(2) <= zedPtr->mBaseline * 40)
        {
            tKeys.close[keyPos] = true;
        }
    }
    if ( !toerase.empty() )
    {
        for ( size_t i{0}, end{toerase.size()}; i < end; i++)
        {
            this->eraseKFConnection(toerase[i]);
        }
    }

}

void MapPoint::eraseKFConnection(KeyFrame* kF)
{
    kFWithFIdx.erase(kF);
}

bool MapPoint::GetInFrame() const
{
    return inFrame;
}

bool MapPoint::GetIsOutlier() const
{
    return isOutlier;
}

bool MapPoint::getActive() const
{
    return isActive;
}

void MapPoint::SetInFrame(bool infr)
{
    inFrame = infr;
}

void MapPoint::SetIsOutlier(bool isOut)
{
    isOutlier = isOut;
}

void MapPoint::setActive(bool act)
{
    isActive = act;
}

Eigen::Vector4d MapPoint::getWordPose4d() const
{
    return wp;
}

Eigen::Vector3d MapPoint::getWordPose3d() const
{
    return wp3d;
}



void MapPoint::setWordPose4d(const Eigen::Vector4d& p)
{
    wp = p;
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
}

void MapPoint::setWordPose3d(const Eigen::Vector3d& p)
{
    wp3d = p;
    wp = Eigen::Vector4d(p(0), p(1), p(2), 1.0);
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
    mp->added = true;
    mapPoints.insert(std::pair<unsigned long, MapPoint*>(pIdx, mp));
    pIdx++;
}

void Map::addMapPoint(MapPoint* mp)
{
    mp->added = true;
    mapPoints.insert(std::pair<unsigned long, MapPoint*>(pIdx, mp));
    pIdx++;
}

void Map::removeKeyFrame(int idx)
{
    keyFrames.erase((unsigned long) idx);
}

void Map::addKeyFrame(Eigen::Matrix4d _pose)
{
    KeyFrame* kF = new KeyFrame(_pose, kIdx);
    keyFrames.insert(std::pair<unsigned long, KeyFrame*>(kIdx, kF));
    kIdx ++;
}

void Map::addKeyFrame(KeyFrame* kF)
{
    keyFrames.insert(std::pair<unsigned long, KeyFrame*>(kIdx, kF));
    kIdx ++;
}

} // namespace vio_slam