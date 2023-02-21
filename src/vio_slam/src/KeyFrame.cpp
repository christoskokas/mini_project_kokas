#include "KeyFrame.h"

namespace vio_slam
{

void KeyFrame::getConnectedKFs(const Map* map, std::vector<KeyFrame*>& activeKF, const int N)
{
    // activeKF.reserve(20);
    // activeKF.emplace_back(this);
    int count {1};
    for ( int32_t i{connections.size() - 2}, end{0}; i >= end; i--)
    {
        if ( connections[i] > 0 )
        {
            activeKF.emplace_back(map->keyFrames.at(i));
            count++;
        }
        if ( count >= N )
            break;
    }
}

void KeyFrame::setBackPose(const Eigen::Matrix4d& _backPose)
{
    backPose = _backPose;
    backPoseInv = backPose.inverse();
}

void KeyFrame::getConnectedKFs(std::vector<KeyFrame*>& activeKF, const int N)
{
    // activeKF.reserve(20);
    // activeKF.emplace_back(this);
    int count {1};
    for ( std::vector<std::pair<int,KeyFrame*>>::const_iterator it = sortedKFWeights.begin(), end = sortedKFWeights.end(); it != end; it++)
    {
        const std::pair<int,KeyFrame*>& conn = *it;
        if ( conn.second != this )
        {
            activeKF.emplace_back(conn.second);
            count++;
        }
        if ( count >= N )
            break;
    }
}

void KeyFrame::calcConnections()
{
    std::unordered_map<KeyFrame*, int> connWeights;
    for (std::vector<MapPoint*>::const_iterator it = localMapPoints.begin(), end = localMapPoints.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatches.begin(), kfend = mp->kFMatches.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            connWeights[kfCand] ++;
        }
    }

    for (std::vector<MapPoint*>::const_iterator it = localMapPointsR.begin(), end = localMapPointsR.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatches.begin(), kfend = mp->kFMatches.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            const std::pair<int,int>& keyPos = kf->second;
            if ( keyPos.first >= 0 || keyPos.second < 0 )
                continue;
            connWeights[kfCand] ++;
        }
    }

    for (std::vector<MapPoint*>::const_iterator it = localMapPointsB.begin(), end = localMapPointsB.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatchesB.begin(), kfend = mp->kFMatchesB.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            connWeights[kfCand] ++;
        }
    }

    for (std::vector<MapPoint*>::const_iterator it = localMapPointsRB.begin(), end = localMapPointsRB.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatchesB.begin(), kfend = mp->kFMatchesB.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            const std::pair<int,int>& keyPos = kf->second;
            if ( keyPos.first >= 0 || keyPos.second < 0 )
                continue;
            connWeights[kfCand] ++;
        }
    }

    const int threshold = 15;
    std::vector<std::pair<int,KeyFrame*>> orderedConn;
    orderedConn.reserve(connWeights.size());
    for (std::unordered_map<KeyFrame*, int>::const_iterator it = connWeights.begin(), end(connWeights.end()); it != end; it ++)
    {
        KeyFrame* kfCand = it->first;
        int weight = it->second;
        if ( weight >= threshold )
            orderedConn.emplace_back(weight, kfCand);
    }
    std::sort(orderedConn.rbegin(), orderedConn.rend());
    sortedKFWeights = orderedConn;
}

void KeyFrame::eraseMPConnection(const std::pair<int,int>& mpPos)
{
    if ( mpPos.first >= 0 )
        eraseMPConnection(mpPos.first);
    if ( mpPos.second >= 0 )
        eraseMPConnectionR(mpPos.second);
}

void KeyFrame::eraseMPConnectionB(const std::pair<int,int>& mpPos)
{
    if ( mpPos.first >= 0 )
        eraseMPConnectionB(mpPos.first);
    if ( mpPos.second >= 0 )
        eraseMPConnectionRB(mpPos.second);
}

void KeyFrame::eraseMPConnection(const int mpPos)
{
    localMapPoints[mpPos] = nullptr;
    unMatchedF[mpPos] = -1;
}

void KeyFrame::eraseMPConnectionB(const int mpPos)
{
    localMapPointsB[mpPos] = nullptr;
    unMatchedFB[mpPos] = -1;
}

void KeyFrame::eraseMPConnectionR(const int mpPos)
{
    localMapPointsR[mpPos] = nullptr;
    unMatchedFR[mpPos] = -1;
}

void KeyFrame::eraseMPConnectionRB(const int mpPos)
{
    localMapPointsRB[mpPos] = nullptr;
    unMatchedFRB[mpPos] = -1;
}

KeyFrame::KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx) : numb(_numb), frameIdx(_frameIdx)
{
    pose.setPose(_pose);
    leftIm = _leftIm.clone();
    rLeftIm = rLIm.clone();
}

KeyFrame::KeyFrame(const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx) : numb(_numb), frameIdx(_frameIdx)
{
    pose.refPose = _refPose;
    pose.setPose(realPose);
    leftIm = _leftIm.clone();
    rLeftIm = rLIm.clone();
}

KeyFrame::KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, Eigen::MatrixXd _homoPoints3D, const int _numb) : numb(_numb), frameIdx(_numb)
{
    points3D = points;
    pose.setPose(poseT);
    homoPoints3D = _homoPoints3D;
}

KeyFrame::KeyFrame(Eigen::Matrix4d _pose, const int _numb) : numb(_numb), frameIdx(_numb)
{
    pose.setPose(_pose);
}

KeyFrame::KeyFrame(Eigen::Matrix4d poseT, std::vector<cv::Point3d> points, const int _numb) : numb(_numb), frameIdx(_numb)
{
    points3D = points;
    pose.setPose(poseT);
    const size_t end {points3D.size()};
    Eigen::MatrixX4d temp(end,4);
    for (size_t i {0}; i<end;i++)
    {
        temp(i,0) = points3D[i].x;
        temp(i,1) = points3D[i].y;
        temp(i,2) = points3D[i].z;
        temp(i,3) = 1;
    }
    homoPoints3D = temp;
}

Eigen::Vector4d KeyFrame::getWorldPosition(int idx)
{
    return pose.pose * homoPoints3D.row(idx).transpose();
}

Eigen::Matrix4d KeyFrame::getPose()
{
    return pose.pose;
}

};