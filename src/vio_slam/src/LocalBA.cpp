#include "LocalBA.h"

namespace vio_slam
{

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), fm(_fm), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

// From ColMap/src/base/triangulation.cc

Eigen::Vector3d LocalMapper::TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices,
    const std::vector<Eigen::Vector2d>& points) {
  CHECK_EQ(proj_matrices.size(), points.size());

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point = points[i].homogeneous().normalized();
    const Eigen::Matrix<double, 3, 4> term =
        proj_matrices[i] - point * point.transpose() * proj_matrices[i];
    A += term.transpose() * term;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

  return eigen_solver.eigenvectors().col(0).hnormalized();
}

void LocalMapper::calcProjMatrices(std::unordered_map<int, Eigen::Matrix<double,3,4>>& projMatrices)
{
    Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrisics;
    const int aKFsize {map->activeKeyFrames.size()};
    std::vector<KeyFrame*>::const_iterator it, end(map->activeKeyFrames.end());
    for ( it = map->activeKeyFrames.begin(); it != end; it++)
    {
        const int kIdx {(*it)->numb};
        Eigen::Matrix<double,3,4> extr = (*it)->pose.pose.block<3,4>(0,0);
        Eigen::Matrix<double,3,4> extrRight = extr * zedPtr->extrinsics;
        extr = K * extr;
        extrRight = K * extrRight;
        if ( kIdx != 0 )
        {
            projMatrices.emplace(kIdx, extr);
            projMatrices.emplace( - kIdx, extrRight);
        }
        else
        {
            projMatrices.emplace(aKFsize, extr);
            projMatrices.emplace( - aKFsize, extrRight);
        }
    }
}

void LocalMapper::drawLBA(const char* com,std::vector<std::vector<std::pair<int, int>>>& matchedIdxs, const KeyFrame* lastKF, const KeyFrame* otherKF)
{
    std::vector<cv::Point2f> last, other;
    const int kIdx {otherKF->numb};
    for (size_t i {0}, end{lastKF->keys.keyPoints.size()}; i < end; i ++)
    {
        for (size_t j{0}, jend{matchedIdxs[i].size()}; j < jend; j ++)
        {
            if (matchedIdxs[i][j].first == kIdx)
            {
                last.emplace_back(lastKF->keys.keyPoints[i].pt);
                other.emplace_back(otherKF->keys.keyPoints[matchedIdxs[i][j].second].pt);
            }
        }
    }
    cv::Mat im = otherKF->rLeftIm.clone();
    for ( size_t i{0}, end{last.size()}; i < end; i ++)
    {
        cv::circle(im, last[i],2,cv::Scalar(0,255,0));
        cv::line(im, last[i], other[i],cv::Scalar(0,0,255));
        cv::circle(im, other[i],2,cv::Scalar(255,0,0));

    }
    cv::imshow(com, im);
    cv::waitKey(1);
}

void LocalMapper::predictKeysPos(TrackedKeys& keys, const Eigen::Matrix4d& curPose, const Eigen::Matrix4d& camPoseInv)
{
    keys.predKeyPoints = keys.keyPoints;
    for ( size_t i {0}, end{keys.keyPoints.size()}; i < end; i ++)
    {
        if ( keys.estimatedDepth[i] <= 0 )
            continue;
        const double zp = (double)keys.estimatedDepth[i];
        const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
        const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
        Eigen::Vector4d p(xp, yp, zp, 1);
        p = curPose * p;
        p = camPoseInv * p;

        if ( p(2) <= 0.0)
        {
            keys.predKeyPoints[i].pt = cv::Point2f(-1,-1);
            continue;
        }

        const double invZ = 1.0f/p(2);


        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        if ( u < 0 )
            u = 0.0;
        if ( v < 0 )
            v = 0.0;
        if ( u >= w )
            u = w - 1.0;
        if ( v >= h )
            v = h - 1.0;

        keys.predKeyPoints[i].pt = cv::Point2f((float)u, (float)v);

    }
}

void LocalMapper::processMatches(std::vector<std::pair<int, int>>& matchesOfPoint, std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    const int aKFSize = map->activeKeyFrames.size();
    std::vector<std::pair<int, int>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        proj_matrices.emplace_back(allProjMatrices.at(it->first));
        int kFIdx = it->first;
        if ( map->activeKeyFrames.back()->numb == 0)
            if ( abs(it->first) == aKFSize )
                kFIdx = 0;

        if ( it->first > 0)
        {
            KeyFrame* kF = map->keyFrames.at(kFIdx);
            Eigen::Vector2d vec2d((double)kF->keys.keyPoints[it->second].pt.x, (double)kF->keys.keyPoints[it->second].pt.y);
            points.emplace_back(vec2d);
        }
        else
        {
            KeyFrame* kF = map->keyFrames.at( - kFIdx);
            Eigen::Vector2d vec2d((double)kF->keys.rightKeyPoints[it->second].pt.x, (double)kF->keys.rightKeyPoints[it->second].pt.y);
            points.emplace_back(vec2d);
        }
    }
}

void LocalMapper::computeAllMapPoints()
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;

    std::vector<KeyFrame*>::const_iterator it, end(map->activeKeyFrames.end());
    for ( it = map->activeKeyFrames.begin(); it != end; it++)
    {
        std::vector<MapPoint*>::const_iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            if ( (*itmp)->GetIsOutlier() )
                continue;

            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        }
    }
    
    Logging("size", allMapPoints.size(),3);
    KeyFrame* lastKF = map->activeKeyFrames.front();

    const int lastKFIdx = lastKF->numb;

    std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());


    const int aKFsize {map->activeKeyFrames.size()};
    bool first = true;
    for ( it = map->activeKeyFrames.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        predictKeysPos(lastKF->keys, lastKF->pose.pose, (*it)->pose.poseInverse);
        fm->matchLocalBA(matchedIdxs, lastKF, (*it), aKFsize, 3, first);
        first = false;
        // drawLBA("lel",matchedIdxs, lastKF,(*it));
    }



    // match only last keyframe.

    // to act as an idx finder
    // you find last keyframe leftIdx of matched feature and go to that place in the vector and add the Kidx and leftIdx
    // std::vector<kIdx, leftIdx>(currentKeys.size());

    // after matching calculate 3d mappoints and then add all the mappoints to the allmappoints while appending to std::unordered_map<KeyFrame*, size_t> kFWithFIdx to be able to get them for the optimization
    // the mappoints that are calculated and are not outliers after optimization should update the unmachedFeatures in the keyframes class


    // if estimated depth > 0 then use projection matrix from right Idx.
    // if currframe = 1 then use only those that have estimated depth ( 3 projection matrices at least )


    // here match all different features of each keyframe create a map of which keyframe to which are matched so the proj matrices can be used to calculate the 3d point



    // the 3d point calc is given to the first keyframe

    // std::vector<Eigen::Matrix<double,3,4>> projMatrices;
    std::unordered_map<int, Eigen::Matrix<double,3,4>> allProjMatrices;
    allProjMatrices.reserve(aKFsize);

    calcProjMatrices(allProjMatrices);

    for ( size_t i{0}, end {lastKF->keys.keyPoints.size()}; i < end; i ++)
    {
        std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() != 2)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatches(matchesOfPoint, allProjMatrices, proj_mat, pointsVec);
        Eigen::Vector3d vec3d = TriangulateMultiViewPoint(proj_mat, pointsVec);
        Logging("calc 3d", vec3d, 3);
        Logging("est depth", lastKF->keys.estimatedDepth[i],3);
        Logging("KF pose",lastKF->pose.pose,3);
    }
    




    // for each active keyframe
    // if feature not matched and estim depth > 0
    // add mappoint

    // afterwards fuse mappoints somehow

    // feature match features with no estim depth and then triangulate them. features with estim depth dont need triangulation because their position will be optimized by ceres.

    // then feature match all features (even those with no estim depth) between all keyframes and triangulate them into mappoints using the provided poses.
    // remove mappoints that their triangulations have big differences
    // predict each mappoint position in each keyframe ( if no estim depth then no prediction )
    // have a link of mappoints and features for each keyframe (the unmatchedF)

    // run through all mappoints one by one and get each keyframe observation (if any) and pass them to ceres
    // have fixed keyframes ( taken from the connections of each keyframe ) if they are not active.
    // check each KF connection to see if there are connections with inactive keyframes (check if connections[idx] = kIdx -> if kIdx->active = false (these are fixed keyframes))
    
}

void LocalMapper::beginLocalMapping()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        if ( map->keyFrameAdded )
        {
            Timer matchingInLBA("matching LBA");
            map->keyFrameAdded = false;
            computeAllMapPoints();
        }
        std::this_thread::sleep_for(20ms);
    }
}



} // namespace vio_slam