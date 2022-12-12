#include "LocalBA.h"

namespace vio_slam
{

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), fm(_fm)
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
    std::vector<KeyFrame*>::const_iterator it, end(map->activeKeyFrames.end());
    for ( it = map->activeKeyFrames.begin(); it != end; it++)
    {
        const int kIdx {(*it)->numb};
        Eigen::Matrix<double,3,4> extr = (*it)->pose.pose.block<3,4>(0,0);
        Eigen::Matrix<double,3,4> extrRight = extr * zedPtr->extrinsics;
        extr = K * extr;
        extrRight = K * extrRight;
        projMatrices.emplace(kIdx, extr);
        if ( kIdx != 0 )
            projMatrices.emplace( - kIdx, extrRight);
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

    std::vector<std::vector<std::pair<int, int>>> matchedIdxs(map->activeKeyFrames.front()->keys.keyPoints.size(),std::vector<std::pair<int, int>>());

    KeyFrame* lastKF = map->activeKeyFrames.front();

    for ( it = map->activeKeyFrames.begin(); it != end; it++)
    {
        if ( *it == map->activeKeyFrames.front() )
            continue;

        fm->matchLocalBA(matchedIdxs, lastKF->keys, (*it)->keys);
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
    std::unordered_map<int, Eigen::Matrix<double,3,4>> projMatrices;
    projMatrices.reserve(map->activeKeyFrames.size());

    calcProjMatrices(projMatrices);
    

    Logging("size", allMapPoints.size(),3);




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
    while ( !map->endOfFrames )
    {
        if ( map->keyFrameAdded )
        {
            map->keyFrameAdded = false;
            computeAllMapPoints();
        }
    }
}



} // namespace vio_slam