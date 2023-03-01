#include "LocalBA.h"

namespace vio_slam
{

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), fm(_fm), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm, Map* _mapB, Zed_Camera* _zedPtrB, FeatureMatcher* _fmB) : map(_map), zedPtr(_zedPtr), fm(_fm), mapB(_mapB), zedPtrB(_zedPtrB), fmB(_fmB), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, Zed_Camera* _zedPtrB, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), zedPtrB(_zedPtrB), fm(_fm), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
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

void LocalMapper::calcProjMatrices(std::unordered_map<int, Eigen::Matrix<double,3,4>>& projMatrices, std::vector<KeyFrame*>& actKeyF)
{
    Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrisics;
    Eigen::Matrix<double,3,3>& Kr = zedPtr->cameraRight.intrisics;
    const int aKFsize {actKeyF.size()};
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = Kr;
    // projR.block<3,3>(0,0) = K;
    projR(0,3) = -fx * zedPtr->extrinsics(0,3);
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        const int kIdx {(*it)->numb};
        Eigen::Matrix<double,4,4> extr2 = (*it)->pose.poseInverse;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        // if ( (*it)->numb != 0 )
        // {
        //     Eigen::Matrix<double,4,4> extrRight = (*it)->pose.poseInverse;
        //     extrRight =  projR * extrRight;
        //     Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        //     projMatrices.emplace( - kIdx, extrR);
        // }
        projMatrices.emplace(kIdx, extr);
        // if ( kIdx != 0 )
        // {
        //     projMatrices.emplace(kIdx, extr);
        //     // projMatrices.emplace( - kIdx, extrRight2);
        // }
        // else
        // {
        //     projMatrices.emplace(aKFsize, extr);
        //     projMatrices.emplace( - aKFsize, extrRight2);
        // }
    }
}

void LocalMapper::calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF)
{
    Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrisics;
    Eigen::Matrix<double,3,3>& Kr = zedPtr->cameraRight.intrisics;
    const int aKFsize {actKeyF.size()};
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = Kr;
    // projR.block<3,3>(0,0) = K;
    // projR(0,3) = -fx * zedPtr->extrinsics(0,3);
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        const int kIdx {(*it)->numb};
        Eigen::Matrix<double,4,4> extr2 = (*it)->pose.poseInverse;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        Eigen::Matrix<double,4,4> extrRight = ((*it)->pose.pose * zedPtr->extrinsics).inverse();
        extrRight =  projR * extrRight;
        Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        projMatrices.emplace(*it, std::make_pair(extr, extrR));
    }
}

void LocalMapper::calcProjMatricesRB(const Zed_Camera* zedCam,std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF,const bool back)
{
    const Eigen::Matrix<double,3,3>& K = zedCam->cameraLeft.intrisics;
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = K;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        const Eigen::Matrix4d& poseCam = (back) ? (*it)->backPose : (*it)->pose.pose;
        const Eigen::Matrix4d& poseCamInv = (back) ? (*it)->backPoseInv : (*it)->pose.poseInverse;
        Eigen::Matrix<double,4,4> extr2 = poseCamInv;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        Eigen::Matrix<double,4,4> extrRight = (poseCam * zedCam->extrinsics).inverse();
        extrRight =  projR * extrRight;
        Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        projMatrices.emplace(*it, std::make_pair(extr, extrR));
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

void LocalMapper::drawPred(KeyFrame* lastKF, std::vector<cv::KeyPoint>& keys,std::vector<cv::KeyPoint>& predKeys)
{
    cv::Mat im = lastKF->rLeftIm.clone();
    for ( size_t i{0}, end{keys.size()}; i < end; i ++)
    {
        if ( predKeys[i].pt.x <= 0 )
            continue;
        cv::circle(im, keys[i].pt,2,cv::Scalar(0,255,0));
        cv::line(im, keys[i].pt, predKeys[i].pt,cv::Scalar(0,0,255));
        cv::circle(im, predKeys[i].pt,2,cv::Scalar(255,0,0));
    }
    cv::imshow("predictions", im);
    cv::waitKey(1);
}

void LocalMapper::drawPred(KeyFrame* lastKF, std::vector<cv::KeyPoint>& keys,std::vector<cv::Point2f>& predKeys)
{
    cv::Mat im = lastKF->rLeftIm.clone();
    for ( size_t i{0}, end{keys.size()}; i < end; i ++)
    {
        if ( predKeys[i].x <= 0 )
            continue;
        cv::circle(im, keys[i].pt,2,cv::Scalar(0,255,0));
        cv::line(im, keys[i].pt, predKeys[i],cv::Scalar(0,0,255));
        cv::circle(im, predKeys[i],2,cv::Scalar(255,0,0));
    }
    cv::imshow("New predictions", im);
    cv::waitKey(1);
}

void LocalMapper::predictKeysPos(TrackedKeys& keys, const Eigen::Matrix4d& curPose, const Eigen::Matrix4d& camPoseInv, std::vector<float>& keysAngles, const std::vector<Eigen::Vector4d>& p4d, std::vector<cv::Point2f>& predPoints)
{
    // cv::KeyPoint::convert(keys.keyPoints, predPoints);
    predPoints.resize(keys.keyPoints.size());
    keysAngles.resize(keys.keyPoints.size(), -5.0);
    for ( size_t i {0}, end{keys.keyPoints.size()}; i < end; i ++)
    {
        predPoints[i] = keys.keyPoints[i].pt;

        Eigen::Vector4d p = camPoseInv * p4d[i];

        if ( p(2) <= 0.0)
        {
            predPoints[i] = cv::Point2f(-1,-1);
            continue;
        }

        const double invZ = 1.0f/p(2);


        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        if ( u < 15 || v < 15 || u >= w - 15 || v >= h - 15 )
        {
            predPoints[i] = cv::Point2f(-1,-1);
            continue;
        }

        // if ( u < 0 )
        //     u = 0.0;
        // if ( v < 0 )
        //     v = 0.0;
        // if ( u >= w )
        //     u = w - 1.0;
        // if ( v >= h )
        //     v = h - 1.0;

        predPoints[i] = cv::Point2f((float)u, (float)v);
        keysAngles[i] = atan2((float)v - keys.keyPoints[i].pt.y, (float)u - keys.keyPoints[i].pt.x);

    }
}

void LocalMapper::processMatches(std::vector<std::pair<int, int>>& matchesOfPoint, std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, std::vector<KeyFrame*>& actKeyF)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    // const int aKFSize = actKeyF.size();
    std::vector<std::pair<int, int>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        const int kFIdx = it->first;
        const int keyIdx = it->second;
        proj_matrices.emplace_back(allProjMatrices.at(kFIdx));

        if ( it->first >= 0)
        {
            const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kFIdx))->keys.keyPoints;
            Eigen::Vector2d vec2d((double)keys[keyIdx].pt.x, (double)keys[keyIdx].pt.y);
            points.emplace_back(vec2d);
        }
        else
        {
            const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kFIdx))->keys.rightKeyPoints;
            Eigen::Vector2d vec2d((double)keys[keyIdx].pt.x, (double)keys[keyIdx].pt.y);
            points.emplace_back(vec2d);
        }
    }
}

void LocalMapper::processMatchesR(std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    // const int aKFSize = actKeyF.size();
    std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        const TrackedKeys& keys = kF->keys;
        const std::pair<int,int>& keyPos = it->second;
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.keyPoints[keyPos.first].pt.x, (double)keys.keyPoints[keyPos.first].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).first);
        }

        if ( keyPos.second >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.rightKeyPoints[keyPos.second].pt.x, (double)keys.rightKeyPoints[keyPos.second].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).second);
        }
    }
}

void LocalMapper::processMatchesRB(std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, const bool back)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    // const int aKFSize = actKeyF.size();
    std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        const TrackedKeys& keys = (back) ? kF->keysB : kF->keys;
        const std::pair<int,int>& keyPos = it->second;
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.keyPoints[keyPos.first].pt.x, (double)keys.keyPoints[keyPos.first].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).first);
        }

        if ( keyPos.second >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.rightKeyPoints[keyPos.second].pt.x, (double)keys.rightKeyPoints[keyPos.second].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).second);
        }
    }
}

void LocalMapper::processMatchesB(Map* map, std::vector<std::pair<int, int>>& matchesOfPoint, std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, std::vector<KeyFrame*>& actKeyF)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    // const int aKFSize = actKeyF.size();
    std::vector<std::pair<int, int>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        const int kFIdx = it->first;
        const int keyIdx = it->second;
        proj_matrices.emplace_back(allProjMatrices.at(kFIdx));

        if ( it->first >= 0)
        {
            const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kFIdx))->keys.keyPoints;
            Eigen::Vector2d vec2d((double)keys[keyIdx].pt.x, (double)keys[keyIdx].pt.y);
            points.emplace_back(vec2d);
        }
        else
        {
            const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kFIdx))->keys.rightKeyPoints;
            Eigen::Vector2d vec2d((double)keys[keyIdx].pt.x, (double)keys[keyIdx].pt.y);
            points.emplace_back(vec2d);
        }
    }
}

void LocalMapper::projectToPlane(Eigen::Vector4d& vec, cv::Point2f& p2f)
{
    const double invZ {1.0/vec(2)};
    const double u {fx*vec(0)*invZ + cx};
    const double v {fy*vec(1)*invZ + cy};
    p2f = cv::Point2f((float)u, (float)v);
}

bool LocalMapper::checkReprojErr(Eigen::Vector4d& calcVec, std::vector<std::pair<int, int>>& matchesOfPoint, const std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices)
{
    if ( calcVec(2) <= 0 )
        return false;
    // Eigen::Vector4d p4d(calcVec(0), calcVec(1), calcVec(2),1.0);
    int count {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        int kfNumb {matchesOfPoint[i].first};
        const KeyFrame* kF = map->keyFrames.at(abs(kfNumb));
        Eigen::Vector3d p3dnew = allProjMatrices.at(kfNumb) * calcVec;
        p3dnew = p3dnew/p3dnew(2);
        // p4dnew = kF->pose.poseInverse * calcVec;
        // projectToPlane(p4dnew,p2f);
        cv::Point2f p2f((float)p3dnew(0), (float)p3dnew(1));
        float err1,err2;
        if (kfNumb >= 0)
        {
            err1 = kF->keys.keyPoints[matchesOfPoint[i].second].pt.x - p2f.x;
            err2 = kF->keys.keyPoints[matchesOfPoint[i].second].pt.y - p2f.y;

        }
        else
        {
            err1 = kF->keys.rightKeyPoints[matchesOfPoint[i].second].pt.x - p2f.x;
            err2 = kF->keys.rightKeyPoints[matchesOfPoint[i].second].pt.y - p2f.y;

        }
        float err = err1*err1 + err2*err2;
        // Logging("err", err,3);
        if ( err < reprjThreshold )
        {
            matchesOfPoint[count++] = matchesOfPoint[i];
        }
    }
    matchesOfPoint.resize(count);
    if ( count > 3 )
        return true;
    else
        return false;
}

bool LocalMapper::checkReprojErrNew(KeyFrame* lastKF, const int keyPos, Eigen::Vector4d& calcVec, std::vector<std::pair<int, int>>& matchesOfPoint, const std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_mat, std::vector<Eigen::Vector2d>& pointsVec)
{
    if ( calcVec(2) <= 0 )
    {
        matchesOfPoint.clear();
        return false;
    }
    // Eigen::Vector4d p4d(calcVec(0), calcVec(1), calcVec(2),1.0);
    int count {0};
    bool correctKF {false};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<int, int>& match = matchesOfPoint[i];
        bool right {false};
        int kfNumb {match.first};
        const int lastKFNumb = lastKF->numb;
        // Logging("1", "",3);
        Eigen::Vector3d p3dnew = allProjMatrices.at(kfNumb) * calcVec;
        // Logging("2", "",3);
        p3dnew = p3dnew/p3dnew(2);
        // p4dnew = kF->pose.poseInverse * calcVec;
        // projectToPlane(p4dnew,p2f);
        cv::Point2f p2f((float)p3dnew(0), (float)p3dnew(1));
        float err1,err2;
        // Logging("3", "",3);
        if (kfNumb >= 0)
        {
        
            std::vector<cv::KeyPoint> keys;
            bool out {false};
            try {
                keys = map->keyFrames.at(abs(kfNumb))->keys.keyPoints;
            } 
            catch(std::out_of_range& e)
            {
                out = true;
                Logging("OUT OF RANGEEEEEEEEEEEEEEEEEEEEEE", "",3);
            }
            if ( out )
            {
                matchesOfPoint.clear();
                return false;
            }


            // const std::vector<cv::KeyPoint>& keys = map->keyFrames.at((unsigned long)abs(kfNumb))->keys.keyPoints;
            err1 = keys[match.second].pt.x - p2f.x;
            err2 = keys[match.second].pt.y - p2f.y;

        }
        else
        {
            right = true;
            std::vector<cv::KeyPoint> keys;
            bool out {false};
            try {
                keys = map->keyFrames.at(abs(kfNumb))->keys.rightKeyPoints;
            } 
            catch(std::out_of_range& e)
            {
                out = true;
                Logging("OUT OF RANGEEEEEEEEEEEEEEEEEEEEEE", "",3);
            }
            if ( out )
            {
                matchesOfPoint.clear();
                return false;
            }
            // const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kfNumb))->keys.rightKeyPoints;
            err1 = keys[match.second].pt.x - p2f.x;
            err2 = keys[match.second].pt.y - p2f.y;

        }
        float err = err1*err1 + err2*err2;
        // Logging("err", err,3);
        // Logging("4", "",3);
        if ( err < reprjThreshold )
        {
            matchesOfPoint[count] = match;
            proj_mat[count] = proj_mat[i];
            pointsVec[count] = pointsVec[i];
            count ++;
            if ( kfNumb == lastKFNumb )
                correctKF = true;
        }
        // else
        // {
        //     // remove connections here
        // }
    }
    matchesOfPoint.resize(count);
    if ( count > (minCount - 1)  && correctKF )
        return true;
    else
    {
        // here make mp outlier
        return false;
    }
}

bool LocalMapper::checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec)
{
    // Eigen::Vector4d p4d(calcVec(0), calcVec(1), calcVec(2),1.0);
    int count {0};
    bool correctKF {false};
    const int lastKFNumb = lastKF->numb;
    int projCount {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<KeyFrame *, std::pair<int, int>>& match = matchesOfPoint[i];
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const TrackedKeys& keys = kFCand->keys;
        
        std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        const int kFCandNumb {kFCand->numb};
        bool cor {false};
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold )
            {
                keyPos.first = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( keyPos.second >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold )
            {
                keyPos.second = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( cor )
            count++;
    }
    matchesOfPoint.resize(count);
    if ( count >= minCount  && correctKF )
        return true;
    else
        return false;
}

bool LocalMapper::checkReprojErrNewRB(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec, const bool back)
{
    // Eigen::Vector4d p4d(calcVec(0), calcVec(1), calcVec(2),1.0);
    int count {0};
    bool correctKF {false};
    const int lastKFNumb = lastKF->numb;
    int projCount {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<KeyFrame *, std::pair<int, int>>& match = matchesOfPoint[i];
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const TrackedKeys& keys = (back) ? kFCand->keysB : kFCand->keys;
        
        std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        const int kFCandNumb {kFCand->numb};
        bool cor {false};
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.keyPoints[keyPos.first].octave;
            const double weight = (double)kFCand->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.first = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( keyPos.second >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.rightKeyPoints[keyPos.second].octave;
            const double weight = lastKF->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.second = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( cor )
            count++;
    }
    matchesOfPoint.resize(count);
    if ( count >= minCount  && correctKF )
        return true;
    else
        return false;
}

bool LocalMapper::checkReprojErrNewB(Map* map, KeyFrame* lastKF, const int keyPos, Eigen::Vector4d& calcVec, std::vector<std::pair<int, int>>& matchesOfPoint, const std::unordered_map<int, Eigen::Matrix<double,3,4>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_mat, std::vector<Eigen::Vector2d>& pointsVec)
{
    if ( calcVec(2) <= 0 )
    {
        matchesOfPoint.clear();
        return false;
    }
    // Eigen::Vector4d p4d(calcVec(0), calcVec(1), calcVec(2),1.0);
    int count {0};
    bool correctKF {false};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<int, int>& match = matchesOfPoint[i];
        bool right {false};
        int kfNumb {match.first};
        const int lastKFNumb = lastKF->numb;
        // Logging("1", "",3);
        Eigen::Vector3d p3dnew = allProjMatrices.at(kfNumb) * calcVec;
        // Logging("2", "",3);
        p3dnew = p3dnew/p3dnew(2);
        // p4dnew = kF->pose.poseInverse * calcVec;
        // projectToPlane(p4dnew,p2f);
        cv::Point2f p2f((float)p3dnew(0), (float)p3dnew(1));
        float err1,err2;
        // Logging("3", "",3);
        if (kfNumb >= 0)
        {
        
            std::vector<cv::KeyPoint> keys;
            bool out {false};
            try {
                keys = map->keyFrames.at(abs(kfNumb))->keys.keyPoints;
            } 
            catch(std::out_of_range& e)
            {
                out = true;
                Logging("OUT OF RANGEEEEEEEEEEEEEEEEEEEEEE", "",3);
            }
            if ( out )
            {
                matchesOfPoint.clear();
                return false;
            }


            // const std::vector<cv::KeyPoint>& keys = map->keyFrames.at((unsigned long)abs(kfNumb))->keys.keyPoints;
            err1 = keys[match.second].pt.x - p2f.x;
            err2 = keys[match.second].pt.y - p2f.y;

        }
        else
        {
            right = true;
            std::vector<cv::KeyPoint> keys;
            bool out {false};
            try {
                keys = map->keyFrames.at(abs(kfNumb))->keys.rightKeyPoints;
            } 
            catch(std::out_of_range& e)
            {
                out = true;
                Logging("OUT OF RANGEEEEEEEEEEEEEEEEEEEEEE", "",3);
            }
            if ( out )
            {
                matchesOfPoint.clear();
                return false;
            }
            // const std::vector<cv::KeyPoint>& keys = map->keyFrames.at(abs(kfNumb))->keys.rightKeyPoints;
            err1 = keys[match.second].pt.x - p2f.x;
            err2 = keys[match.second].pt.y - p2f.y;

        }
        float err = err1*err1 + err2*err2;
        // Logging("err", err,3);
        // Logging("4", "",3);
        if ( err < reprjThreshold )
        {
            matchesOfPoint[count] = match;
            proj_mat[count] = proj_mat[i];
            pointsVec[count] = pointsVec[i];
            count ++;
            if ( kfNumb == lastKFNumb )
                correctKF = true;
        }
        // else
        // {
        //     // remove connections here
        // }
    }
    matchesOfPoint.resize(count);
    if ( count > (minCount - 1)  && correctKF )
        return true;
    else
    {
        // here make mp outlier
        return false;
    }
}

void LocalMapper::addMultiViewMapPoints(const Eigen::Vector4d& posW, const std::vector<std::pair<int, int>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& keyPos)
{
    const TrackedKeys& temp = lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    bool toAdd {true};
    const int lastKFNumb {lastKF->numb};
    // if ( lastKF->unMatchedF[keyPos] >= 0 )
    // {
    //     mpnew = lastKF->localMapPoints[keyPos];
    //     toAdd = false;
    // }
    // else
    MapPoint* mp = new MapPoint(posW, temp.Desc.row(keyPos),temp.keyPoints[keyPos], temp.close[keyPos], lastKF->numb, mpIdx);
    if ( lastKF->keys.estimatedDepth[keyPos] > 0)
        mp->desc.push_back(lastKF->keys.rightDesc.row(lastKF->keys.rightIdxs[keyPos]));

    // MapPoint* mp;
    // *mp = *mpnew;
    mpIdx++;
    int count {0};
    // std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        if ( matchesOfPoint[i].first >= 0)
        {
            KeyFrame* kf = map->keyFrames.at(matchesOfPoint[i].first);
            const TrackedKeys& keys = kf->keys;
            const size_t keyPosCand = matchesOfPoint[i].second;
            mp->kFWithFIdx.insert(std::pair<KeyFrame*, size_t>(kf, matchesOfPoint[i].second));
            count ++;
            if ( kf->numb != lastKFNumb)
            {
                mp->desc.push_back(kf->keys.Desc.row(matchesOfPoint[i].second));
                if ( keys.estimatedDepth[keyPosCand] > 0 )
                    mp->desc.push_back(keys.rightDesc.row(keys.rightIdxs[keyPosCand]));
            }
            // kf->localMapPoints[matchesOfPoint[i].second] = &mp;
            // kf->unMatchedF[matchesOfPoint[i].second] = mp.kdx;
        }
    }
    mp->trackCnt = count;
    pointsToAdd[keyPos] = mp;
}

void LocalMapper::addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos)
{
    const TrackedKeys& temp = lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    bool toAdd {true};
    const int lastKFNumb {lastKF->numb};
    MapPoint* mp = nullptr;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( kFCand->numb == lastKFNumb )
        {
            if ( keyPos.first >= 0 )
            {
                mp = new MapPoint(posW, temp.Desc.row(keyPos.first),temp.keyPoints[keyPos.first], temp.close[keyPos.first], lastKF->numb, mpIdx);
            }
            else if ( keyPos.second >= 0 )
            {
                mp = new MapPoint(posW, temp.rightDesc.row(keyPos.second),temp.rightKeyPoints[keyPos.second], temp.close[keyPos.second], lastKF->numb, mpIdx);
            }
            break;
        }
    }
    if ( !mp )
        return;

    mpIdx++;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
    }
    mp->update(lastKF);
    pointsToAdd[mpPos] = mp;
}

void LocalMapper::addMultiViewMapPointsRB(const Eigen::Vector4d& posW, const std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos, const bool back)
{
    const TrackedKeys& temp = (back) ? lastKF->keysB : lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    bool toAdd {true};
    const int lastKFNumb {lastKF->numb};
    MapPoint* mp = nullptr;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( kFCand->numb == lastKFNumb )
        {
            if ( keyPos.first >= 0 )
            {
                mp = new MapPoint(posW, temp.Desc.row(keyPos.first),temp.keyPoints[keyPos.first], temp.close[keyPos.first], lastKF->numb, mpIdx);
            }
            else if ( keyPos.second >= 0 )
            {
                mp = new MapPoint(posW, temp.rightDesc.row(keyPos.second),temp.rightKeyPoints[keyPos.second], false, lastKF->numb, mpIdx);
            }
            break;
        }
    }
    if ( !mp )
        return;

    mpIdx++;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( back )
            mp->kFMatchesB.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
        else
            mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));

    }
    mp->update(lastKF, back);
    pointsToAdd[mpPos] = mp;
}

void LocalMapper::addMultiViewMapPointsB(Map* map, const Eigen::Vector4d& posW, const std::vector<std::pair<int, int>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& keyPos)
{
    const TrackedKeys& temp = lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    bool toAdd {true};
    const int lastKFNumb {lastKF->numb};
    // if ( lastKF->unMatchedF[keyPos] >= 0 )
    // {
    //     mpnew = lastKF->localMapPoints[keyPos];
    //     toAdd = false;
    // }
    // else
    MapPoint* mp = new MapPoint(posW, temp.Desc.row(keyPos),temp.keyPoints[keyPos], temp.close[keyPos], lastKF->numb, mpIdx);
    if ( lastKF->keys.estimatedDepth[keyPos] > 0)
        mp->desc.push_back(lastKF->keys.rightDesc.row(lastKF->keys.rightIdxs[keyPos]));

    // MapPoint* mp;
    // *mp = *mpnew;
    mpIdx++;
    int count {0};
    // std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        if ( matchesOfPoint[i].first >= 0)
        {
            KeyFrame* kf = map->keyFrames.at(matchesOfPoint[i].first);
            const TrackedKeys& keys = kf->keys;
            const size_t keyPosCand = matchesOfPoint[i].second;
            mp->kFWithFIdx.insert(std::pair<KeyFrame*, size_t>(kf, matchesOfPoint[i].second));
            count ++;
            if ( kf->numb != lastKFNumb)
            {
                mp->desc.push_back(kf->keys.Desc.row(matchesOfPoint[i].second));
                if ( keys.estimatedDepth[keyPosCand] > 0 )
                    mp->desc.push_back(keys.rightDesc.row(keys.rightIdxs[keyPosCand]));
            }
            // kf->localMapPoints[matchesOfPoint[i].second] = &mp;
            // kf->unMatchedF[matchesOfPoint[i].second] = mp.kdx;
        }
    }
    mp->trackCnt = count;
    pointsToAdd[keyPos] = mp;
}

void LocalMapper::calcp4d(KeyFrame* lastKF, std::vector<Eigen::Vector4d>& p4d)
{
    const TrackedKeys& keys = lastKF->keys;
    const Eigen::Matrix4d& curPose = lastKF->pose.pose;
    p4d.reserve(keys.keyPoints.size());
    for ( size_t i {0}, end{keys.keyPoints.size()}; i < end; i ++)
    {
        double zp;
        if ( keys.estimatedDepth[i] <= 0 )
            zp = 200.0;
        else
            zp = (double)keys.estimatedDepth[i];
        const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
        const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
        Eigen::Vector4d p(xp, yp, zp, 1);
        p = curPose * p;
        p4d.emplace_back(p);
    }
}

void LocalMapper::triangulateCeres(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose)
{
    // Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix4d& camPose = lastKFPose;
    ceres::Problem problem;
    // ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    // Logging("before", p3d,3);
    ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(7.815f));
    // Logging("p3d", p3d,3);
    for (size_t i {0}, end{obs.size()}; i < end; i ++)
    {
        ceres::CostFunction* costf = MultiViewTriang::Create(camPose, proj_matrices[i], obs[i]);
        problem.AddResidualBlock(costf, loss_function /* squared loss */,p3d.data());

        // problem.SetManifold(frame_qcw.coeffs().data(),quaternion_local_parameterization);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.use_explicit_schur_complement = true;
    options.max_num_iterations = 20;
    // options.minimizer_progress_to_stdout = false;
    problem.SetParameterLowerBound(p3d.data(), 2, 0.1);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector4d p4d(p3d(0), p3d(1), p3d(2), 1.0);
    p4d = lastKFPose * p4d;
    // Logging("p4d", p4d, 3);
    // Logging("after", p3d,3);
    p3d(0) = p4d(0);
    p3d(1) = p4d(1);
    p3d(2) = p4d(2);
    // Logging("sum",summary.BriefReport(),3);
    // Logging("after", p3d,3);

}

void LocalMapper::triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first)
{
    // Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix4d& camPose = lastKFPose;
    ceres::Problem problem;
    // ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    // Logging("before", p3d,3);
    ceres::LossFunction* loss_function = nullptr;
    if ( first )
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    // Logging("p3d", p3d,3);
    for (size_t i {0}, end{obs.size()}; i < end; i ++)
    {
        ceres::CostFunction* costf = MultiViewTriang::Create(camPose, proj_matrices[i], obs[i]);
        problem.AddResidualBlock(costf, loss_function /* squared loss */,p3d.data());

        // problem.SetManifold(frame_qcw.coeffs().data(),quaternion_local_parameterization);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.use_explicit_schur_complement = true;
    options.max_num_iterations = 20;
    // options.minimizer_progress_to_stdout = false;
    problem.SetParameterLowerBound(p3d.data(), 2, 0.1);
    // problem.SetParameterUpperBound(p3d.data(), 2, 500.0);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector4d p4d(p3d(0), p3d(1), p3d(2), 1.0);
    p4d = lastKFPose * p4d;
    // Logging("p4d", p4d, 3);
    // Logging("after", p3d,3);
    p3d(0) = p4d(0);
    p3d(1) = p4d(1);
    p3d(2) = p4d(2);
    // Logging("sum",summary.BriefReport(),3);
    // Logging("after", p3d,3);

}

void LocalMapper::addToMap(KeyFrame* lastKF, const std::vector<MapPoint*>& pointsToAdd)
{
    int newMapPointsCount {0};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        if (!pointsToAdd[i] )
        {
            if ( lastKF->localMapPoints[i] )
            {
                if ( !lastKF->localMapPoints[i]->GetInFrame() )
                    lastKF->localMapPoints[i]->SetIsOutlier(true);
                // lastKF->localMapPoints[i] = nullptr;
                lastKF->unMatchedF[i] = -1;
                lastKF->keys.estimatedDepth[i] = -1;
                lastKF->keys.close[i] = false;
                lastKF->keys.rightIdxs[i] = -1;
                // here remove all localmappoints from keyframes for the current mappoint
            }
            continue;
        }
        // Logging("added MP","",3);
        // std::unordered_map<vio_slam::KeyFrame *, size_t>::iterator it, Mapend(pointsToAdd[i]->kFWithFIdx.end());
        // for (it = pointsToAdd[i]->kFWithFIdx.begin(); it != Mapend; it++)
        // {
        //     it->first->unMatchedF[it->second] = false;
        // }
        if ( !lastKF->localMapPoints[i] )
        {
            map->activeMapPoints.emplace_back(pointsToAdd[i]);
            map->addMapPoint(pointsToAdd[i]);
            newMapPointsCount ++;
        }
        lastKF->localMapPoints[i]->copyMp(pointsToAdd[i], zedPtr);
        
        // lastKF->localMapPoints[i] = pointsToAdd[i];
        lastKF->unMatchedF[i] = pointsToAdd[i]->kdx;
    }
    Logging("Success!", newMapPointsCount, 3);

}

void LocalMapper::addToMapRemoveCon(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<int, int>>>& matchedIdxs)
{
    int newMapPointsCount {0};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if (!pointsToAdd[i] )
        {
            if ( mp )
            {
                if ( !mp->GetInFrame() && mp->kFWithFIdx.size() < minCount )
                    mp->SetIsOutlier(true);
                // lastKF->eraseMPConnection(i);
                // std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
                // removeCon(mp, matchesOfPoint, lastKF->numb);
                // lastKF->localMapPoints[i] = nullptr;
                // lastKF->unMatchedF[i] = -1;
                // lastKF->keys.estimatedDepth[i] = -1;
                // lastKF->keys.close[i] = false;
                // lastKF->keys.rightIdxs[i] = -1;
                // here remove all localmappoints from keyframes for the current mappoint
            }
            continue;
        }
        // Logging("added MP","",3);
        // std::unordered_map<vio_slam::KeyFrame *, size_t>::iterator it, Mapend(pointsToAdd[i]->kFWithFIdx.end());
        // for (it = pointsToAdd[i]->kFWithFIdx.begin(); it != Mapend; it++)
        // {
        //     it->first->unMatchedF[it->second] = false;
        // }
        if ( !mp )
        {
            mp = pointsToAdd[i];
            std::unordered_map<KeyFrame*, size_t>::const_iterator itn, endn(mp->kFWithFIdx.end());
            for ( itn = mp->kFWithFIdx.begin(); itn != endn; itn++)
            {
                KeyFrame* kFcand = itn->first;
                size_t keyPos = itn->second;
                kFcand->localMapPoints[keyPos] = mp;
                kFcand->unMatchedF[keyPos] = mp->kdx;
                TrackedKeys& tKeys = kFcand->keys;
                if ( tKeys.estimatedDepth[keyPos] <= 0 )
                    continue;
                Eigen::Vector4d pCam = kFcand->pose.getInvPose() * mp->getWordPose4d();
                tKeys.estimatedDepth[keyPos] = pCam(2);
                if ( pCam(2) <= zedPtr->mBaseline * 40 )
                    tKeys.close[keyPos] = true;
                // if ( kf->numb == lastKF->numb - 3 )
                // {
                    // const Eigen::Matrix4d camPose = kf->pose.getInvPose();
                    // Eigen::Vector4d pointCam = camPose * mp->getWordPose4d();
                    // // if ( pointCam(2) <= 0)
                    // //     continue;
                    // double u {fx*pointCam(0)/pointCam(2) + cx};
                    // double v {fy*pointCam(1)/pointCam(2) + cy};
                    // if ( pointCam(2) <= 0 )
                    // {
                    //     Logging("key", kf->keys.keyPoints[keyPos].pt,3);
                    //     Logging("keyMP", cv::Point2f((float)u, (float)v),3);
                    //     Logging("pointCam", pointCam,3);
                    // }
                // }
            }
        }
        else
            mp->copyMp(pointsToAdd[i], zedPtr);
        // Logging("position in world", mp->getWordPose3d(),3);
        if ( !mp->added )
        {
            map->activeMapPoints.emplace_back(mp);
            map->addMapPoint(mp);
            mp->added = true;
            newMapPointsCount ++;
        }
        mp->lastObsKF = lastKF;
        mp->lastObsL = lastKF->keys.keyPoints[i];
        if ( !mp->GetInFrame() && mp->kFWithFIdx.size() < 3 )
            mp->SetIsOutlier(true);
        // lastKF->localMapPoints[i] = pointsToAdd[i];
        // lastKF->unMatchedF[i] = pointsToAdd[i]->kdx;
    }
    Logging("Success!", newMapPointsCount, 3);

}

void LocalMapper::addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs)
{
    int newMapPointsCount {0};
    const int lastKFNumb {lastKF->numb};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* newMp = pointsToAdd[i];
        if (!newMp )
            continue;
        std::unordered_map<vio_slam::KeyFrame *, std::pair<int, int>>::iterator it, endMp(newMp->kFMatches.end());
        for ( it = newMp->kFMatches.begin(); it != endMp; it++)
        {
            KeyFrame* kFCand = it->first;
            std::pair<int,int>& keyPos = it->second;
            newMp->addConnection(kFCand, keyPos);
        }
        map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
        newMapPointsCount ++;
    }

}

void LocalMapper::addNewMapPointsB(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const bool back)
{
    int newMapPointsCount {0};
    const int lastKFNumb {lastKF->numb};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* newMp = pointsToAdd[i];
        if (!newMp )
            continue;
        std::unordered_map<vio_slam::KeyFrame *, std::pair<int, int>>::iterator it = (back) ? newMp->kFMatchesB.begin() : newMp->kFMatches.begin();
        std::unordered_map<vio_slam::KeyFrame *, std::pair<int, int>>::iterator endMp = (back) ? newMp->kFMatchesB.end() : newMp->kFMatches.end();
        for (; it != endMp; it++)
        {
            KeyFrame* kFCand = it->first;
            std::pair<int,int>& keyPos = it->second;
            if ( back )
                newMp->addConnectionB(kFCand, keyPos);
            else
                newMp->addConnection(kFCand, keyPos);

        }
        if ( back )
            map->activeMapPointsB.emplace_back(newMp);
        else
            map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
        newMapPointsCount ++;
    }
    std::cout << "newMapPointsCount : " << newMapPointsCount << std::endl;
}

void LocalMapper::addToMapRemoveConB(Map* map, KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<int, int>>>& matchedIdxs)
{
    int newMapPointsCount {0};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if (!pointsToAdd[i] )
        {
            if ( mp )
            {
                if ( !mp->GetInFrame() && mp->kFWithFIdx.size() < minCount )
                {
                    // mp->desc.release();
                    mp->SetIsOutlier(true);
                }
                // lastKF->eraseMPConnection(i);
                // std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
                // removeCon(mp, matchesOfPoint, lastKF->numb);
                // lastKF->localMapPoints[i] = nullptr;
                // lastKF->unMatchedF[i] = -1;
                // lastKF->keys.estimatedDepth[i] = -1;
                // lastKF->keys.close[i] = false;
                // lastKF->keys.rightIdxs[i] = -1;
                // here remove all localmappoints from keyframes for the current mappoint
            }
            continue;
        }
        // Logging("added MP","",3);
        // std::unordered_map<vio_slam::KeyFrame *, size_t>::iterator it, Mapend(pointsToAdd[i]->kFWithFIdx.end());
        // for (it = pointsToAdd[i]->kFWithFIdx.begin(); it != Mapend; it++)
        // {
        //     it->first->unMatchedF[it->second] = false;
        // }
        if ( !mp )
        {
            mp = pointsToAdd[i];
            std::unordered_map<KeyFrame*, size_t>::const_iterator itn, endn(mp->kFWithFIdx.end());
            for ( itn = mp->kFWithFIdx.begin(); itn != endn; itn++)
            {
                KeyFrame* kFcand = itn->first;
                size_t keyPos = itn->second;
                kFcand->localMapPoints[keyPos] = mp;
                kFcand->unMatchedF[keyPos] = mp->kdx;
                TrackedKeys& tKeys = kFcand->keys;
                if ( tKeys.estimatedDepth[keyPos] <= 0 )
                    continue;
                Eigen::Vector4d pCam = kFcand->pose.getInvPose() * mp->getWordPose4d();
                tKeys.estimatedDepth[keyPos] = pCam(2);
                if ( pCam(2) <= zedPtr->mBaseline * 40 )
                    tKeys.close[keyPos] = true;
                // if ( kf->numb == lastKF->numb - 3 )
                // {
                    // const Eigen::Matrix4d camPose = kf->pose.getInvPose();
                    // Eigen::Vector4d pointCam = camPose * mp->getWordPose4d();
                    // // if ( pointCam(2) <= 0)
                    // //     continue;
                    // double u {fx*pointCam(0)/pointCam(2) + cx};
                    // double v {fy*pointCam(1)/pointCam(2) + cy};
                    // if ( pointCam(2) <= 0 )
                    // {
                    //     Logging("key", kf->keys.keyPoints[keyPos].pt,3);
                    //     Logging("keyMP", cv::Point2f((float)u, (float)v),3);
                    //     Logging("pointCam", pointCam,3);
                    // }
                // }
            }
        }
        else
            mp->copyMp(pointsToAdd[i], zedPtr);
        // Logging("position in world", mp->getWordPose3d(),3);
        if ( !mp->added )
        {
            map->activeMapPoints.emplace_back(mp);
            map->addMapPoint(mp);
            mp->added = true;
            newMapPointsCount ++;
        }
        mp->lastObsKF = lastKF;
        mp->lastObsL = lastKF->keys.keyPoints[i];
        if ( !mp->GetInFrame() && mp->kFWithFIdx.size() < 3 )
        {
            // mp->desc.release();
            mp->SetIsOutlier(true);
        }
        // lastKF->localMapPoints[i] = pointsToAdd[i];
        // lastKF->unMatchedF[i] = pointsToAdd[i]->kdx;
    }
    Logging("Success!", newMapPointsCount, 3);

}

void LocalMapper::removeCon(MapPoint* mp, std::vector<std::pair<int, int>>& matchesOfPoint, const int lastKFNumb)
{
    std::unordered_map<KeyFrame*, size_t> temp;
    temp.reserve(matchesOfPoint.size());
    std::unordered_map<KeyFrame*, size_t>::const_iterator it, end(mp->kFWithFIdx.end());
    for ( it = mp->kFWithFIdx.begin(); it != end; it++)
    {
        if ( (*it).first->numb <= (lastKFNumb - 10))
        {
            temp.insert((*it));
        }
    }
    if ( matchesOfPoint.empty() )
    {
        // mp->SetIsOutlier(true);
        // mp->kFWithFIdx.clear();
        mp->kFWithFIdx = temp;
        // mp = nullptr;
        return;
    }
    for ( auto& match : matchesOfPoint )
    {
        const int kfNumb = match.first;
        KeyFrame* kf = map->keyFrames.at(abs(kfNumb));
        if ( kfNumb >= 0 )
        {
            temp[kf] = (size_t)match.second;
        }
    }
    mp->kFWithFIdx = temp;
}

void LocalMapper::computeAllMapPoints(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    // Logging("size", allMapPoints.size(),3);
    KeyFrame* lastKF = actKeyF.front();

    const int lastKFIdx = lastKF->numb;

    std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());

    std::vector<Eigen::Vector4d> p4d;
    calcp4d(lastKF, p4d);
    const int aKFsize {actKeyF.size()};
    bool first = true;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        if ( (*it)->numb < lastKFIdx - 5)
            continue;
        std::vector<float> keysAngles;
        std::vector<cv::Point2f> predPoints;
        predictKeysPos(lastKF->keys, lastKF->pose.pose, (*it)->pose.poseInverse, keysAngles, p4d, predPoints);
        // drawPred(lastKF, lastKF->keys.keyPoints, predPoints);
        fm->matchLocalBA(matchedIdxs, lastKF, (*it), aKFsize, 6, first, keysAngles, predPoints);
        // if (first)
        // drawLBA("LBA matches",matchedIdxs, lastKF,(*it));
        first = false;
        
        // drawPred(lastKF, lastKF->keys.keyPoints, lastKF->keys.predKeyPoints);
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

    calcProjMatrices(allProjMatrices, actKeyF);
    int newMapPointsCount {0};
    std::vector<MapPoint*> pointsToAdd;
    pointsToAdd.resize(lastKF->keys.keyPoints.size(),nullptr);
    lastKF->localMapPoints.reserve(lastKF->keys.keyPoints.size());
    // Logging("activeSizeLOCAL", map->activeMapPoints.size(),3);
    for ( size_t i{0}, end {lastKF->keys.keyPoints.size()}; i < end; i ++)
    {
        std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() < 4)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatches(matchesOfPoint, allProjMatrices, proj_mat, pointsVec, actKeyF);
        double zp;
        if (lastKF->keys.estimatedDepth[i] > 0)
            zp = (double)lastKF->keys.estimatedDepth[i];
        else
            zp = 200.0;
        const double xp = (double)(((double)lastKF->keys.keyPoints[i].pt.x-cx)*zp/fx);
        const double yp = (double)(((double)lastKF->keys.keyPoints[i].pt.y-cy)*zp/fy);
        Eigen::Vector4d vecCalc(xp, yp, zp, 1);
        // vecCalc = lastKF->pose.pose * vecCalc;
        
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeres(vec3d, proj_mat, pointsVec, lastKF->pose.pose);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        // Eigen::Vector4d vecCalcCheckD = lastKF->pose.poseInverse * vecCalc;
        // if ( vecCalcCheckD(2) <= 0)
        //     continue;
        // Eigen::Vector3d vec3(vecCalc(0), vecCalc(1), vecCalc(2));
        // Eigen::Vector3d vec3d = TriangulateMultiViewPoint(proj_mat, pointsVec);
        // Eigen::Vector4d temp(vec3d(0),vec3d(1),vec3d(2),1);
        if ( !checkReprojErr(vecCalc, matchesOfPoint, allProjMatrices) )
            continue;
        // else
        //     Logging("Success!", "", 3);

        addMultiViewMapPoints(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        
        newMapPointsCount ++;
        // Logging("calc 3d", vecCalcbef, 3);
        // Logging("calc 3d", vecCalc, 3);


        // Logging("est depth", lastKF->keys.estimatedDepth[i],3);
        // if ( lastKF->keys.estimatedDepth[i]> 0)
        // {
        //     const double zp = (double)lastKF->keys.estimatedDepth[i];
        //     const double xp = (double)(((double)lastKF->keys.keyPoints[i].pt.x-cx)*zp/fx);
        //     const double yp = (double)(((double)lastKF->keys.keyPoints[i].pt.y-cy)*zp/fy);
        //     Eigen::Vector4d vecCalc(xp, yp, zp, 1);
        //     vecCalc = lastKF->pose.pose * vecCalc;
        //     Eigen::Vector3d vec3(vecCalc(0), vecCalc(1), vecCalc(2));
        //     Logging("vecCalc", vec3, 3);
        // }
        // Logging("KF pose",lastKF->pose.pose,3);
    }
    addToMap(lastKF, pointsToAdd);
    Logging("Success!", newMapPointsCount, 3);
    
    // std::unordered_map<MapPoint*, std::pair<Eigen::Vector3d,std::pair<int,std::pair<Eigen::Matrix<double,7,1>
    // Mappoints / all KeyframeIdx check if it is keyframe or frame, pose, observation on keyframe // if keyframe is fixed

    // kfidx can be negative if it is fixed
    // map<mappoint*,vector<pair<vec3d,pair<kfidx,pair<pose, obs>>>>

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

void LocalMapper::calcAllMpsOfKF(std::vector<std::vector<std::pair<int, int>>>& matchedIdxs, KeyFrame* lastKF, std::vector<vio_slam::KeyFrame *>& actKeyF, const int kFsize, std::vector<Eigen::Vector4d>& p4d)
{
    const int keysSize {lastKF->keys.keyPoints.size()};
    const int kfNumbDif {lastKF->numb - kFsize};
    p4d.reserve(keysSize);
    std::vector<bool> addedKF(actKeyF.front()->numb + 1, false);
    for (size_t i{0}, end{actKeyF.size()}; i < end; i++)
    {
        addedKF[actKeyF[i]->numb] = true;
    }
    for ( size_t i{0}; i < keysSize; i++)
    {
        matchedIdxs[i].reserve(300);
        MapPoint* mp = lastKF->localMapPoints[i];
        if ( !mp )
        {
            double zp;
            if ( lastKF->keys.estimatedDepth[i] > 0 )
                zp = (double)lastKF->keys.estimatedDepth[i];
            else
                zp = 1000.0;
            const double xp = (double)(((double)lastKF->keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)lastKF->keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = lastKF->pose.pose * p4dcam;
            p4d.emplace_back(p4dcam);
            if ( lastKF->unMatchedF[i] < 0 )
                matchedIdxs[i].emplace_back(lastKF->numb,(int)i);
            continue;
        }
        p4d.emplace_back(mp->getWordPose4d());
        if ( lastKF->unMatchedF[i] >= 0 )
            continue;
        std::unordered_map<KeyFrame*, size_t>::const_iterator kfit, kfend(mp->kFWithFIdx.end());
        for ( kfit = mp->kFWithFIdx.begin(); kfit != kfend; kfit++)
        {
            KeyFrame* kf = kfit->first;
            // if ( kf->keyF )
            const size_t keyPos = kfit->second;
            // if ( kf->numb <= kfNumbDif || kf->numb > lastKF->numb)
            //     continue;
            // const TrackedKeys& keys = kf->keys;
            if ( kf->numb > lastKF->numb )
                continue;
            // if ( kf->numb <= kfNumbDif/*  && !kf->keyF */ )
            //     continue;
            matchedIdxs[i].emplace_back(kf->numb,(int)keyPos);
            if ( !addedKF[kf->numb] )
            {
                addedKF[kf->numb] = true;
                actKeyF.emplace_back(kf);
            }
            // if ( keys.estimatedDepth[keyPos] > 0 && kf->numb != 0)
            //     matchedIdxs[i].emplace_back( - kf->numb,(int)keys.rightIdxs[keyPos]);
            // if ( allKeys.find(kf->numb) == allKeys.end() )
            // {
            //     allKeys.insert(kf->numb);
            // }
        }
    }



}

void LocalMapper::predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, std::vector<std::pair<float, float>>& keysAngles, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints)
{
    // cv::KeyPoint::convert(keys.keyPoints, predPoints);
    // predPoints.resize(keys.keyPoints.size());
    // keysAngles.resize(keys.keyPoints.size(), -5.0);
    const Eigen::Matrix4d camPoseInvR = (camPose * zedPtr->extrinsics).inverse();

    const double fxr {zedPtr->cameraRight.fx};
    const double fyr {zedPtr->cameraRight.fy};
    const double cxr {zedPtr->cameraRight.cx};
    const double cyr {zedPtr->cameraRight.cy};
    
    const cv::Point2f noPoint(-1.-1);
    const float noAngle {-5.0};
    for ( size_t i {0}, end{p4d.size()}; i < end; i ++)
    {
        const Eigen::Vector4d& wp = p4d[i].first;
        const std::pair<int,int>& keyPos = p4d[i].second;

        Eigen::Vector4d p = camPoseInv * wp;
        Eigen::Vector4d pR = camPoseInvR * wp;

        if ( p(2) <= 0.0 || pR(2) <= 0.0)
        {
            predPoints.emplace_back(noPoint, noPoint);
            keysAngles.emplace_back(noAngle, noAngle);
            continue;
        }

        const double invZ = 1.0f/p(2);
        const double invZR = 1.0f/pR(2);

        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        double uR {fxr*pR(0)*invZR + cxr};
        double vR {fyr*pR(1)*invZR + cyr};

        float angL {noAngle}, angR {noAngle};

        if ( keyPos.first >= 0 )
        {
            angL = atan2((float)v - keys.keyPoints[keyPos.first].pt.y, (float)u - keys.keyPoints[keyPos.first].pt.x);
        }
        if ( keyPos.second >= 0 )
        {
            angR = atan2((float)v - keys.rightKeyPoints[keyPos.second].pt.y, (float)u - keys.rightKeyPoints[keyPos.second].pt.x);
        }

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        cv::Point2f predL((float)u, (float)v), predR((float)uR, (float)vR);

        if ( u < 15 || v < 15 || u >= w - 15 || v >= h - 15 )
        {
            predL = noPoint;
            angL = noAngle;
        }
        if ( uR < 15 || vR < 15 || uR >= w - 15 || vR >= h - 15 )
        {
            predR = noPoint;
            angR = noAngle;
        }

        predPoints.emplace_back(predL, predR);
        keysAngles.emplace_back(angL, angR);

    }
}

void LocalMapper::predictKeysPosRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints)
{
    // cv::KeyPoint::convert(keys.keyPoints, predPoints);
    // predPoints.resize(keys.keyPoints.size());
    // keysAngles.resize(keys.keyPoints.size(), -5.0);
    const Eigen::Matrix4d camPoseInvR = (camPose * zedCam->extrinsics).inverse();

    const double fx {zedCam->cameraLeft.fx};
    const double fy {zedCam->cameraLeft.fy};
    const double cx {zedCam->cameraLeft.cx};
    const double cy {zedCam->cameraLeft.cy};
    predPoints.reserve(p4d.size());
    const cv::Point2f noPoint(-1.-1);
    for ( size_t i {0}, end{p4d.size()}; i < end; i ++)
    {
        const Eigen::Vector4d& wp = p4d[i].first;
        const std::pair<int,int>& keyPos = p4d[i].second;

        Eigen::Vector4d p = camPoseInv * wp;
        Eigen::Vector4d pR = camPoseInvR * wp;

        if ( p(2) <= 0.0 || pR(2) <= 0.0)
        {
            predPoints.emplace_back(noPoint, noPoint);
            continue;
        }

        const double invZ = 1.0f/p(2);
        const double invZR = 1.0f/pR(2);

        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        double uR {fx*pR(0)*invZR + cx};
        double vR {fy*pR(1)*invZR + cy};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        cv::Point2f predL((float)u, (float)v), predR((float)uR, (float)vR);

        if ( u < 0 || v < 0 || u >= w || v >= h )
        {
            predL = noPoint;
        }
        if ( uR < 0 || vR < 0 || uR >= w || vR >= h )
        {
            predR = noPoint;
        }

        predPoints.emplace_back(predL, predR);

    }
}

void LocalMapper::calcAllMpsOfKFR(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale)
{
    const size_t keysSize {lastKF->keys.keyPoints.size()};
    const size_t RkeysSize {lastKF->keys.rightKeyPoints.size()};
    const int kfNumbDif {lastKF->numb - kFsize};
    const double fxr = zedPtr->cameraRight.fx;
    const double fyr = zedPtr->cameraRight.fy;
    const double cxr = zedPtr->cameraRight.cx;
    const double cyr = zedPtr->cameraRight.cy;
    const TrackedKeys& keys = lastKF->keys;
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                zp = (double)keys.medianDepth;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = lastKF->pose.pose * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - lastKF->pose.pose.block<3,1>(0,3);
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( lastKF->unMatchedF[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - lastKF->pose.pose.block<3,1>(0,3);
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    int realCount {0};
    for ( size_t i{keysSize}; i < keysSize + RkeysSize; i++, realCount++)
    {
        if ( keys.leftIdxs[realCount] >= 0 )
            continue;
        if ( lastKF->unMatchedFR[realCount] >= 0 )
            continue;
        const double zp = (double)keys.medianDepth;
        const double xp = (double)(((double)keys.rightKeyPoints[realCount].pt.x-cx)*zp/fx);
        const double yp = (double)(((double)keys.rightKeyPoints[realCount].pt.y-cy)*zp/fy);
        Eigen::Vector4d p4dcam(xp, yp, zp, 1);
        p4dcam = lastKF->pose.pose * p4dcam;
        p4d.emplace_back(p4dcam, std::make_pair(-1, (int)realCount));
        Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
        pos = pos - lastKF->pose.pose.block<3,1>(0,3);
        float dist = pos.norm();
        int level = keys.rightKeyPoints[realCount].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
        
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale)
{
    const size_t keysSize {lastKF->keys.keyPoints.size()};
    const size_t RkeysSize {lastKF->keys.rightKeyPoints.size()};
    const int kfNumbDif {lastKF->numb - kFsize};
    const double fxr = zedPtr->cameraRight.fx;
    const double fyr = zedPtr->cameraRight.fy;
    const double cxr = zedPtr->cameraRight.cx;
    const double cyr = zedPtr->cameraRight.cy;
    const TrackedKeys& keys = lastKF->keys;
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                continue;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = lastKF->pose.pose * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - lastKF->pose.pose.block<3,1>(0,3);
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( lastKF->unMatchedF[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - lastKF->pose.pose.block<3,1>(0,3);
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::calcAllMpsOfKFROnlyEstB(const Zed_Camera* zedCam, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale, const bool back)
{
    const int kfNumbDif {lastKF->numb - kFsize};
    const Eigen::Matrix4d& pose4d = (back) ? lastKF->backPose : lastKF->pose.pose;
    const std::vector<int>& unMatchVec = (back) ? lastKF->unMatchedFB : lastKF->unMatchedF;
    const std::vector<MapPoint*>& localMPs = (back) ? lastKF->localMapPointsB : lastKF->localMapPoints;
    const TrackedKeys& keys = (back) ? lastKF->keysB : lastKF->keys;
    const Eigen::Vector3d traToRem = pose4d.block<3,1>(0,3);
    double fxr = zedCam->cameraLeft.fx;
    double fyr = zedCam->cameraLeft.fy;
    double cxr = zedCam->cameraLeft.cx;
    double cyr = zedCam->cameraLeft.cy;
    const size_t keysSize {keys.keyPoints.size()};
    const size_t RkeysSize {keys.rightKeyPoints.size()};
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = localMPs[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                continue;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = pose4d * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - traToRem;
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( unMatchVec[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - traToRem;
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::triangulateNewPoints(std::vector<vio_slam::KeyFrame *>& activeKF)
{
    const int kFsize {10};
    KeyFrame* lastKF = activeKF.front();
    const int lastKFIdx = lastKF->numb;
    std::vector<vio_slam::KeyFrame *> actKeyF;
    actKeyF.emplace_back(lastKF);
    actKeyF.reserve(kFsize);
    // const int startact {1};
    // const int endact {startact + kFsize - 1};
    // for (size_t i {startact}; i < endact; i++)
    // {
    //     const int idx {lastKFIdx - i};
    //     if ( idx < 0 )
    //         break;
    //     if ( lastKF->numb == idx )
    //         continue;
    //     actKeyF.emplace_back(map->keyFrames.at(idx));
    // }
    actKeyF = activeKF;
    std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());

    std::vector<Eigen::Vector4d> p4d;
    // calcp4d(lastKF, p4d);
    std::vector<vio_slam::KeyFrame *> allSeenKF = actKeyF;
    calcAllMpsOfKF(matchedIdxs, lastKF, allSeenKF, kFsize, p4d);
    const int aKFsize {actKeyF.size()};
    bool first = true;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        // if ( (*it)->numb < lastKFIdx - kFsize)
        //     continue;
        std::vector<float> keysAngles;
        std::vector<cv::Point2f> predPoints;
        predictKeysPos(lastKF->keys, lastKF->pose.pose, (*it)->pose.poseInverse, keysAngles, p4d, predPoints);
        // drawPred(lastKF, lastKF->keys.keyPoints, predPoints);
        fm->matchLocalBA(matchedIdxs, lastKF, (*it), aKFsize, 10, first, keysAngles, predPoints);
        // if (first)
        // drawLBA("LBA matches",matchedIdxs, lastKF,(*it));
        cv::waitKey(1);
        first = false;
        
    }

    std::unordered_map<int, Eigen::Matrix<double,3,4>> allProjMatrices;
    allProjMatrices.reserve(2 * allSeenKF.size());

    calcProjMatrices(allProjMatrices, allSeenKF);
    std::vector<MapPoint*> pointsToAdd;
    pointsToAdd.resize(lastKF->keys.keyPoints.size(),nullptr);
    // pointsToAdd.reserve(lastKF->keys.keyPoints.size());
    // lastKF->localMapPoints.reserve(lastKF->keys.keyPoints.size());
    for ( size_t i{0}, end {lastKF->keys.keyPoints.size()}; i < end; i ++)
    {
        std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatches(matchesOfPoint, allProjMatrices, proj_mat, pointsVec, actKeyF);
        Eigen::Vector4d vecCalc = lastKF->pose.getInvPose() * p4d[i];
        // if ( lastKF->unMatchedF[i] >= 0 )
        // {
        //     vecCalc = lastKF->localMapPoints[i]->getWordPose4d();
        // }
        // else
        // {
        //     if (lastKF->keys.estimatedDepth[i] > 0)
        //         zp = (double)lastKF->keys.estimatedDepth[i];
        //     else
        //         zp = 200.0;
        //     const double xp = (double)(((double)lastKF->keys.keyPoints[i].pt.x-cx)*zp/fx);
        //     const double yp = (double)(((double)lastKF->keys.keyPoints[i].pt.y-cy)*zp/fy);
        //     vecCalc = Eigen::Vector4d(xp, yp, zp, 1);
        // }
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNew(lastKF, i, vecCalc, matchesOfPoint, allProjMatrices, proj_mat, pointsVec) )
            continue;

        vecCalc = lastKF->pose.getInvPose() * vecCalc;
        vec3d(0) = vecCalc(0);
        vec3d(1) = vecCalc(1);
        vec3d(2) = vecCalc(2);

        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, false);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNew(lastKF, i, vecCalc, matchesOfPoint, allProjMatrices, proj_mat, pointsVec) )
            continue;

        addMultiViewMapPoints(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        
    }
    // addToMap(lastKF, pointsToAdd);
    addToMapRemoveCon(lastKF, pointsToAdd, matchedIdxs);
}

void LocalMapper::triangulateNewPointsR(std::vector<vio_slam::KeyFrame *>& activeKF)
{
    const int kFsize {actvKFMaxSize};
    std::vector<vio_slam::KeyFrame *> actKeyF;
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    KeyFrame* lastKF = actKeyF.front();
    const int lastKFIdx = lastKF->numb;
    // std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());
    std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>> matchedIdxs;

    // std::vector<Eigen::Vector4d> p4d;
    std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>> p4d;
    std::vector<float> maxDistsScale;
    // calcAllMpsOfKFR(matchedIdxs, lastKF, kFsize, p4d,maxDistsScale);
    calcAllMpsOfKFROnlyEst(matchedIdxs, lastKF, kFsize, p4d,maxDistsScale);
    // calcAllMpsOfKF(matchedIdxs, lastKF, allSeenKF, kFsize, p4d);
    const int aKFsize {actKeyF.size()};
    {
    bool first = true;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        std::vector<std::pair<float, float>> keysAngles;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> predPoints;
        // predict keys for both right and left camera
        predictKeysPosR(lastKF->keys, (*it)->pose.pose, (*it)->pose.poseInverse, keysAngles, p4d, predPoints);
        int matches = fm->matchByProjectionRPredLBA(lastKF, (*it), matchedIdxs, 4, predPoints, keysAngles, maxDistsScale, p4d, true);
        first = false;
        
    }
    }

    // std::unordered_map<int, Eigen::Matrix<double,3,4>> allProjMatrices;
    std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>> allProjMatrices;
    allProjMatrices.reserve(2 * actKeyF.size());
    calcProjMatricesR(allProjMatrices, actKeyF);
    std::vector<MapPoint*> pointsToAdd;
    const size_t mpCandSize {matchedIdxs.size()};
    pointsToAdd.resize(mpCandSize,nullptr);
    int newMaPoints {0};
    for ( size_t i{0}; i < mpCandSize; i ++)
    {
        std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesR(matchesOfPoint, allProjMatrices, proj_mat, pointsVec);
        Eigen::Vector4d vecCalc = lastKF->pose.getInvPose() * p4d[i].first;
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewR(lastKF, vecCalc, matchesOfPoint, proj_mat, pointsVec) )
            continue;

        addMultiViewMapPointsR(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        newMaPoints++;
    }

    // std::cout << "New MapPoints " << newMaPoints << std::endl;
    // addToMap(lastKF, pointsToAdd);
    addNewMapPoints(lastKF, pointsToAdd, matchedIdxs);
}

void LocalMapper::triangulateNewPointsRB(const Zed_Camera* zedCam,std::vector<vio_slam::KeyFrame *>& activeKF, const bool back)
{
    const int kFsize {actvKFMaxSize};
    std::vector<vio_slam::KeyFrame *> actKeyF;
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    KeyFrame* lastKF = actKeyF.front();
    const int lastKFIdx = lastKF->numb;
    // std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());
    std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>> matchedIdxs;

    // std::vector<Eigen::Vector4d> p4d;
    std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>> p4d;
    std::vector<float> maxDistsScale;
    

    calcAllMpsOfKFROnlyEstB(zedCam, matchedIdxs, lastKF, kFsize, p4d,maxDistsScale, back);


    const int aKFsize {actKeyF.size()};
    {
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> predPoints;
        const Eigen::Matrix4d& camPose = (back) ? (*it)->backPose : (*it)->pose.pose;
        const Eigen::Matrix4d& camPoseInv = (back) ? (*it)->backPoseInv : (*it)->pose.poseInverse;
        predictKeysPosRB(zedCam, camPose, camPoseInv, p4d, predPoints);
        int matches = fm->matchByProjectionRPredLBAB(zedCam,lastKF, (*it), matchedIdxs, 4, predPoints, maxDistsScale, p4d, back);
        
    }
    }

    // std::unordered_map<int, Eigen::Matrix<double,3,4>> allProjMatrices;
    std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>> allProjMatrices;
    allProjMatrices.reserve(2 * actKeyF.size());
    calcProjMatricesRB(zedCam,allProjMatrices, actKeyF, back);
    std::vector<MapPoint*> pointsToAdd;
    const size_t mpCandSize {matchedIdxs.size()};
    pointsToAdd.resize(mpCandSize,nullptr);
    const Eigen::Matrix4d lastKFPoseInv = (back) ? lastKF->backPoseInv : lastKF->pose.poseInverse;
    const Eigen::Matrix4d lastKFPose = (back) ? lastKF->backPose : lastKF->pose.pose;
    for ( size_t i{0}; i < mpCandSize; i ++)
    {
        std::vector<std::pair<vio_slam::KeyFrame *, std::pair<int, int>>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesRB(matchesOfPoint, allProjMatrices, proj_mat, pointsVec,back);
        Eigen::Vector4d vecCalc = lastKFPoseInv * p4d[i].first;
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKFPose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewRB(lastKF, vecCalc, matchesOfPoint, proj_mat, pointsVec, back) )
            continue;

        addMultiViewMapPointsRB(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i, back);
    }
    addNewMapPointsB(lastKF, pointsToAdd, matchedIdxs,back);
}

void LocalMapper::triangulateNewPointsB(Map* map, std::vector<vio_slam::KeyFrame *>& activeKF)
{
    const int kFsize {actvKFMaxSize/2};
    KeyFrame* lastKF = activeKF.front();
    const int lastKFIdx = lastKF->numb;
    std::vector<vio_slam::KeyFrame *> actKeyF;
    actKeyF.emplace_back(lastKF);
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    std::vector<std::vector<std::pair<int, int>>> matchedIdxs(lastKF->keys.keyPoints.size(),std::vector<std::pair<int, int>>());

    std::vector<Eigen::Vector4d> p4d;
    // calcp4d(lastKF, p4d);
    std::vector<vio_slam::KeyFrame *> allSeenKF = actKeyF;
    calcAllMpsOfKF(matchedIdxs, lastKF, allSeenKF, kFsize, p4d);
    const int aKFsize {actKeyF.size()};
    bool first = true;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        // if ( (*it)->numb < lastKFIdx - kFsize)
        //     continue;
        std::vector<float> keysAngles;
        std::vector<cv::Point2f> predPoints;
        predictKeysPos(lastKF->keys, lastKF->pose.pose, (*it)->pose.poseInverse, keysAngles, p4d, predPoints);
        // drawPred(lastKF, lastKF->keys.keyPoints, predPoints);
        fm->matchLocalBA(matchedIdxs, lastKF, (*it), aKFsize, 10, first, keysAngles, predPoints);
        // if (first)
        // drawLBA("LBA matches",matchedIdxs, lastKF,(*it));
        // cv::waitKey(1);
        first = false;
        
    }

    std::unordered_map<int, Eigen::Matrix<double,3,4>> allProjMatrices;
    allProjMatrices.reserve(2 * allSeenKF.size());

    calcProjMatrices(allProjMatrices, allSeenKF);
    std::vector<MapPoint*> pointsToAdd;
    pointsToAdd.resize(lastKF->keys.keyPoints.size(),nullptr);
    for ( size_t i{0}, end {lastKF->keys.keyPoints.size()}; i < end; i ++)
    {
        std::vector<std::pair<int, int>>& matchesOfPoint = matchedIdxs[i];
        if (matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesB(map, matchesOfPoint, allProjMatrices, proj_mat, pointsVec, actKeyF);
        Eigen::Vector4d vecCalc = lastKF->pose.getInvPose() * p4d[i];
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewB(map, lastKF, i, vecCalc, matchesOfPoint, allProjMatrices, proj_mat, pointsVec) )
            continue;

        vecCalc = lastKF->pose.getInvPose() * vecCalc;
        vec3d(0) = vecCalc(0);
        vec3d(1) = vecCalc(1);
        vec3d(2) = vecCalc(2);

        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, false);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewB(map, lastKF, i, vecCalc, matchesOfPoint, allProjMatrices, proj_mat, pointsVec) )
            continue;

        addMultiViewMapPointsB(map, vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        
    }
    // addToMap(lastKF, pointsToAdd);
    addToMapRemoveConB(map, lastKF, pointsToAdd, matchedIdxs);
}

bool LocalMapper::checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

bool LocalMapper::checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    posC = qc1c2 * posC + tc1c2;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

void LocalMapper::localBA(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        // localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        std::vector<cv::Point2f> kFkeys, mpKeys;
        if ( (*it)->fixed )
            fixedKF = true;
        // Logging("BEFORE BA", (*it)->getPose(),3);
        int count {0};
        std::vector<MapPoint*>::const_iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++, count++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            // if ( mp->kFWithFIdx.size() < 3 )
            //     continue;
            // if ( (*itmp)->kFWithFIdx.size() < 3 )
            //     continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, size_t>::const_iterator kf, endkf(mp->kFWithFIdx.end());
            for (kf = mp->kFWithFIdx.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                // if ( kFCand->active )
                // {
                //     hasKF = true;
                //     // keyFMatches++;
                //     continue;
                // }
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                }
                blocks++;
                // keyFMatches++;
            }
            const Eigen::Matrix4d camPose = (*it)->pose.getInvPose();
            Eigen::Vector4d pointCam = camPose * mp->getWordPose4d();
            if ( pointCam(2) <= 0)
                continue;
            double u {fx*pointCam(0)/pointCam(2) + cx};
            double v {fy*pointCam(1)/pointCam(2) + cy};
            kFkeys.emplace_back((*it)->keys.keyPoints[count].pt);
            mpKeys.emplace_back((float)u,(float)v);
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));

            // else if ( keyFMatches < 3 )
            // {
            //     outliersMP.emplace_back(mp);
            // }
        }
#if DRAWMATCHES

        drawPointsTemp<cv::Point2f,cv::Point2f>("lba", (*it)->rLeftIm, kFkeys, mpKeys);
        // Logging("mpkeys",mpKeys[10], 3);
        // Logging("kFkeys",kFkeys[10], 3);
        cv::waitKey(1);
#endif
    }
    Logging("fixed size", fixedKFs.size(),3);
    Logging("local size", localKFs.size(),3);
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    // Logging("mappoints size", allMapPoints.size(),3);
    
    // Logging("before", localKFs[actKeyF.front()],3);
    // Logging("Local Bundle Adjustment Starting...", "",3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, size_t>::iterator kf, endkf(itmp->first->kFWithFIdx.end());
        for (kf = itmp->first->kFWithFIdx.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] )
                continue;
            if ( (!itmp->first->GetInFrame() && itmp->first->kFWithFIdx.size() < minCount) )
            {
                mpOutliers[mpCount] = true;
                continue;
            }
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            KeyFrame* kftemp = kf->first;
            size_t keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            // Logging("size", kf->first->keys.keyPoints.size(),3);
            // Logging("index", kf->second,3);
            // Logging("kfnumb", kf->first->numb,3);
            ceres::CostFunction* costf;
            // if ( (*kf).first->keys.close[kf->second] )
            // {
            //     const cv::KeyPoint& obs = kf->first->keys.keyPoints[kf->second];
            //     const float depth = kf->first->keys.estimatedDepth[kf->second];
            //     Eigen::Vector3d obs3d = get3d(obs, depth);
            //     costf = LocalBundleAdjustmentICP::Create(K, obs3d, 1.0f);
            // }
            // else
            // {
                const cv::KeyPoint& obs = kf->first->keys.keyPoints[kf->second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                costf = LocalBundleAdjustment::Create(K, obs2d, 1.0f);
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }

        }
        if ( mpIsOut || (!itmp->first->GetInFrame() && itmp->first->kFWithFIdx.size() < minCount) )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 4;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
    //     Logging("summ", summary.BriefReport(),3);
    // Logging("lelout", lelout, 3);
    wrongMatches.clear();

    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, size_t>::iterator kf, endkf(mp->kFWithFIdx.end());
        for (kf = mp->kFWithFIdx.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            const cv::KeyPoint& kp = kfCand->keys.keyPoints[kf->second];
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            
            bool outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const int keyPos = mp->kFWithFIdx.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        // Logging("AFTER BA", localkf->first->getPose(),3);
        // localkf->first->active = true;
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] )
            itmp->first->SetIsOutlier(true);
        else
        {
            // Logging ("before pose", itmp->first->getWordPose3d(), 3);
            // itmp->first->setWordPose3d(itmp->second);
            // Logging ("after pose", itmp->first->getWordPose3d(), 3);
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    // std::vector<MapPoint*>::iterator mpout, mpoutend(outliersMP.end());
    // for ( mpout = outliersMP.begin(); mpout != mpoutend; mpout++ )
    // {
    //     MapPoint* mp = *mpout;
    //     if ( !mp->GetInFrame() )
    //         mp->SetIsOutlier(true);
    // }
    

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    
    // Logging("after", localKFs[actKeyF.front()],3);
}

void LocalMapper::localBAB(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs, fixedKFs;
    std::unordered_map<KeyFrame*, KeyFrame*> localKFsB, fixedKFsB;
    // std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    localKFsB.reserve(actKeyF.size());
    fixedKFsB.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        KeyFrame* KFB = (*it)->KFBack;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        localKFsB[KFB] = *it;
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        // localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        std::vector<cv::Point2f> kFkeys, mpKeys;
        if ( (*it)->fixed )
            fixedKF = true;
        // Logging("BEFORE BA", (*it)->getPose(),3);
        int count {0};
        std::vector<MapPoint*>::const_iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++, count++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            // if ( mp->kFWithFIdx.size() < 3 )
            //     continue;
            // if ( (*itmp)->kFWithFIdx.size() < 3 )
            //     continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, size_t>::const_iterator kf, endkf(mp->kFWithFIdx.end());
            for (kf = mp->kFWithFIdx.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                }
                blocks++;
                // keyFMatches++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));

        }

        std::vector<MapPoint*>::const_iterator endmpB((*it)->localMapPointsB.end());
        for ( itmp = (*it)->localMapPointsB.begin(); itmp != endmpB; itmp++, count++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            bool hasKF {true};
            std::unordered_map<KeyFrame*, size_t>::const_iterator kf, endkf(mp->kFWithFIdx.end());
            for (kf = mp->kFWithFIdx.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (localKFsB.find(kFCand) == localKFsB.end())
                {
                    fixedKFsB[kFCand] = kFCand->KFBack;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            Eigen::Vector4d p4d = (*itmp)->getWordPose4d();
            if ( isnan(p4d(0)))
            {
                Logging("mp is nan","",3);
            }
            if ( isnan(p4d(1)))
            {
                Logging("mp is nan","",3);
            }
            if ( isnan(p4d(2)))
            {
                Logging("mp is nan","",3);
            }
            if ( isnan(p4d(3)))
            {
                Logging("mp is nan","",3);
            }

        }
#if DRAWMATCHES

        drawPointsTemp<cv::Point2f,cv::Point2f>("lba", (*it)->rLeftIm, kFkeys, mpKeys);
        // Logging("mpkeys",mpKeys[10], 3);
        // Logging("kFkeys",kFkeys[10], 3);
        cv::waitKey(1);
#endif
    }
    Logging("fixed size", fixedKFs.size(),3);
    Logging("local size", localKFs.size(),3);
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        KeyFrame* lastKFB = lastKF->KFBack;
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
        localKFsB.erase(lastKFB);
        fixedKFsB[lastKFB] = lastKFB->KFBack;
    }

    Eigen::Matrix4d transfC1C2;
    transfC1C2 << -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, -1.0, -0.3,
         0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix4d transfC1C2Inv = transfC1C2.inverse();
    // Logging("mappoints size", allMapPoints.size(),3);
    
    // Logging("before", localKFs[actKeyF.front()],3);
    // Logging("Local Bundle Adjustment Starting...", "",3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix<double, 7,1> trials71 = Converter::Matrix4dToMatrix_7_1(transfC1C2Inv);

    Eigen::Matrix3d Rott = transfC1C2Inv.block<3,3>(0,0);
    const Eigen::Quaterniond qt(Rott);
    const Eigen::Matrix<double,3,1>trials71t = trials71.block<3,1>(0,0);
    // const Eigen::Matrix<double,3,1> trials71t = trials71.block<3,1>(0,0);
    // const Eigen::Matrix<double,3,1> trials71t = trials71.block<4,1>(3,0);
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, size_t>::iterator kf, endkf(itmp->first->kFWithFIdx.end());
        for (kf = itmp->first->kFWithFIdx.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] )
                continue;
            if ( (!itmp->first->GetInFrame() && itmp->first->kFWithFIdx.size() < minCount) )
            {
                mpOutliers[mpCount] = true;
                continue;
            }
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            KeyFrame* kftemp = kf->first;
            size_t keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            // Logging("size", kf->first->keys.keyPoints.size(),3);
            // Logging("index", kf->second,3);
            // Logging("kfnumb", kf->first->numb,3);
            ceres::CostFunction* costf;
            // if ( (*kf).first->keys.close[kf->second] )
            // {
            //     const cv::KeyPoint& obs = kf->first->keys.keyPoints[kf->second];
            //     const float depth = kf->first->keys.estimatedDepth[kf->second];
            //     Eigen::Vector3d obs3d = get3d(obs, depth);
            //     costf = LocalBundleAdjustmentICP::Create(K, obs3d, 1.0f);
            // }
            // else
            // {
                const cv::KeyPoint& obs = kf->first->keys.keyPoints[kf->second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                costf = LocalBundleAdjustment::Create(K, obs2d, 1.0f);
                // costf = LocalBundleAdjustmentB::Create(K, trials71t, Rott, obs2d, 1.0f);

                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                costf = LocalBundleAdjustment::Create(K, obs2d, 1.0f);
                // costf = LocalBundleAdjustmentB::Create(K, trials71t, Rott, obs2d, 1.0f);

                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            else if (localKFsB.find(kf->first) != localKFsB.end())
            {
                // costf = LocalBundleAdjustment::Create(K, obs2d, 1.0f);
                costf = LocalBundleAdjustmentB::Create(K, trials71t, Rott, obs2d, 1.0f);
                KeyFrame* kFF = kf->first->KFBack;

                ordering->AddElementToGroup(localKFs[kFF].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kFF].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kFF].block<3,1>(0,0).data(), localKFs[kFF].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kFF].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kFF->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kFF].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kFF].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFsB.find(kf->first) != fixedKFsB.end())
            {
                // costf = LocalBundleAdjustment::Create(K, obs2d, 1.0f);
                costf = LocalBundleAdjustmentB::Create(K, trials71t, Rott, obs2d, 1.0f);
                KeyFrame* kFF = kf->first->KFBack;

                ordering->AddElementToGroup(fixedKFs[kFF].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kFF].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kFF].block<3,1>(0,0).data(), fixedKFs[kFF].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kFF].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kFF].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kFF].block<4,1>(3,0).data());
            }


        }
        if ( mpIsOut || (!itmp->first->GetInFrame() && itmp->first->kFWithFIdx.size() < minCount) )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 4;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
    //     Logging("summ", summary.BriefReport(),3);
    // Logging("lelout", lelout, 3);
    wrongMatches.clear();

    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, size_t>::iterator kf, endkf(mp->kFWithFIdx.end());
        for (kf = mp->kFWithFIdx.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            if ( localKFs.find(kfCand) == localKFs.end())
            {
                kfCand = kf->first->KFBack;
                if ( localKFsB.find(kfCand) == localKFsB.end() )
                    continue;

            } 
            

            const cv::KeyPoint& kp = kfCand->keys.keyPoints[kf->second];
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            
            bool outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const int keyPos = mp->kFWithFIdx.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        Eigen::Matrix4d newPoseInv = Converter::Matrix_7_1_ToMatrix4d(localkf->second);
        localkf->first->pose.setInvPose(newPoseInv);
        // Logging("first",localkf->first->pose.pose,3);
        Eigen::Matrix4d newPoseInvB = transfC1C2Inv*newPoseInv;

        localkf->first->KFBack->pose.setInvPose(newPoseInvB);
        // Logging("second",localkf->first->KFBack->pose.pose,3);

        // Logging("AFTER BA", localkf->first->getPose(),3);
        // localkf->first->active = true;
        localkf->first->LBA = true;
    }


    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] )
            itmp->first->SetIsOutlier(true);
        else
        {
            // Logging ("before pose", itmp->first->getWordPose3d(), 3);
            itmp->first->setWordPose3d(itmp->second);
            // Logging ("after pose", itmp->first->getWordPose3d(), 3);
            // itmp->first->updatePos(map->keyFrames.at(itmp->first->kdx)->pose.getInvPose(), zedPtr);
        }
    }

    // std::vector<MapPoint*>::iterator mpout, mpoutend(outliersMP.end());
    // for ( mpout = outliersMP.begin(); mpout != mpoutend; mpout++ )
    // {
    //     MapPoint* mp = *mpout;
    //     if ( !mp->GetInFrame() )
    //         mp->SetIsOutlier(true);
    // }
    

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    mapB->LBADone = true;
    
    // Logging("after", localKFs[actKeyF.front()],3);
}

void LocalMapper::localBAR(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LBAID = lastActKF;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        // Logging("BEFORE BA", (*it)->getPose(),3);
        std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == lastActKF )
                continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
        std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
        for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == lastActKF )
                continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                const std::pair<int,int>& keyPos = kf->second;
                if ( keyPos.first >= 0 || keyPos.second < 0 )
                    continue;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix3d& Kr = zedPtr->cameraRight.intrisics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    // Timer baiter("baITER");
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
            {
                mpOutliers[mpCount] = true;
                break;
            }
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = kf->first->keys.keyPoints[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
                close = kf->first->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
                right = true;
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
                right = true;

                ordering->AddElementToGroup(itmp->second.data(), 0);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        if ( mpIsOut /* || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) */ )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 1;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
        // Logging("summ", summary.FullReport(),3);
    // Logging("lelout", lelout, 3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);

    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        // Logging("AFTER BA", localkf->first->getPose(),3);
        // localkf->first->active = true;
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            // Logging ("before pose", itmp->first->getWordPose3d(), 3);
            // itmp->first->setWordPose3d(itmp->second);
            // Logging ("after pose", itmp->first->getWordPose3d(), 3);
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    // std::vector<MapPoint*>::iterator mpout, mpoutend(outliersMP.end());
    // for ( mpout = outliersMP.begin(); mpout != mpoutend; mpout++ )
    // {
    //     MapPoint* mp = *mpout;
    //     if ( !mp->GetInFrame() )
    //         mp->SetIsOutlier(true);
    // }
    

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    
    // Logging("after", localKFs[actKeyF.front()],3);
}

void LocalMapper::loopClosureR(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    Timer loopClosureTime("loopClosureTimer");
    std::cout << "Loop Closure Detected! Starting Optimization.." << std::endl;
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    KeyFrame* lCCand = actKeyF.front();
    localKFs[lCCand] = Converter::Matrix4dToMatrix_7_1(map->LCPose.inverse());
    lCCand->fixed = true;
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LCID = lastActKF;
        if ( (*it)->numb == lastActKF )
            continue;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        // Logging("BEFORE BA", (*it)->getPose(),3);
        std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LCID == lastActKF )
                continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LCID == lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LCID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LCID = lastActKF;
        }
        std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
        for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LCID == lastActKF )
                continue;
            
            bool hasKF {true};
            // int keyFMatches {0};
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                const std::pair<int,int>& keyPos = kf->second;
                if ( keyPos.first >= 0 || keyPos.second < 0 )
                    continue;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LCID == lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LCID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LCID = lastActKF;
        }
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    Timer baiter("baITER");
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
                break;
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = kf->first->keys.keyPoints[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
                close = kf->first->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
                right = true;
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
                right = true;

                ordering->AddElementToGroup(itmp->second.data(), 0);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        if ( mpIsOut /* || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) */ )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 8;
    options.max_num_iterations = 45;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
        Logging("summ", summary.FullReport(),3);
    // Logging("lelout", lelout, 3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);
    int wrMCnt {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
            {
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                wrMCnt ++;
            }
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                        wrMCnt ++;
                    }
                }
            }

        }
    }
    std::cout << "wrongMatches Count :" << wrMCnt << std::endl;
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    

    map->endLCIdx = actKeyF.front()->numb;
    map->LCDone = true;
    map->LCStart = false;
    map->aprilTagDetected = false;
    
}

void LocalMapper::insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const int lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LBAID == lastActKF )
            continue;
        
        bool hasKF {true};
        // int keyFMatches {0};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf = (back) ? mp->kFMatchesB.begin() : mp->kFMatches.begin();
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkf = (back) ? mp->kFMatchesB.end() : mp->kFMatches.end();
        for (; kf != endkf; kf++)
        {
            KeyFrame* kFCand = kf->first;
            if ( !kFCand->keyF || kFCand->numb > lastActKF )
                continue;
            if (kFCand->LBAID == lastActKF )
                continue;
            if (localKFs.find(kFCand) == localKFs.end())
            {
                fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                kFCand->LBAID = lastActKF;
            }
            blocks++;
        }
        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LBAID = lastActKF;
    }
}

void LocalMapper::insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const int lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LCID == lastActKF )
            continue;
        
        bool hasKF {true};
        // int keyFMatches {0};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf = (back) ? mp->kFMatchesB.begin() : mp->kFMatches.begin();
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkf = (back) ? mp->kFMatchesB.end() : mp->kFMatches.end();
        for (; kf != endkf; kf++)
        {
            KeyFrame* kFCand = kf->first;
            if ( !kFCand->keyF || kFCand->numb > lastActKF )
                continue;
            if (kFCand->LCID == lastActKF )
                continue;
            if (localKFs.find(kFCand) == localKFs.end())
            {
                fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                kFCand->LCID = lastActKF;
            }
            blocks++;
        }
        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LCID = lastActKF;
    }
}

void LocalMapper::localBARB(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LBAID = lastActKF;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        insertMPsForLBA((*it)->localMapPoints,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLBA((*it)->localMapPointsR,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLBA((*it)->localMapPointsB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
        insertMPsForLBA((*it)->localMapPointsRB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix3d& KB = zedPtrB->cameraLeft.intrisics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qcr = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tcr = estimPoseRInv.block<3,1>(0,3);

    const Eigen::Matrix4d estimPoseBInv = zedPtr->TCamToCamInv;
    const Eigen::Matrix4d estimPoseBRInv = zedPtrB->extrinsics.inverse() * estimPoseBInv;
    const Eigen::Matrix3d qc1c2B = estimPoseBInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2B = estimPoseBInv.block<3,1>(0,3);
    const Eigen::Matrix3d qc1c2BR = estimPoseBRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2BR = estimPoseBRInv.block<3,1>(0,3);

    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
                break;
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                // if ( keyPos.second < 0 )
                // {
                //     std::cout << "POSSSS                  "<<keyPos.second << std::endl;
                //     std::cout << "close        "<<keys.close[keyPos.first] << std::endl;
                //     std::cout << "estimatedDepth        "<<keys.estimatedDepth[keyPos.first] << std::endl;
                //     std::cout << "rightIdxs        "<<keys.rightIdxs[keyPos.first] << std::endl;

                // }
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(itmp->first->kFMatchesB.end());
        for (kf = itmp->first->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatchesB.size() < minCount) )
                break;
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keysB;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2B, qc1c2B, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                
                if ( keyPos.second < 0 )
                    continue;
                // if ( keyPos.second < 0 )
                // {
                //     std::cout << "POSSSS2222222222        "<<keyPos.second << std::endl;
                //     std::cout << "close        "<<keys.close[keyPos.first] << std::endl;
                //     std::cout << "estimatedDepth        "<<keys.estimatedDepth[keyPos.first] << std::endl;
                //     std::cout << "rightIdxs        "<<keys.rightIdxs[keyPos.first] << std::endl;

                // }
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }
        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 1;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
        // Logging("summ", summary.FullReport(),3);
    // Logging("lelout", lelout, 3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);
    // wrongMatches.clear();


    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(mp->kFMatchesB.end());
        for (kf = mp->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keysB.keyPoints[keyPos.first];
                close = kfCand->keysB.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keysB.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlierR(KB,qc1c2B, tc1c2B, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keysB.rightKeyPoints[keyPos.second];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    bool outlierR = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            if ( mp->kFMatches.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
                kF->eraseMPConnection(keyPos);
                mp->eraseKFConnection(kF);
            }
            if ( mp->kFMatchesB.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatchesB.at(kF);
                kF->eraseMPConnectionB(keyPos);
                mp->eraseKFConnectionB(kF);
            }
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        localkf->first->setBackPose(localkf->first->pose.pose * zedPtr->TCamToCam);
        // Logging("AFTER BA", localkf->first->getPose(),3);
        // localkf->first->active = true;
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount && itmp->first->kFMatchesB.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            // Logging ("before pose", itmp->first->getWordPose3d(), 3);
            // itmp->first->setWordPose3d(itmp->second);
            // Logging ("after pose", itmp->first->getWordPose3d(), 3);
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    // std::vector<MapPoint*>::iterator mpout, mpoutend(outliersMP.end());
    // for ( mpout = outliersMP.begin(); mpout != mpoutend; mpout++ )
    // {
    //     MapPoint* mp = *mpout;
    //     if ( !mp->GetInFrame() )
    //         mp->SetIsOutlier(true);
    // }
    

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    
    // Logging("after", localKFs[actKeyF.front()],3);
}

void LocalMapper::loopClosureRB(std::vector<vio_slam::KeyFrame *>& actKeyF)
{
    Timer loopClosureTime("loopClosureTimer");
    std::cout << "Loop Closure Detected! Starting Optimization.." << std::endl;
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    int lastActKF {actKeyF.front()->numb};
    KeyFrame* lCCand = actKeyF.front();
    localKFs[lCCand] = Converter::Matrix4dToMatrix_7_1(map->LCPose.inverse());
    lCCand->fixed = true;
    // std::vector<MapPoint*> outliersMP;
    // outliersMP.reserve(1000);
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LCID = lastActKF;
        if ( (*it)->numb == lastActKF )
            continue;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        insertMPsForLC((*it)->localMapPoints,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLC((*it)->localMapPointsR,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLC((*it)->localMapPointsB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
        insertMPsForLC((*it)->localMapPointsRB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrisics;
    const Eigen::Matrix3d& KB = zedPtrB->cameraLeft.intrisics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qcr = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tcr = estimPoseRInv.block<3,1>(0,3);

    const Eigen::Matrix4d estimPoseBInv = zedPtr->TCamToCamInv;
    const Eigen::Matrix4d estimPoseBRInv = zedPtrB->extrinsics.inverse() * estimPoseBInv;
    const Eigen::Matrix3d qc1c2B = estimPoseBInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2B = estimPoseBInv.block<3,1>(0,3);
    const Eigen::Matrix3d qc1c2BR = estimPoseBRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2BR = estimPoseBRInv.block<3,1>(0,3);

    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount) )
                break;
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                // if ( keyPos.second < 0 )
                // {
                //     std::cout << "POSSSS                  "<<keyPos.second << std::endl;
                //     std::cout << "close        "<<keys.close[keyPos.first] << std::endl;
                //     std::cout << "estimatedDepth        "<<keys.estimatedDepth[keyPos.first] << std::endl;
                //     std::cout << "rightIdxs        "<<keys.rightIdxs[keyPos.first] << std::endl;

                // }
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(itmp->first->kFMatchesB.end());
        for (kf = itmp->first->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatchesB.size() < minCount) )
                break;
            // Logging("1","",3);
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                // Logging("2","",3);
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keysB;
            std::pair<int,int>& keyPos = kf->second;

            // if ( !kftemp->localMapPoints[keyPos] || kftemp->localMapPoints[keyPos] != (*itmp).first )
            //     continue;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            // Logging("3","",3);
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2B, qc1c2B, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
            }
            // }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                // problem.SetParameterBlockConstant(itmp->second.data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                
                if ( keyPos.second < 0 )
                    continue;
                // if ( keyPos.second < 0 )
                // {
                //     std::cout << "POSSSS2222222222        "<<keyPos.second << std::endl;
                //     std::cout << "close        "<<keys.close[keyPos.first] << std::endl;
                //     std::cout << "estimatedDepth        "<<keys.estimatedDepth[keyPos.first] << std::endl;
                //     std::cout << "rightIdxs        "<<keys.rightIdxs[keyPos.first] << std::endl;

                // }
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    // problem.SetParameterBlockConstant(itmp->second.data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }
        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
        // if ( timesIn < 3 )
        //     Logging("times in ", timesIn, 3);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    // options.num_threads = 4;
    options.num_threads = 8;
    options.max_num_iterations = 45;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.use_explicit_schur_complement = true;
    Timer solv("solvetimer");
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if ( !first )
        Logging("summ", summary.FullReport(),3);
    // Logging("lelout", lelout, 3);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);
    // wrongMatches.clear();


    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(mp->kFMatchesB.end());
        for (kf = mp->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keysB.keyPoints[keyPos.first];
                close = kfCand->keysB.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keysB.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlierR(KB,qc1c2B, tc1c2B, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keysB.rightKeyPoints[keyPos.second];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    bool outlierR = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlier )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            if ( mp->kFMatches.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
                kF->eraseMPConnection(keyPos);
                mp->eraseKFConnection(kF);
            }
            if ( mp->kFMatchesB.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatchesB.at(kF);
                kF->eraseMPConnectionB(keyPos);
                mp->eraseKFConnectionB(kF);
            }
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        localkf->first->setBackPose(localkf->first->pose.pose * zedPtr->TCamToCam);
        // Logging("AFTER BA", localkf->first->getPose(),3);
        // localkf->first->active = true;
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && itmp->first->kFMatches.size() < minCount && itmp->first->kFMatchesB.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            // Logging ("before pose", itmp->first->getWordPose3d(), 3);
            // itmp->first->setWordPose3d(itmp->second);
            // Logging ("after pose", itmp->first->getWordPose3d(), 3);
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    

    map->endLCIdx = actKeyF.front()->numb;
    map->LCDone = true;
    map->LCStart = false;
    map->aprilTagDetected = false;
    
    // Logging("after", localKFs[actKeyF.front()],3);
}

Eigen::Vector3d LocalMapper::get3d(const cv::KeyPoint& key, const float depth)
{
    const double zp = (double)depth;
    const double xp = (double)(((double)key.pt.x-cx)*zp/fx);
    const double yp = (double)(((double)key.pt.y-cy)*zp/fy);
    return Eigen::Vector3d(xp, yp, zp);
}

void LocalMapper::beginLocalMapping()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        // Logging("Local Mapping Thread Running...","",3);
        if ( map->keyFrameAdded && !map->LBADone && !map->LCStart )
        {
            // std::vector<vio_slam::KeyFrame *> actKeyF = map->activeKeyFrames;

            std::vector<vio_slam::KeyFrame *> actKeyF;
            KeyFrame* lastKF = map->activeKeyFrames.front();
            actKeyF.reserve(20);
            actKeyF.emplace_back(lastKF);
            lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
            
            {
            triangulateNewPointsR(actKeyF);
            }
            {
            localBAR(actKeyF);
            }

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(2ms);
    }
    std::cout << "LocalMap Thread Exited!" << std::endl;
}

void LocalMapper::beginLocalMappingB()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        // Logging("Local Mapping Thread Running...","",3);
        if ( map->keyFrameAdded && !map->LBADone )
        {
            // std::vector<vio_slam::KeyFrame *> actKeyF = map->activeKeyFrames;

            std::vector<vio_slam::KeyFrame *> actKeyF;
            KeyFrame* lastKF = map->activeKeyFrames.front();
            actKeyF.reserve(20);
            actKeyF.emplace_back(lastKF);
            lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
            {
            // Timer triang("triang");
            std::thread front(&LocalMapper::triangulateNewPointsRB, this, std::ref(zedPtr), std::ref(actKeyF), false);
            std::thread back(&LocalMapper::triangulateNewPointsRB, this, std::ref(zedPtrB), std::ref(actKeyF), true);
            front.join();
            back.join();
            }
            {
            // Timer ba("ba");
            localBARB(actKeyF);
            }

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(2ms);
    }
    std::cout << "LocalMap Thread Exited!" << std::endl;

}

void LocalMapper::beginLoopClosure()
{
    using namespace std::literals::chrono_literals;
    while ( true )
    {
        if ( map->LCStart )
        {
            std::vector<vio_slam::KeyFrame *> activeKF;
            activeKF.reserve(map->LCCandIdx);
            KeyFrame* kFLCCand = map->keyFrames.at(map->LCCandIdx);
            activeKF.emplace_back(kFLCCand);
            kFLCCand->getConnectedKFsLC(map, activeKF);

            loopClosureR(activeKF);

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(100ms);
    }
    std::cout << "LoopClosure Thread Exited!" << std::endl;
}

void LocalMapper::beginLoopClosureB()
{
    using namespace std::literals::chrono_literals;
    while ( true )
    {
        if ( map->LCStart )
        {
            std::vector<vio_slam::KeyFrame *> activeKF;
            activeKF.reserve(map->LCCandIdx);
            KeyFrame* kFLCCand = map->keyFrames.at(map->LCCandIdx);
            activeKF.emplace_back(kFLCCand);
            kFLCCand->getConnectedKFsLC(map, activeKF);

            loopClosureRB(activeKF);

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(100ms);
    }
    std::cout << "LoopClosure Thread Exited!" << std::endl;
}

} // namespace vio_slam