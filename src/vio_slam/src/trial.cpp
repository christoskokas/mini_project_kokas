#include "trial.h"

void RobustMatcher::findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    cv::FAST(image, keypoints,15,true);
}