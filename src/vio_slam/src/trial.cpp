#include "trial.h"

void RobustMatcher::findFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    cv::FAST(image, keypoints,15,true);
}

void RobustMatcher::findFeaturesAdaptive(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    int rows = 10;
    int cols = 10;
    cv::Size imgSize(image.cols/cols,image.rows/rows);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            cv::Mat patch = image.rowRange(row*imgSize.height, (row+1)*imgSize.height).colRange(col*imgSize.width, (col+1)*imgSize.width);
            std::vector< cv::KeyPoint > tempkeys;
            cv::FAST(patch,tempkeys,15,true);
            if(tempkeys.empty())
            {
                cv::FAST(patch,tempkeys,10,true);
            }
            if(!tempkeys.empty())
            {
                for (auto key:tempkeys)
                {
                    key.pt.x +=col*imgSize.width;
                    key.pt.y +=col*imgSize.height;
                    keypoints.push_back(key);
                }
            }
        }
        
    }
    
}
