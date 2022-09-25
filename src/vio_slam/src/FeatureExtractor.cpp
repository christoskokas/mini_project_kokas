#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    std::cout << edgeThreshold << std::endl;
    cv::Size imgSize(image.cols,image.rows);
    if (nonMaxSuppression)
    {
        findFast(image, fastKeys);
        // int x {-3},y{0};
        // while(true)
        // {
        //     if (x == -3)
        //     {
        //         if (y < 1)
        //         {
        //             y ++;
        //             continue;
        //         }
        //         else
        //         {
        //             y ++;
        //             x ++;
        //             continue;
        //         }
        //     }
        //     if (x == 3)
        //     {
        //         if (y > 1)
        //         {
        //             y ++;
        //             continue;
        //         }
        //         else
        //         {
        //             y ++;
        //             x ++;
        //             continue;
        //         }
        //     }
        // }

    }
}

void FeatureExtractor::findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    std::vector <bool> xkeys(image.rows),ykeys(image.cols);
    const int radius = 3;
    const int contingousPixels = 12;
    for (size_t jCols = edgeThreshold; jCols < image.cols - edgeThreshold; jCols++)
    {
        for (size_t iRows = edgeThreshold; iRows < image.rows - edgeThreshold; iRows++)
        {
            bool nonMaxCheck = false;
            for (size_t i = iRows - radius; i < iRows + radius; i++)
            {
                for (size_t j = jCols - radius; j < jCols + radius; j++)
                {
                    if (xkeys[i] && ykeys[j])
                    {
                        nonMaxCheck = true;
                        break;   
                    }
                }
                if (nonMaxCheck)
                    break;
                
            }
            if (nonMaxCheck)
                break;

            // candidate pixel Intensity
            const int cPInt = (int)image.at<uchar>(iRows,jCols);

            const int p1 = (int)image.at<uchar>(iRows - radius,jCols);
            const int p9 = (int)image.at<uchar>(iRows + radius,jCols);
            if ((p1 > cPInt - maxFastThreshold) && (p1 < cPInt + maxFastThreshold) && (p9 > cPInt - maxFastThreshold) && (p9 < cPInt + maxFastThreshold))
                continue;
                
            const int p5 = (int)image.at<uchar>(iRows,jCols + radius);
            const int p13 = (int)image.at<uchar>(iRows,jCols - radius);
            if ((p5 > cPInt - maxFastThreshold) && (p5 < cPInt + maxFastThreshold) && (p13 > cPInt - maxFastThreshold) && (p13 < cPInt + maxFastThreshold))
                continue;

            std::vector < int > darker;
            std::vector < int > brighter;

            if (p1 < cPInt - maxFastThreshold)
                darker.push_back(1);
            else if (p1 > cPInt + maxFastThreshold)
                brighter.push_back(1);
                
            if (p5 < cPInt - maxFastThreshold)
                darker.push_back(5);
            else if (p5 > cPInt + maxFastThreshold)
                brighter.push_back(5);

            if (p9 < cPInt - maxFastThreshold)
                darker.push_back(9);
            else if (p9 > cPInt + maxFastThreshold)
                brighter.push_back(9);

            if (p13 < cPInt - maxFastThreshold)
                darker.push_back(13);
            else if (p13 > cPInt + maxFastThreshold)
                brighter.push_back(13);

            if (brighter.size() != 3 && darker.size() != 3)
                continue;
            else
            {
                fastKeys.push_back(cv::KeyPoint(cv::Point2f(iRows, jCols),3.0f));
                xkeys[iRows] = true;
                ykeys[jCols] = true;
            }

        }
        
    }
}

FeatureExtractor::FeatureExtractor(FeatureChoice _choice, const int _nfeatures, const int _edgeThreshold, const int _maxFastThreshold, const int _minFastThreshold, const bool _nonMaxSuppression) : choice(_choice), nFeatures(_nfeatures), edgeThreshold(_edgeThreshold), maxFastThreshold(_maxFastThreshold), minFastThreshold(_minFastThreshold), nonMaxSuppression(_nonMaxSuppression)
{
}

} // namespace vio_slam