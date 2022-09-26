#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    findFast(image, fastKeys);
}

void FeatureExtractor::findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    const int radius = 3;
    const int contingousPixels = 9;
    const int patternSize = 16;
    const int N = patternSize + contingousPixels;
    const int fastThresh = maxFastThreshold;
    int numbOfFeatures = 0;
    cv::Mat1d im1d = image;
    cv::Mat1b trial = cv::Mat1b::zeros(image.rows,image.cols);
    
    // create threshold mask
    // from -255 to -threshold = 1
    // from -threshold to + threshold = 0
    // from +threshold to 255 = 2
    uchar threshold_mask[512];
    for(int i = -255; i <= 255; i++ )
        threshold_mask[i+255] = (uchar)(i < -fastThresh ? 1 : i > fastThresh ? 2 : 0);


    // although we need 12 contingous we get memory for 25 pixels (repeating the first 9) to account for a corner that starts at pixel 15 the 12 contingous pixels.
    int pixels[N];
    getPixelOffset(pixels, (int)trial.step);
    for (int32_t iRows = edgeThreshold; iRows < image.rows - edgeThreshold; iRows++)
    {
        const uchar* rowPtr = image.ptr(iRows);
        for (int32_t jCols = edgeThreshold; jCols < image.cols - edgeThreshold; jCols++, rowPtr++)
        {
            // highSpeedTest(rowPtr, pixels, fastThresh);
            
            // candidate pixel Intensity
            const int cPInt = rowPtr[0];
            if ((cPInt < 10 || (cPInt > 250)))
                continue;
            // pointer to start of mask and add 255 (to get to the middle) and remove the candidate's pixel intensity.
            // that way the pixels to be checked that are either darker of brighter is easily accessible
            const uchar* tab = &threshold_mask[0] - cPInt + 255;

            // &= bitwise AND, | bitwise OR
            int d = tab[rowPtr[pixels[0]]] | tab[rowPtr[pixels[8]]];

            if( d == 0 )
                continue;

            d &= tab[rowPtr[pixels[2]]] | tab[rowPtr[pixels[10]]];
            d &= tab[rowPtr[pixels[4]]] | tab[rowPtr[pixels[12]]];
            d &= tab[rowPtr[pixels[6]]] | tab[rowPtr[pixels[14]]];

            if( d == 0 )
                continue;

            d &= tab[rowPtr[pixels[1]]] | tab[rowPtr[pixels[9]]];
            d &= tab[rowPtr[pixels[3]]] | tab[rowPtr[pixels[11]]];
            d &= tab[rowPtr[pixels[5]]] | tab[rowPtr[pixels[13]]];
            d &= tab[rowPtr[pixels[7]]] | tab[rowPtr[pixels[15]]];

            bool add = false;

            if (d & 1)
            {
                int thr = cPInt - fastThresh, count = 0;
                for (int k=0;k < N;k ++)
                {
                    const int x = rowPtr[pixels[k]];
                    if (x < thr)
                    {
                        if (++count > contingousPixels)
                        {
                            add =true;
                            break;
                        }
                    }
                    else 
                        count = 0;
                }

            }
            if (d & 2)
            {
                int thr = cPInt + fastThresh, count = 0;
                for (int k=0;k < N;k ++)
                {
                    const int x = rowPtr[pixels[k]];
                    if (x > thr)
                    {
                        if (++count > contingousPixels)
                        {
                            add =true;
                            break;
                        }
                    }
                    else
                        count = 0;
                }

            }

            if (add)
            {
                if (nonMaxSuppression)
                {
                    if (!trial(iRows,jCols))
                    {
                        fastKeys.push_back(cv::KeyPoint(cv::Point2f(jCols, iRows),3.0f));
                        trial.rowRange(cv::Range(iRows - edgeThreshold + 1,iRows + edgeThreshold -1)).colRange(cv::Range(jCols - edgeThreshold + 1,jCols + edgeThreshold -1)) = 1;
                    }
                }
                else
                {
                    fastKeys.push_back(cv::KeyPoint(cv::Point2f(jCols, iRows),3.0f));

                }
            }
            // if (fastKeys.size() > nFeatures)
            //     break;

        }
        // if (fastKeys.size() > nFeatures)
        //     break;
    }
    std::cout << "size " << fastKeys.size() << std::endl;
}

void FeatureExtractor::highSpeedTest(const uchar* rowPtr, int pixels[25], const int fastThresh)
{
    // candidate pixel Intensity
    const int cPInt = rowPtr[0];

    // pixel 1
    int32_t darker = (rowPtr[pixels[0]] + fastThresh) < cPInt ? 1 : 0;
    // pixel 9

}

void FeatureExtractor::getPixelOffset(int pixels[25], int rowStride)
{
    static const int offsets[16][2] =
    {
        {0, -3}, { 1, -3}, { 2, -2}, { 3, -1}, { 3, 0}, { 3,  1}, { 2,  2}, { 1,  3},
        {0,  3}, {-1,  3}, {-2,  2}, {-3,  1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
    };
    int k = 0;
    for(; k < 16; k++ )
        pixels[k] = offsets[k][0] + offsets[k][1] * rowStride;
    for( ; k < 25; k++ )
        pixels[k] = pixels[k - 16];

}

FeatureExtractor::FeatureExtractor(FeatureChoice _choice, const int _nfeatures, const int _edgeThreshold, const int _maxFastThreshold, const int _minFastThreshold, const bool _nonMaxSuppression) : choice(_choice), nFeatures(_nfeatures), edgeThreshold(_edgeThreshold), maxFastThreshold(_maxFastThreshold), minFastThreshold(_minFastThreshold), nonMaxSuppression(_nonMaxSuppression)
{
}

} // namespace vio_slam