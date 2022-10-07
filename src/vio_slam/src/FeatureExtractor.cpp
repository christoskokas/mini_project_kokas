#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
}

void FeatureExtractor::findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    Timer orb;
    computePyramid(image);
    separateImage(image, fastKeys);
}

void FeatureExtractor::computePyramid(cv::Mat& image)
{
    const int fastEdge = 3;
    imagePyramid[0] = image.colRange(edgeThreshold - fastEdge,image.cols - edgeThreshold + fastEdge).rowRange(edgeThreshold - fastEdge, image.rows - edgeThreshold + fastEdge);
    for (int32_t i = 0; i < nLevels - 1; i++)
    {
        const int mult = 10;
        const int width = (((int)(imagePyramid[i].cols / imScale) + mult/2) / mult) * mult + 2 * fastEdge;
        const int height = (((int)(imagePyramid[i].rows / imScale) + mult/2) / mult) * mult + 2 * fastEdge;

        cv::Size cz(width,height);

        cv::resize(imagePyramid[i],imagePyramid[i + 1],cz, 0, 0, cv::INTER_LINEAR);
        const float imageScaleCols = imagePyramid[i].cols/(float)imagePyramid[i + 1].cols;
        const float imageScaleRows = imagePyramid[i].rows/(float)imagePyramid[i + 1].rows;
        scalePyramid[i + 1] = scalePyramid[i] * ((imageScaleCols + imageScaleRows)/2);
        scaleInvPyramid[i + 1] = scaleInvPyramid[i] / ((imageScaleCols + imageScaleRows)/2);
    }
}

void FeatureExtractor::separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    fastKeys.reserve(2000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    std::vector<std::vector < cv::KeyPoint >> prevImageKeys;
    prevImageKeys.resize(gridCols * gridRows);
    for (size_t level = 0; level < nLevels; level++)
    {
        // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

        const int rowJump = (imagePyramid[level].rows - 2 * fastEdge) / gridRows;
        const int colJump = (imagePyramid[level].cols - 2 * fastEdge) / gridCols;

        const float pyramidDifRow = imagePyramid[0].rows/(float)imagePyramid[level].rows;
        const float pyramidDifCol = imagePyramid[0].cols/(float)imagePyramid[level].cols;

        int count {0};
        
        for (int32_t row = 0; row < gridRows; row++)
        {
            
            const int imRowStart = row * rowJump;
            const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

            for (int32_t col = 0; col < gridCols; col++)
            {

                const int imColStart = col * colJump;
                const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

                std::vector < cv::KeyPoint > temp;

                cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

                if (temp.empty())
                {
                    cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
                }
                if (!temp.empty())
                {
                    cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                    for ( std::vector < cv::KeyPoint>::iterator it=temp.begin(); it !=temp.end(); it++)
                    {
                        // {
                        //     // remove keypoints that are too bright or too dark
                        //     const int cPInt = imagePyramid[i].at<uchar>((*it).pt.x,(*it).pt.y);
                        //     if ( cPInt < 10 || cPInt > 250 )
                        //         continue;
                        // }

                        (*it).pt.x = ((*it).pt.x + imColStart) * pyramidDifCol + edgeWFast;
                        (*it).pt.y = ((*it).pt.y + imRowStart) * pyramidDifRow + edgeWFast;
                        (*it).octave = level;
                        (*it).class_id = count;

                        if (level == 0)
                            continue;
                        
                        getNonMaxSuppression(prevImageKeys[row*gridCols + col],*it);

                    }
                    if (level == 0)
                    {
                        prevImageKeys[row*gridCols + col].reserve(temp.size() + 100);
                        prevImageKeys.push_back(temp);
                        continue;
                    }
                    cv::KeyPointsFilter::removeDuplicated(prevImageKeys[row*gridCols + col]);

                }
                count++;
            }
        }
    }
    for (size_t level = 0; level < gridCols * gridRows; level++)
    {
        // cv::KeyPointsFilter::retainBest(prevImageKeys[i],numberPerCell);
        const int patchSizeScaled = patchSize * scalePyramid[level];
        for ( std::vector < cv::KeyPoint>::iterator it=prevImageKeys[level].begin(); it !=prevImageKeys[level].end(); it++)
        {
            // compute angle

            fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y),patchSizeScaled,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
        }
    }
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::getNonMaxSuppression(std::vector < cv::KeyPoint >& prevImageKeys, cv::KeyPoint& it)
{
    bool found = false;
    for ( std::vector < cv::KeyPoint>::iterator it2=prevImageKeys.begin(); it2 !=prevImageKeys.end(); it2++)
    {
        if ( checkDistance(*it2, it, 3) )
        {
            if ((it).response > (*it2).response)
            {
                (*it2).pt.x = (it).pt.x;
                (*it2).pt.y = (it).pt.y;
                (*it2).response = (it).response;
                (*it2).octave = (it).octave;
            }
            else
            {
                found = true;
                break;
            }
        }
    }
    if (!found)
        prevImageKeys.push_back(it);
}

bool FeatureExtractor::checkDistance(cv::KeyPoint& first, cv::KeyPoint& second, int distance)
{
    return (abs(first.pt.x - second.pt.x) < distance && abs(first.pt.y - second.pt.y) < distance );
}

void FeatureExtractor::findFast(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    const int radius = 3;
    const int contingousPixels = 9;
    const int patternSize = 16;
    const int N = patternSize + contingousPixels;
    int fastThresh = maxFastThreshold;
    int numbOfFeatures = 0;
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
    int score {0};
    for (int32_t iRows = edgeThreshold; iRows < image.rows - edgeThreshold; iRows++)
    {
        const uchar* rowPtr = image.ptr<uchar>(iRows);
        uchar* trialPtr = trial.ptr<uchar>(iRows);

        for (int32_t jCols = edgeThreshold; jCols < image.cols - edgeThreshold; jCols++, rowPtr++, trialPtr++)
        {
            // highSpeedTest(rowPtr, pixels, fastThresh);
            
            // candidate pixel Intensity
            score = checkIntensities(rowPtr,threshold_mask,pixels, fastThresh);
            
            if ( score != 0 )
            {
                int lel = (score/9);
                if (nonMaxSuppression)
                {
                    if ( trialPtr[jCols - 1] == 0 && trialPtr[jCols - (int)trial.step] ==0)
                    {
                        numbOfFeatures++;
                        continue;
                    }
                    if (trialPtr[jCols] < trialPtr[jCols - 1])
                        trialPtr[jCols] = 0;
                    else
                        trialPtr[jCols - 1] = 0;
                    if (iRows == 3)
                        continue;
                    if (trialPtr[jCols] < trialPtr[jCols - (int)trial.step])
                        trialPtr[jCols] = 0;
                    else
                        trialPtr[jCols - (int)trial.step] = 0;
                }
                else
                    numbOfFeatures ++;
                if (!nonMaxSuppression || (lel > trialPtr[jCols - 1] && lel > trialPtr[jCols - 2]))
                {
                }
            }

            if (numbOfFeatures > nFeatures)
                break;

        }
        if (numbOfFeatures > nFeatures)
            break;
    }
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

int FeatureExtractor::checkIntensities(const uchar* rowPtr, uchar threshold_mask[512], int pixels[25], int thresh)
{
    int fastThresh = thresh;
    const int cPInt = rowPtr[0];
    if ((cPInt < 10 || (cPInt > 250)))
        return 0;
    // pointer to start of mask and add 255 (to get to the middle) and remove the candidate's pixel intensity.
    // that way the pixels to be checked that are either darker of brighter is easily accessible
    const uchar* tab = &threshold_mask[0] - cPInt + 255;

    // &= bitwise AND, | bitwise OR
    int d = tab[rowPtr[pixels[0]]] | tab[rowPtr[pixels[8]]];

    if( d == 0 )
        return 0;

    d &= tab[rowPtr[pixels[2]]] | tab[rowPtr[pixels[10]]];
    d &= tab[rowPtr[pixels[4]]] | tab[rowPtr[pixels[12]]];
    d &= tab[rowPtr[pixels[6]]] | tab[rowPtr[pixels[14]]];

    if( d == 0 )
        return 0;

    d &= tab[rowPtr[pixels[1]]] | tab[rowPtr[pixels[9]]];
    d &= tab[rowPtr[pixels[3]]] | tab[rowPtr[pixels[11]]];
    d &= tab[rowPtr[pixels[5]]] | tab[rowPtr[pixels[13]]];
    d &= tab[rowPtr[pixels[7]]] | tab[rowPtr[pixels[15]]];

    int score {0};

    if (d & 1)
    {
        int thr = cPInt - fastThresh, count = 0;
        for (int k=0;k < 25;k ++)
        {
            const int x = rowPtr[pixels[k]];
            if (x < thr)
            {
                if (++count > 8)
                {
                    for (size_t i = k - 9; i < k; i++)
                        score += cPInt - rowPtr[pixels[i]];
                    return score;
                }
            }
            else 
                count = 0;
        }

    }
    if (d & 2)
    {
        int thr = cPInt + fastThresh, count = 0;
        for (int k=0;k < 25;k ++)
        {
            const int x = rowPtr[pixels[k]];
            if (x > thr)
            {
                if (++count > 8)
                {
                    for (size_t i = k - 9; i < k; i++)
                        score += rowPtr[pixels[i]] - cPInt;
                    return score;
                }
            }
            else
                count = 0;
        }
    }
    return score;
}

FeatureExtractor::FeatureExtractor(FeatureChoice _choice, const int _nfeatures, const int _nLevels, const float _imScale, const int _edgeThreshold, const int _patchSize, const int _maxFastThreshold, const int _minFastThreshold, const bool _nonMaxSuppression) : choice(_choice), nFeatures(_nfeatures), nLevels(_nLevels), imScale(_imScale), edgeThreshold(_edgeThreshold), patchSize(_patchSize), maxFastThreshold(_maxFastThreshold), minFastThreshold(_minFastThreshold), nonMaxSuppression(_nonMaxSuppression)
{
    // Different Implementation, instead of downsampling the image 8 times (making it smaller)
    // because this project consists of 2 cameras 1 facing backwards and 1 facing forwards, the pyramid will contain both upsampled and downsampled images.
    // 5 scale downsampled 2 scales upsampled
    imagePyramid.resize(nLevels);
    scalePyramid.resize(nLevels);
    scaleInvPyramid.resize(nLevels);
    scalePyramid[0] = 1.0f;
    scaleInvPyramid[0] = 1.0f;
    
    
}

} // namespace vio_slam