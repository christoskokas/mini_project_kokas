#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
}

void FeatureExtractor::findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
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
        // std::cout << "size : " << imagePyramid[i].cols << " " << imagePyramid[i].rows << std::endl;
        // std::cout << "cvsize : " << width << " " << height << std::endl;
    }
}

void FeatureExtractor::separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    // ProcessTime loop("lloop");
    fastKeys.reserve(2000);
    const int fastEdge = 3;
    for (size_t i = 0; i < nLevels; i++)
    {
        // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.
        // const int numberPerCell = 2*nFeatures*rows*cols/(image.cols*image.rows);
        const int rowJump = (imagePyramid[i].rows - 2 * fastEdge) / gridRows;
        const int colJump = (imagePyramid[i].cols - 2 * fastEdge) / gridCols;

        const float pyramidDifRow = imagePyramid[0].rows/(float)imagePyramid[i].rows;
        const float pyramidDifCol = imagePyramid[0].cols/(float)imagePyramid[i].cols;
        // cv::Mat1b edgedImage = image.colRange(cv::Range(edgeThreshold - fastEdge,image.cols - edgeThreshold + fastEdge)).rowRange(cv::Range(edgeThreshold - fastEdge,image.rows - edgeThreshold + fastEdge));
        // const int colEnd = imagePyramid[i].cols - edgeThreshold - fastEdge;
        // const int rowEnd = imagePyramid[i].rows - edgeThreshold - fastEdge;
        int count {0};
        for (int32_t row = 0; row < gridRows; row++)
        {
            
            const int imRowStart = row * rowJump;
            const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

            const uchar* rowPtr = imagePyramid[i].ptr<uchar>(row);

            for (int32_t col = 0; col < gridCols; col++)
            {
                // cv::Mat1b patch = image.colRange(cv::Range(col, col + cols + 2 * fastEdge)).rowRange(cv::Range(row, row + rows + 2*fastEdge));
                {
                    const int cPInt = rowPtr[col];
                    if (cPInt < 10 || cPInt > 250)
                        continue;
                }
                const int imColStart = col * colJump;
                const int imColEnd = (col + 1) * colJump + 2 * fastEdge;
                // std::cout << "row : " << imRowStart << " row + 1 : " << imRowEnd << " col : " << imColStart << " col + 1 : " << imColEnd << " rowJump " << rowJump << " colJump " << colJump <<std::endl;
                // if (imColEnd > imagePyramid[i].cols)
                //     continue;
                // // imcolend = imagePyramid[i].cols
                // if (imRowEnd > imagePyramid[i].rows)
                //     continue;

                std::vector < cv::KeyPoint > temp;

                cv::FAST(imagePyramid[i].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);
                // cv::imshow("yes", imagePyramid[i].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)));
                // cv::waitKey(0);
                if (temp.empty())
                    cv::FAST(imagePyramid[i].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
                if (!temp.empty())
                {
                    // const float rowOffset = imRowStart * (imagePyramid[0].rows/(float)imagePyramid[i].rows) + edgeThreshold - fastEdge;
                    // const float colOffset = imColStart * (imagePyramid[0].cols/(float)imagePyramid[i].cols) + edgeThreshold - fastEdge;

                    // std::cout << "col off " << colOffset << " row off " << rowOffset << std::endl;
                    // std::cout << "col " << imColStart << " row " << imRowStart << std::endl;
                    // std::cout << "divide cols " << imagePyramid[0].cols/(float)imagePyramid[i].cols << " divide rows " << imagePyramid[0].rows/(float)imagePyramid[i].rows << std::endl;
                    cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                    for ( std::vector < cv::KeyPoint>::iterator it=temp.begin(); it !=temp.end(); it++)
                    {
                        (*it).pt.x = ((*it).pt.x + imColStart) * pyramidDifCol;
                        (*it).pt.y = ((*it).pt.y + imRowStart) * pyramidDifRow;
                        (*it).octave = i;
                        (*it).class_id = count;
                        // fastKeys.push_back(*it);
                        fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y),(*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                    }
                }
                count++;
            }
        }
    }
    // std::cout << " size : " << fastKeys.size() << std::endl;
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    const int edgeWFast = edgeThreshold - fastEdge;
    for ( std::vector < cv::KeyPoint>::iterator it=fastKeys.begin(); it !=fastKeys.end(); it++)
    {
        (*it).pt.x += edgeWFast;
        (*it).pt.y += edgeWFast;
    }
    // loop.totalTime();
    // std::cout << "after size : " << fastKeys.size() << " nfeat : " << nFeatures << std::endl;
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
    ProcessTime loop("loop");
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
    loop.totalTime();
    std::cout << "size " << numbOfFeatures << std::endl;
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

FeatureExtractor::FeatureExtractor(FeatureChoice _choice, const int _nfeatures, const int _nLevels, const float _imScale, const int _edgeThreshold, const int _maxFastThreshold, const int _minFastThreshold, const bool _nonMaxSuppression) : choice(_choice), nFeatures(_nfeatures), nLevels(_nLevels), imScale(_imScale), edgeThreshold(_edgeThreshold), maxFastThreshold(_maxFastThreshold), minFastThreshold(_minFastThreshold), nonMaxSuppression(_nonMaxSuppression)
{
    // Different Implementation, instead of downsampling the image 8 times (making it smaller)
    // because this project consists of 2 cameras 1 facing backwards and 1 facing forwards, the pyramid will contain both upsampled and downsampled images.
    // 5 scale downsampled 2 scales upsampled
    imagePyramid.resize(nLevels);
    scalePyramid.resize(nLevels);
    scaleInvPyramid.resize(nLevels);
    scalePyramid[0] = 1.0f;
    scaleInvPyramid[0] = 1.0f;
    for (int32_t i = 0; i < nLevels - 1; i++)
    {
        scalePyramid[i + 1] = scalePyramid[i] * imScale;
        scaleInvPyramid[i + 1] = scaleInvPyramid[i] / imScale;

    }
    
    
}

} // namespace vio_slam