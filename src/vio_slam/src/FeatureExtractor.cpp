#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
}

void FeatureExtractor::findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    const int rows{20},cols{20};
    computePyramid(image);
    separateImage(image, fastKeys, rows, cols);
}

void FeatureExtractor::computePyramid(cv::Mat& image)
{

    for (int32_t i = 0; i < nLevels; i++)
    {
        cv::Size cz(cvRound((float)image.cols*scalePyramid[i]), cvRound((float)image.rows*scalePyramid[i]));
        cv::resize(image,imagePyramid[i],cz, 0, 0, cv::INTER_LINEAR);
        std::cout << "size : " << imagePyramid[i].cols << " " << imagePyramid[i].rows << std::endl;
    }
}

void FeatureExtractor::separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const int rows, const int cols)
{
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.
    // const int numberPerCell = 2*nFeatures*rows*cols/(image.cols*image.rows);
    const int fastEdge = 3;
    const int rowJump = (image.rows - 2 * edgeThreshold) / gridRows;
    const int colJump = (image.cols - 2 * edgeThreshold) / gridCols;
    // cv::Mat1b edgedImage = image.colRange(cv::Range(edgeThreshold - fastEdge,image.cols - edgeThreshold + fastEdge)).rowRange(cv::Range(edgeThreshold - fastEdge,image.rows - edgeThreshold + fastEdge));
    ProcessTime loop("lloop");
    const int colEnd = image.cols - edgeThreshold - fastEdge;
    const int rowEnd = image.rows - edgeThreshold - fastEdge;
    fastKeys.reserve(2000);
    int count {0};
    for (int32_t row = edgeThreshold - fastEdge; row < rowEnd;row += rowJump)
    {
        for (int32_t col = edgeThreshold - fastEdge; col < colEnd; col += colJump)
        {
            // cv::Mat1b patch = image.colRange(cv::Range(col, col + cols + 2 * fastEdge)).rowRange(cv::Range(row, row + rows + 2*fastEdge));
            // std::cout << "row : " << row << " row + 1 : " << row + rows + 2*fastEdge << " col : " << col << " col + 1 : " << col + cols + 2*fastEdge <<std::endl;
            const int imColStart = col;
            const int imRowStart = row;
            const int imColEnd = col + colJump + 2 * fastEdge;
            const int imRowEnd = row + rowJump + 2 * fastEdge;


            std::vector < cv::KeyPoint > temp;

            cv::FAST(image.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.empty())
                cv::FAST(image.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            if (!temp.empty())
            {
                for ( std::vector < cv::KeyPoint>::iterator it=temp.begin(); it !=temp.end(); it++)
                {
                    (*it).pt.x += col;
                    (*it).pt.y += row;
                    (*it).class_id += count;
                    fastKeys.push_back(*it);
                    // fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y),(*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
            }
            count ++;
            // if (fastKeys.size() > 959)
            //     break;
        }
        // if (fastKeys.size() > 959)
        //     break;
    }
    loop.totalTime();
    std::cout << "count : " << count << " size : " << fastKeys.size() << std::endl;
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    std::cout << "after size : " << fastKeys.size() << " nfeat : " << nFeatures << std::endl;
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
    scalePyramid.reserve(nLevels);
    for (int32_t i = - nLevels / 2 - 1; i < nLevels / 2 - 1; i++)
        scalePyramid.emplace_back((float)pow(imScale,i));
    
    
    
}

} // namespace vio_slam