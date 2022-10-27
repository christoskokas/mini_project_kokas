#include "FeatureExtractor.h"

namespace vio_slam
{

void FeatureExtractor::findFeatures(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
}

void FeatureExtractor::findORBWithCV(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    Timer orb("ORB CV");
    cv::Ptr<cv::ORB> detector;
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);
    fastKeys.reserve(2000);
    std::vector<std::vector < cv::KeyPoint >> prevImageKeys;
    prevImageKeys.resize(gridCols * gridRows);
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

    const int rowJump = (croppedImage.rows - 2 * fastEdge) / gridRows;
    const int colJump = (croppedImage.cols - 2 * fastEdge) / gridCols;


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

            detector = cv::ORB::create(numberPerCell,1.3f,5,0,0,2,cv::ORB::HARRIS_SCORE,31,maxFastThreshold);
            detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,cv::Mat());

            if (temp.size() < numberPerCell)
            {
                detector = cv::ORB::create(numberPerCell,1.3f,5,0,0,2,cv::ORB::HARRIS_SCORE,31,minFastThreshold);
                detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,cv::Mat());
            }
            if (!temp.empty())
            {
                cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                for ( std::vector < cv::KeyPoint>::iterator it=temp.begin(); it !=temp.end(); it++)
                {
                    
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    (*it).class_id = count;
                    fastKeys.push_back(*it);
                    // getNonMaxSuppression(prevImageKeys[row*gridCols + col],*it);

                }
                // cv::KeyPointsFilter::removeDuplicated(prevImageKeys[row*gridCols + col]);

            }
            count++;
        }
    }
    Logging("keypoint angle",fastKeys[100].angle,1);
    Logging("Keypoint Size Before removal", fastKeys.size(),1);
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);

}

void FeatureExtractor::findORB(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc)
{
    // Timer orb;
    computePyramid(image);

    separateImage(image, fastKeys);
    // separateImageSubPixel(image,fastKeys);

    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::HARRIS_SCORE,patchSize,maxFastThreshold)};
    detector->compute(image,fastKeys,Desc);
}

float FeatureExtractor::computeOrientation(const cv::Mat& image, const cv::Point2f& point)
{
    int m10 {0}, m01{0};
    const int step {(int)image.step};
    const uchar* center = &image.at<uchar> (cvRound(point.y), cvRound(point.x));

    for (int32_t row = 0; row < halfPatchSize; row++)
    {
        int sumIntensities {0};
        for (int32_t col = -halfPatchSize; col < halfPatchSize; col++)
        {
            const int centerP {center[col + row*step]}, centerM {center[col - row*step]};
            sumIntensities += centerP - centerM;
            m10 += col * (centerP + centerM);
        }
        m01 += row * sumIntensities;
    }

    return cv::fastAtan2((float)m01, (float)m10);

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

void FeatureExtractor::findFAST(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc)
{
    findFASTGrids(image,fastKeys);
    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold)};
    detector->compute(image,fastKeys,Desc);

}

void FeatureExtractor::extractORB(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints)
{

    findORB(leftImage,keypoints.left, desc.left);
    findORB(rightImage,keypoints.right, desc.right);
    // cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold)};
    // detector->compute(leftImage, keypoints.left, desc.left);
    // detector->compute(rightImage, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::extractFeaturesPop(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const std::vector<int>& pop)
{

    findFASTGridsPop(leftImage,keypoints.left, pop);
    findFASTGrids(rightImage,keypoints.right);
    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold)};
    detector->compute(leftImage, keypoints.left, desc.left);
    detector->compute(rightImage, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}


void FeatureExtractor::extractFeaturesMask(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const cv::Mat& mask)
{

    findFASTGridsMask(leftImage,keypoints.left, mask);
    findFASTGrids(rightImage,keypoints.right);
    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold)};
    detector->compute(leftImage, keypoints.left, desc.left);
    detector->compute(rightImage, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::extractFeatures(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints)
{

    findFASTGrids(leftImage,keypoints.left);
    findFASTGrids(rightImage,keypoints.right);
    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold)};
    detector->compute(leftImage, keypoints.left, desc.left);
    detector->compute(rightImage, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::findFASTGridsPop(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const std::vector<int>& pop)
{
    fastKeys.reserve(2000);
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/2};
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.
    const int rowJump = (croppedImage.rows - 2 * fastEdge) / gridRows;
    const int colJump = (croppedImage.cols - 2 * fastEdge) / gridCols;

    int count {-1};
    
    for (int32_t row = 0; row < gridRows; row++)
    {
        
        const int imRowStart = row * rowJump;
        const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

        for (int32_t col = 0; col < gridCols; col++)
        {
            count++;

            if (pop[count] > mnNKey)
                continue;

            const int imColStart = col * colJump;
            const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

            std::vector < cv::KeyPoint > temp;

            cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.size() < mnNKey)
            {
                cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                // cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                for (it=temp.begin(); it != end; it++)
                {
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    (*it).class_id = count;
                    fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
            }
        }
    }
    // std::vector < cv::KeyPoint>::iterator it;
    // std::vector < cv::KeyPoint>::const_iterator end(allKeys.end());
    // for (it=allKeys.begin(); it != end; it++)
    // {

    //     // (*it).angle = {computeOrientation(croppedImage, cv::Point2f((*it).pt.x,(*it).pt.y))};

    //     // (*it).angle = 0;

    //     fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
    // }
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::findFASTGridsMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask)
{
    fastKeys.reserve(2000);
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);
    cv::Mat croppedMask = mask.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/2};
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(maxFastThreshold);
    const int rowJump = (croppedImage.rows - 2 * fastEdge) / gridRows;
    const int colJump = (croppedImage.cols - 2 * fastEdge) / gridCols;

    int count {-1};
    
    for (int32_t row = 0; row < gridRows; row++)
    {
        
        const int imRowStart = row * rowJump;
        const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

        for (int32_t col = 0; col < gridCols; col++)
        {
            count++;

            const int imColStart = col * colJump;
            const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

            std::vector < cv::KeyPoint > temp;

            detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)), temp,croppedMask.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)));

            // cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.size() < mnNKey)
            {
                detector->setThreshold(minFastThreshold);
                detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)), temp,croppedMask.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)));
                detector->setThreshold(maxFastThreshold);


                // cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                // cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                for (it=temp.begin(); it != end; it++)
                {
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    (*it).class_id = count;
                    fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
            }
        }
    }
    // std::vector < cv::KeyPoint>::iterator it;
    // std::vector < cv::KeyPoint>::const_iterator end(allKeys.end());
    // for (it=allKeys.begin(); it != end; it++)
    // {

    //     // (*it).angle = {computeOrientation(croppedImage, cv::Point2f((*it).pt.x,(*it).pt.y))};

    //     // (*it).angle = 0;

    //     fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
    // }
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::findFASTGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    fastKeys.reserve(2000);
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/2};
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

    const int rowJump = (croppedImage.rows - 2 * fastEdge) / gridRows;
    const int colJump = (croppedImage.cols - 2 * fastEdge) / gridCols;

    int count {-1};
    
    for (int32_t row = 0; row < gridRows; row++)
    {
        
        const int imRowStart = row * rowJump;
        const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

        for (int32_t col = 0; col < gridCols; col++)
        {
            count++;

            const int imColStart = col * colJump;
            const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

            std::vector < cv::KeyPoint > temp;

            cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.size() < mnNKey)
            {
                cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                // cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                for (it=temp.begin(); it != end; it++)
                {
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    (*it).class_id = count;
                    fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
            }
        }
    }
    // std::vector < cv::KeyPoint>::iterator it;
    // std::vector < cv::KeyPoint>::const_iterator end(allKeys.end());
    // for (it=allKeys.begin(); it != end; it++)
    // {

    //     // (*it).angle = {computeOrientation(croppedImage, cv::Point2f((*it).pt.x,(*it).pt.y))};

    //     // (*it).angle = 0;

    //     fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
    // }
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::separateImage(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    fastKeys.reserve(2000);
    std::vector <float> pyrDifRow, pyrDifCol;
    pyrDifRow.resize(nLevels);
    pyrDifCol.resize(nLevels);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    std::vector<std::vector < cv::KeyPoint >> prevImageKeys;
    prevImageKeys.resize(gridCols * gridRows);
    for (size_t level = 0; level < nLevels; level++)
    {
        // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

        const int rowJump = (imagePyramid[level].rows - 2 * fastEdge) / gridRows;
        const int colJump = (imagePyramid[level].cols - 2 * fastEdge) / gridCols;

        const int numbPerLevelPerCell = numberPerCell/nLevels;

        pyrDifRow[level] = imagePyramid[0].rows/(float)imagePyramid[level].rows;
        pyrDifCol[level] = imagePyramid[0].cols/(float)imagePyramid[level].cols;

        int count {-1};
        
        for (int32_t row = 0; row < gridRows; row++)
        {
            
            const int imRowStart = row * rowJump;
            const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

            for (int32_t col = 0; col < gridCols; col++)
            {
                count++;

                const int imColStart = col * colJump;
                const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

                std::vector < cv::KeyPoint > temp;

                cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

                if (temp.size() < numbPerLevelPerCell)
                {
                    cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
                }
                if (!temp.empty())
                {
                    cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                    std::vector < cv::KeyPoint>::iterator it, end(temp.end());
                    for (it=temp.begin(); it != end; it++)
                    {
                        
                        (*it).pt.x = ((*it).pt.x + imColStart) * pyrDifCol[level] + edgeWFast;
                        (*it).pt.y = ((*it).pt.y + imRowStart) * pyrDifRow[level] + edgeWFast;
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
            }
        }
    }
    // Timer angle("angle timer");
    for (size_t level = 0; level < gridCols * gridRows; level++)
    {
        // cv::KeyPointsFilter::retainBest(prevImageKeys[i],numberPerCell);
        if (prevImageKeys[level].empty())
            continue;
        std::vector < cv::KeyPoint>::iterator it, end(prevImageKeys[level].end());
        for (it=prevImageKeys[level].begin(); it != end; it++)
        {
            // if ((*it).class_id < 0)
            //     continue;
            const int oct {(*it).octave};
            (*it).angle = {computeOrientation(imagePyramid[oct], cv::Point2f(((*it).pt.x - edgeWFast)/pyrDifCol[oct],((*it).pt.y - edgeWFast)/pyrDifCol[oct]))};


            const float size {patchSize * scalePyramid[(*it).octave]};
            fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
        }
    }
    Logging("keypoint angle",fastKeys[100].angle,1);
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::separateImageSubPixel(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    fastKeys.reserve(2000);
    std::vector <float> pyrDifRow, pyrDifCol;
    pyrDifRow.resize(nLevels);
    pyrDifCol.resize(nLevels);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    std::vector<std::vector < cv::KeyPoint >> prevImageKeys;
    prevImageKeys.resize(gridCols * gridRows);
    for (size_t level = 0; level < nLevels; level++)
    {
        // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

        const int rowJump = (imagePyramid[level].rows - 2 * fastEdge) / gridRows;
        const int colJump = (imagePyramid[level].cols - 2 * fastEdge) / gridCols;

        pyrDifRow[level] = imagePyramid[0].rows/(float)imagePyramid[level].rows;
        pyrDifCol[level] = imagePyramid[0].cols/(float)imagePyramid[level].cols;

        int count {-1};
        
        for (int32_t row = 0; row < gridRows; row++)
        {
            
            const int imRowStart = row * rowJump;
            const int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;

            for (int32_t col = 0; col < gridCols; col++)
            {
                count++;

                const int imColStart = col * colJump;
                const int imColEnd = (col + 1) * colJump + 2 * fastEdge;

                std::vector < cv::KeyPoint > temp;

                cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

                if (temp.size() < numberPerCell/nLevels)
                {
                    cv::FAST(imagePyramid[level].colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
                }
                if (!temp.empty())
                {
                    cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                    std::vector < cv::KeyPoint>::iterator it, end(temp.end());
                    for (it=temp.begin(); it != end; it++)
                    {
                        
                        (*it).pt.x = ((*it).pt.x + imColStart) * pyrDifCol[level] + edgeWFast;
                        (*it).pt.y = ((*it).pt.y + imRowStart) * pyrDifRow[level] + edgeWFast;
                        (*it).octave = level;

                        // (*it).class_id = count;

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
            }
        }
    }
    // Timer angle("angle timer");
    for (size_t level = 0; level < gridCols * gridRows; level++)
    {
        // cv::KeyPointsFilter::retainBest(prevImageKeys[i],numberPerCell);
        if (prevImageKeys[level].empty())
            continue;
        std::vector < cv::Point2f > points;
        points.reserve(prevImageKeys[level].size());
        for ( std::vector < cv::KeyPoint>::iterator it=prevImageKeys[level].begin(); it !=prevImageKeys[level].end(); it++)
        {
            points.push_back((*it).pt);
        }
        cv::cornerSubPix(image,points,cv::Size(3,3),cv::Size(1,1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,10,0.001));
        std::vector < cv::Point2f>::iterator it2=points.begin();
        for ( std::vector < cv::KeyPoint>::iterator it=prevImageKeys[level].begin(); it !=prevImageKeys[level].end(); it++, it2++)
        {
            // if ((*it).class_id < 0)
            //     continue;
            const int oct {(*it).octave};
            (*it).angle = {computeOrientation(imagePyramid[oct], cv::Point2f(((*it2).x - edgeWFast)/pyrDifCol[oct],((*it2).y - edgeWFast)/pyrDifCol[oct]))};


            const float size {patchSize * scalePyramid[(*it).octave]};
            fastKeys.emplace_back(cv::Point2f((*it2).x,(*it2).y), size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
        }
    }
    Logging("keypoint angle",fastKeys[100].angle,1);
    cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::getNonMaxSuppression(std::vector < cv::KeyPoint >& prevImageKeys, cv::KeyPoint& it)
{
    bool found = false;
    std::vector < cv::KeyPoint>::iterator it2, end(prevImageKeys.end());
    for (it2=prevImageKeys.begin(); it2 != end; it2++)
    {
        if ( checkDistance(*it2, it, 5) )
        {
            found = true;
            if ((it).response > (*it2).response)
            {
                (*it2).pt.x = (it).pt.x;
                (*it2).pt.y = (it).pt.y;
                (*it2).response = (it).response;
                (*it2).octave = (it).octave;
                // (*it2).class_id += 1;
                
            }
            break;
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

void SubPixelPoints::clone(SubPixelPoints& points)
{
    left = points.left;
    right = points.right;
    depth = points.depth;
    useable = points.useable;
}

void SubPixelPoints::add(SubPixelPoints& points)
{
    const size_t size {left.size() + points.left.size()};

    left.reserve(size);
    right.reserve(size);
    depth.reserve(size);
    useable.reserve(size);

    const size_t end {points.left.size()};

    for (size_t i = 0; i < end; i++)
    {
        left.emplace_back(points.left[i]);
        right.emplace_back(points.right[i]);
        depth.emplace_back(points.depth[i]);
        useable.emplace_back(points.useable[i]);
    }
}

void SubPixelPoints::addLeft(SubPixelPoints& points)
{
    const size_t size {left.size() + points.left.size()};

    left.reserve(size);
    depth.reserve(size);
    useable.reserve(size);

    const size_t end {points.left.size()};

    for (size_t i = 0; i < end; i++)
    {
        left.emplace_back(points.left[i]);
        depth.emplace_back(points.depth[i]);
        useable.emplace_back(points.useable[i]);
    }
}

void SubPixelPoints::clear()
{
    left.clear();
    right.clear();
    useable.clear();
    depth.clear();
    points2D.clear();
}

int FeatureExtractor::getGridRows()
{
    return gridRows;
}

int FeatureExtractor::getGridCols()
{
    return gridCols;
}


} // namespace vio_slam