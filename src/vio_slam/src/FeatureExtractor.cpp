#include "FeatureExtractor.h"

namespace vio_slam
{

static int bit_pattern_31_[256*4] =
{
        8,-3, 9,5/*mean (0), correlation (0)*/,
        4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
        -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
        7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
        2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
        1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
        -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
        -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
        -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
        10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
        -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
        -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
        7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
        -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
        -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
        -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
        12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
        -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
        -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
        11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
        4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
        5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
        3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
        -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
        -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
        -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
        -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
        -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
        -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
        5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
        5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
        1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
        9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
        4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
        2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
        -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
        -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
        4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
        0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
        -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
        -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
        -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
        8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
        0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
        7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
        -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
        10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
        -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
        10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
        -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
        -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
        3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
        5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
        -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
        3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
        2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
        -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
        -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
        -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
        -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
        6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
        -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
        -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
        -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
        3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
        -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
        -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
        2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
        -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
        -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
        5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
        -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
        -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
        -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
        10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
        7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
        -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
        -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
        7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
        -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
        -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
        -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
        7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
        -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
        1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
        2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
        -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
        -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
        7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
        1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
        9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
        -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
        -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
        7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
        12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
        6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
        5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
        2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
        3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
        2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
        9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
        -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
        -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
        1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
        6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
        2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
        6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
        3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
        7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
        -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
        -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
        -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
        -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
        8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
        4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
        -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
        4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
        -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
        -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
        7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
        -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
        -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
        8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
        -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
        1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
        7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
        -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
        11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
        -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
        3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
        5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
        0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
        -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
        0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
        -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
        5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
        3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
        -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
        -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
        -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
        6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
        -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
        -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
        1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
        4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
        -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
        2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
        -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
        4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
        -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
        -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
        7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
        4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
        -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
        7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
        7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
        -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
        -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
        -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
        2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
        10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
        -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
        8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
        2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
        -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
        -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
        -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
        5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
        -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
        -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
        -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
        -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
        -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
        2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
        -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
        -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
        -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
        -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
        6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
        -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
        11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
        7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
        -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
        -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
        -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
        -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
        -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
        -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
        -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
        -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
        1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
        1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
        9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
        5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
        -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
        -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
        -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
        -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
        8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
        2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
        7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
        -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
        -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
        4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
        3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
        -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
        5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
        4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
        -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
        0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
        -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
        3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
        -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
        8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
        -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
        2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
        10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
        6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
        -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
        -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
        -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
        -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
        -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
        4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
        2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
        6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
        3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
        11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
        -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
        4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
        2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
        -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
        -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
        -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
        6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
        0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
        -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
        -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
        -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
        5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
        2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
        -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
        9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
        11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
        3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
        -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
        3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
        -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
        5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
        8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
        7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
        -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
        7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
        9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
        7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
        -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

const float factorPI = (float)(CV_PI/180.f);
static void computeOrbDescriptor(const cv::KeyPoint& kpt,const cv::Mat& img, const cv::Point* pattern, uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
            cvRound(pattern[idx].x*a - pattern[idx].y*b)]


    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

static void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const std::vector<cv::Point>& pattern)
{
    descriptors = cv::Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

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
    const int step {(int)image.step1()};
    const uchar* center = &image.at<uchar> (cvRound(point.y), cvRound(point.x));

    for (int u = -halfPatchSize; u <= halfPatchSize; ++u)
        m10 += u * center[u];

    for (int32_t row = 1; row < halfPatchSize; row++)
    {
        int sumIntensities {0};
        int d = umax[row];

        for (int32_t col = -d; col <= d; col++)
        {
            const int centerP {center[col + row*step]}, centerM {center[col - row*step]};
            sumIntensities += centerP - centerM;
            m10 += col * (centerP + centerM);
        }
        m01 += row * sumIntensities;
    }

    return cv::fastAtan2((float)m01, (float)m10);

}

void FeatureExtractor::computePyramid(const cv::Mat& image)
{
    for (int level = 0; level < nLevels; ++level)
    {
        float scale = scaleInvPyramid[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::Size wholeSize(sz.width + edgeThreshold*2, sz.height + edgeThreshold*2);
        cv::Mat temp(wholeSize, image.type()), masktemp;
        imagePyramid[level] = temp(cv::Rect(edgeThreshold, edgeThreshold, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(imagePyramid[level-1], imagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

            copyMakeBorder(imagePyramid[level], temp, edgeThreshold, edgeThreshold, edgeThreshold, edgeThreshold,
                            cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, edgeThreshold, edgeThreshold, edgeThreshold, edgeThreshold,
                            cv::BORDER_REFLECT_101);
        }
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

void FeatureExtractor::extractORBGrids(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints)
{

    findORBGrids(leftImage,keypoints.left, desc.left);
    findORBGrids(rightImage,keypoints.right, desc.right);
    
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
    
    // computeKeypoints(leftImage,keypoints.left, desc.left, 0);
    // computeKeypoints(rightImage,keypoints.right, desc.right, 1);

    std::thread extractLeft(&FeatureExtractor::computeKeypoints, this, std::ref(leftImage), std::ref(keypoints.left), std::ref(desc.left), 0);
    std::thread extractRight(&FeatureExtractor::computeKeypoints, this, std::ref(rightImage), std::ref(keypoints.right), std::ref(desc.right), 1);
    extractLeft.join();
    extractRight.join();

    // findFASTGridsMask(leftImage,keypoints.left, mask);
    // findFASTGrids(rightImage,keypoints.right);
    

    // cv::Mat leftim = leftImage.clone();
    // cv::Mat rightim = rightImage.clone();

    // cv::GaussianBlur(leftim, leftim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    // cv::GaussianBlur(rightim, rightim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

    // detector->compute(leftim, keypoints.left, desc.left);
    // detector->compute(rightim, keypoints.right, desc.right);


    // desc.left = cv::Mat(keypoints.left.size(), 32, CV_8U);
    // desc.right = cv::Mat(keypoints.right.size(), 32, CV_8U);

    // computeDescriptors(leftim, keypoints.left, desc.left, pattern);
    // computeDescriptors(rightim, keypoints.right, desc.right, pattern);
    // detectorbrisk->compute(leftim, keypoints.left, desc.left);
    // detectorbrisk->compute(rightim, keypoints.right, desc.right);

    // cv::BRISK::compute(leftim, keypoints.left, desc.left);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::extractFeatures(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints)
{

    // findFASTGrids(leftImage,keypoints.left);
    std::thread extractLeft(&FeatureExtractor::computeKeypoints, this, std::ref(leftImage), std::ref(keypoints.left), std::ref(desc.left), 0);
    std::thread extractRight(&FeatureExtractor::computeKeypoints, this, std::ref(rightImage), std::ref(keypoints.right), std::ref(desc.right), 1);
    extractLeft.join();
    extractRight.join();
    // computeKeypoints(leftImage,keypoints.left, desc.left, 0);
    // computeKeypoints(rightImage,keypoints.right, desc.right, 1);
    // cv::Mat outIm = leftImage.clone();
    // for (auto& key:keypoints.left)
    // {
    //     cv::circle(outIm, key.pt,2,cv::Scalar(0,255,0));

    // }
    // cv::imshow("e", outIm);
    // cv::waitKey(0);
    // findFASTGrids(rightImage,keypoints.right);

    // cv::Mat leftim = leftImage.clone();
    // cv::Mat rightim = rightImage.clone();

    // cv::GaussianBlur(leftim, leftim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    // cv::GaussianBlur(rightim, rightim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

    // detector->compute(leftim, keypoints.left, desc.left);
    // detector->compute(rightim, keypoints.right, desc.right);

    // desc.left = cv::Mat(keypoints.left.size(), 32, CV_8U);
    // desc.right = cv::Mat(keypoints.right.size(), 32, CV_8U);

    // computeDescriptors(leftim, keypoints.left, desc.left, pattern);
    // computeDescriptors(rightim, keypoints.right, desc.right, pattern);
    // detectorbrisk->compute(leftim, keypoints.left, desc.left);
    // detectorbrisk->compute(rightim, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::extractFeaturesClose(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints)
{

    findFASTGridsClose(leftImage,keypoints.left);
    findFASTGridsClose(rightImage,keypoints.right);

    cv::Mat leftim = leftImage.clone();
    cv::Mat rightim = rightImage.clone();

    cv::GaussianBlur(leftim, leftim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    cv::GaussianBlur(rightim, rightim, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

    // detector->compute(leftim, keypoints.left, desc.left);
    // detector->compute(rightim, keypoints.right, desc.right);
    computeDescriptors(leftim, keypoints.left, desc.left, pattern);
    computeDescriptors(rightim, keypoints.right, desc.right, pattern);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::extractFeaturesCloseMask(cv::Mat& leftImage, cv::Mat& rightImage, StereoDescriptors& desc, StereoKeypoints& keypoints, const cv::Mat& mask)
{
    findFASTGridsCloseMask(leftImage,keypoints.left, mask);
    findFASTGridsClose(rightImage,keypoints.right);

    detector->compute(leftImage, keypoints.left, desc.left);
    detector->compute(rightImage, keypoints.right, desc.right);

    // updatePoints(leftKeys, rightKeys,points);
    
}

void FeatureExtractor::computeKeypoints(cv::Mat& image, std::vector < std::vector<cv::KeyPoint> >& allKeypoints, const bool right)
{
    computePyramid(image);

    const int fastEdge = 3;

    int gridsWKeys {0};
    std::vector < std::vector<cv::KeyPoint> > allKeypoints(nLevels, std::vector<cv::KeyPoint>());

    std::vector< std::vector< std::vector<cv::KeyPoint>>> cellKeys (gridRows, std::vector<std::vector<cv::KeyPoint>>(gridCols, std::vector<cv::KeyPoint>()));

    for (size_t level {0}; level < nLevels; level ++)
    {
        const int minX = edgeThreshold;
        const int maxX = imagePyramid[level].cols - edgeThreshold;
        const int minY = edgeThreshold;
        const int maxY = imagePyramid[level].rows - edgeThreshold;

        const int wid = maxX - minX;
        const int hig = maxY - minY;

        const int gridW = cvRound((float)wid/gridCols);
        const int gridH = cvRound((float)hig/gridRows);

        const int nGrids = gridCols * gridRows;
        allKeypoints[level].reserve(5 * nFeatures / nGrids);

        const int featuresGrid = (int)ceil((float)5.0f * nFeatures/nGrids);



        float rowJump = gridH + 2 * fastEdge;


        std::vector< std::vector< bool>> mnContrast (gridRows, std::vector<bool>(gridCols));
        
        std::vector<std::vector<int>> keysPerCell(gridRows, std::vector<int>(gridCols));


        

        for (size_t iR = 0; iR < gridRows; iR++)
        {

            const float rStart = minY + iR * gridH - fastEdge;

            if ( rStart >= maxY)
                continue;

            if ( iR == gridRows - 1)
            {
                rowJump = maxY + fastEdge - rStart;
                if ( rowJump <= 0 )
                    continue;
            }

            float colJump = gridW + 2 * fastEdge;

            for (size_t iC = 0; iC < gridCols; iC++)
            {

                float cStart = minX + iC * gridW - fastEdge;

                if ( cStart >= maxX)
                    continue;

                if ( iC == gridCols - 1 )
                {
                    colJump = maxX + fastEdge - cStart;
                    if ( colJump <= 0 )
                        continue;
                }

                // Logging("rows", rStart,3);
                // Logging("roweeeee", rStart + rowJump,3);
                // Logging("cols", cStart,3);
                // Logging("coleeeee", cStart + colJump,3);
                // Logging("iR", iR, 3);
                // Logging("iC", iC, 3);

                const cv::Mat& cellIm = imagePyramid[level].rowRange(rStart, rStart + rowJump).colRange(cStart, cStart + colJump);

                cv::Mat fl;
                cv::flip(cellIm, fl,-1);
                // Logging("fl",cv::norm(im,fl),3);
                if ( cv::norm(cellIm,fl) < mnContr)
                {
                    keysPerCell[iR][iC] = 0;
                    continue;
                }

                mnContrast[iR][iC] = true;

                std::vector<cv::KeyPoint> temp;

                cv::FAST(cellIm, temp, maxFastThreshold,true);
                if (temp.empty())
                    cv::FAST(cellIm, temp, minFastThreshold,true);
                if (!temp.empty())
                {
                    std::vector<cv::KeyPoint>::iterator it;
                    std::vector<cv::KeyPoint>::const_iterator end(temp.end());
                    for (it = temp.begin(); it != end; it++)
                    {
                        it->pt.x += cStart;
                        it->pt.y += rStart;
                        it->octave = level;
                        cellKeys[iR][iC].emplace_back(*it);
                    }
                    if ( level == 0 )
                        gridsWKeys++;

                }
                else
                    keysPerCell[iR][iC] = 0;
            }

        }
    }


    const int desiredFeaturesGrid = (int)floor((float)nFeatures/gridsWKeys) + 1;


    for (size_t iR = 0; iR < gridRows; iR++)
    {
        for (size_t iC = 0; iC < gridCols; iC++)
        {
            const size_t keyCellSize = cellKeys[iR][iC].size();
            int desired;
            if ( !right )
                desired = desiredFeaturesGrid - KeyDestrib[iR][iC];
            else
                desired = desiredFeaturesGrid - KeyDestribRight[iR][iC];
            if ( keyCellSize == 0 || desired <= 0 )
                continue;
            if (keyCellSize > desired)
            {
                cv::KeyPointsFilter::retainBest(cellKeys[iR][iC], desired);
                cellKeys[iR][iC].resize(desired);
            }
            std::vector<cv::KeyPoint>::iterator it;
            std::vector<cv::KeyPoint>::const_iterator end(cellKeys[iR][iC].end());
            for (it = cellKeys[iR][iC].begin(); it != end; it++)
            {
                it->angle = computeOrientation(imagePyramid[it->octave], it->pt);
                allKeypoints[it->octave].emplace_back(*it);
            }
            if ( !right )
                KeyDestrib[iR][iC] += cellKeys[iR][iC].size();
            else
                KeyDestribRight[iR][iC] += cellKeys[iR][iC].size();
        }
    }


    for (size_t level {0}; level < nLevels; level++)
    {
        cv::Mat im = imagePyramid[level].clone();
        cv::Mat desc = cv::Mat(allKeypoints[level].size(), 32, CV_8U);

        cv::GaussianBlur(im, im, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        computeDescriptors(im, allKeypoints[level], desc, pattern);

    }
    //     std::vector<cv::KeyPoint>::iterator it;
    //     std::vector<cv::KeyPoint>::const_iterator end(allKeypoints[level].end());
    //     for ( it = allKeypoints[level].begin(); it != end; it++)
    //         it->angle = computeOrientation(imagePyramid[level], it->pt);

    // for (size_t iR = 0; iR < gridRows; iR++)
    // {
    //     for (size_t iC = 0; iC < gridCols; iC++)
    //     {
    //         std::vector<cv::KeyPoint>::iterator it;
    //         std::vector<cv::KeyPoint>::const_iterator end(cellKeys[iR][iC].end());
    //         for ( it = cellKeys[iR][iC].begin(); it != end; it++)
    //         {
    //             it->angle = computeOrientation(imagePyramid[it->octave], it->pt);
    //             allKeypoints[it->octave].emplace_back(*it);
    //         }
    //     }
    // }




    // detector->compute(leftim, keypoints, desc.left);
    // detector->compute(im, keypoints, desc);

    // Logging("Keypoints size", keypoints.size(),3);

}

void FeatureExtractor::computeKeypointsOld(cv::Mat& image, std::vector <cv::KeyPoint>& keypoints, cv::Mat& desc, const bool right)
{
    const int fastEdge = 3;

    int gridsWKeys {0};


    std::vector< std::vector< std::vector<cv::KeyPoint>>> cellKeys (gridRows, std::vector<std::vector<cv::KeyPoint>>(gridCols, std::vector<cv::KeyPoint>()));

    for (size_t level {0}; level < nLevels; level ++)
    {

        const int minX = edgeThreshold;
        const int maxX = image.cols - edgeThreshold;
        const int minY = edgeThreshold;
        const int maxY = image.rows - edgeThreshold;

        const int wid = maxX - minX;
        const int hig = maxY - minY;

        const int gridW = cvRound((float)wid/gridCols);
        const int gridH = cvRound((float)hig/gridRows);

        const int nGrids = gridCols * gridRows;

        const int featuresGrid = (int)ceil((float)5.0f * nFeatures/nGrids);



        float rowJump = gridH + 2 * fastEdge;


        std::vector< std::vector< bool>> mnContrast (gridRows, std::vector<bool>(gridCols));
        
        std::vector<std::vector<int>> keysPerCell(gridRows, std::vector<int>(gridCols));


        

        for (size_t iR = 0; iR < gridRows; iR++)
        {

            const float rStart = minY + iR * gridH - fastEdge;

            if ( rStart >= maxY)
                continue;

            if ( iR == gridRows - 1)
            {
                rowJump = maxY + fastEdge - rStart;
                if ( rowJump <= 0 )
                    continue;
            }

            float colJump = gridW + 2 * fastEdge;

            for (size_t iC = 0; iC < gridCols; iC++)
            {

                float cStart = minX + iC * gridW - fastEdge;

                if ( cStart >= maxX)
                    continue;

                if ( iC == gridCols - 1 )
                {
                    colJump = maxX + fastEdge - cStart;
                    if ( colJump <= 0 )
                        continue;
                }

                Logging("rows", rStart,3);
                Logging("roweeeee", rStart + rowJump,3);
                Logging("cols", cStart,3);
                Logging("coleeeee", cStart + colJump,3);
                Logging("iR", iR, 3);
                Logging("iC", iC, 3);

                const cv::Mat& cellIm = image.rowRange(rStart, rStart + rowJump).colRange(cStart, cStart + colJump);

                cv::Mat fl;
                cv::flip(cellIm, fl,-1);
                // Logging("fl",cv::norm(im,fl),3);
                if ( cv::norm(cellIm,fl) < mnContr)
                {
                    keysPerCell[iR][iC] = 0;
                    continue;
                }

                mnContrast[iR][iC] = true;

                std::vector<cv::KeyPoint> temp;

                cv::FAST(cellIm, temp, maxFastThreshold,true);
                if (temp.empty())
                    cv::FAST(cellIm, temp, minFastThreshold,true);
                if (!temp.empty())
                {
                    std::vector<cv::KeyPoint>::iterator it;
                    std::vector<cv::KeyPoint>::const_iterator end(temp.end());
                    for (it = temp.begin(); it != end; it++)
                    {
                        it->pt.x += cStart;
                        it->pt.y += rStart;
                        cellKeys[iR][iC].emplace_back(*it);
                    }

                    gridsWKeys++;

                }
                else
                    keysPerCell[iR][iC] = 0;
            }

        }

    }

    const int desiredFeaturesGrid = (int)floor((float)nFeatures/gridsWKeys) + 1;


    for (size_t iR = 0; iR < gridRows; iR++)
    {
        for (size_t iC = 0; iC < gridCols; iC++)
        {
            const size_t keyCellSize = cellKeys[iR][iC].size();
            int desired;
            if ( !right )
                desired = desiredFeaturesGrid - KeyDestrib[iR][iC];
            else
                desired = desiredFeaturesGrid - KeyDestribRight[iR][iC];
            if ( keyCellSize == 0 || desired <= 0 )
                continue;
            if (keyCellSize > desired)
            {
                cv::KeyPointsFilter::retainBest(cellKeys[iR][iC], desired);
                cellKeys[iR][iC].resize(desired);
            }
            std::vector<cv::KeyPoint>::iterator it;
            std::vector<cv::KeyPoint>::const_iterator end(cellKeys[iR][iC].end());
            for (it = cellKeys[iR][iC].begin(); it != end; it++)
            {
                it->angle = computeOrientation(image, it->pt);
                keypoints.emplace_back(*it);
            }
            if ( !right )
                KeyDestrib[iR][iC] += cellKeys[iR][iC].size();
            else
                KeyDestribRight[iR][iC] += cellKeys[iR][iC].size();
        }
    }

    cv::Mat im = image.clone();

    cv::GaussianBlur(im, im, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    computeDescriptors(im, keypoints, desc, pattern);
    // detector->compute(leftim, keypoints, desc.left);
    // detector->compute(im, keypoints, desc);
    Logging("Keypoints size", keypoints.size(),3);

}

void FeatureExtractor::findFASTGridsPop(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const std::vector<int>& pop)
{
    fastKeys.reserve(2000);
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/4};
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
                cv::KeyPointsFilter::retainBest(temp,numberPerCell);
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
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    // Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::findFASTGridsMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask)
{
    fastKeys.reserve(2000);
    std::vector<std::vector <cv::KeyPoint>> allKeys;
    allKeys.reserve((gridCols + 1) * (gridRows + 1));
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/2};
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.
    const int rowJump = (croppedImage.rows) / gridRows;
    const int colJump = (croppedImage.cols) / gridCols;

    int count {-1};
    int gridFoundCount {0};
    for (int32_t row = 0; row < gridRows + 1; row++)
    {
        
        int imRowStart = row * rowJump - fastEdge;
        int imRowEnd = (row + 1) * rowJump + fastEdge;
        if ( imRowEnd > croppedImage.rows )
            imRowEnd = croppedImage.rows;
        if ( imRowStart < 0)
            imRowStart = 0;

        for (int32_t col = 0; col < gridCols + 1; col++)
        {
            count++;

            int imColStart = col * colJump - fastEdge;
            int  imColEnd = (col + 1) * colJump + fastEdge;
            if ( imColEnd > croppedImage.cols )
                imColEnd = croppedImage.cols;
            if ( imColStart < 0 )
                imColStart = 0;

            // Logging("imcolstart",imColStart,3);
            // Logging("imColEnd",imColEnd,3);
            // Logging("imRowStart",imRowStart,3);
            // Logging("imRowEnd",imRowEnd,3);

            std::vector < cv::KeyPoint > temp;

            const cv::Mat& im = croppedImage.colRange(imColStart, imColEnd).rowRange(imRowStart, imRowEnd);

            cv::Mat fl;
            cv::flip(im, fl,-1);
            // Logging("fl",cv::norm(im,fl),3);
            if ( cv::norm(im,fl) < mnContr)
                continue;
            // cv::imshow("c",croppedImage);
            // cv::imshow("im",im);
            // cv::waitKey(0);

            cv::FAST(im,temp,maxFastThreshold,true);


            if (temp.size() < 5)
            {
                temp.clear();
                cv::FAST(im,temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                
                // cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                int er {0};
                for (it=temp.begin(); it != end; it++, er++)
                {
                    if ( (*it).pt.x < 0)
                        continue;
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    if ( mask.at<uchar>(it->pt) == 0 )
                    {
                        temp.erase(temp.begin() + er);
                        continue;
                    }
                    (*it).class_id = count;
                    (*it).angle = 0;
                    // fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
            }
            if (!temp.empty())
            {
                gridFoundCount ++;
                allKeys.emplace_back(temp);
            }
        }
    }

    const int featuresCell {(int)round(nFeatures/gridFoundCount)};

    for (size_t i {0}; i < allKeys.size(); i ++)
    {
        if ( allKeys[i].size() > featuresCell)
        {
            cv::KeyPointsFilter::retainBest(allKeys[i],featuresCell);
            allKeys[i].resize(featuresCell);
        }
        // if ( allKeys[i].empty() )
        //     continue;
        std::vector<cv::KeyPoint>::iterator it;
        std::vector<cv::KeyPoint>::const_iterator end(allKeys[i].end());
        for (it = allKeys[i].begin(); it !=end; it++)
        {
            it->angle = computeOrientation(image,it->pt);
            fastKeys.emplace_back(*it);
        }
    }
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    // fastKeys.resize(nFeatures);
    // std::vector < cv::KeyPoint>::iterator it;
    // std::vector < cv::KeyPoint>::const_iterator end(allKeys.end());
    // for (it=allKeys.begin(); it != end; it++)
    // {

    //     // (*it).angle = {computeOrientation(croppedImage, cv::Point2f((*it).pt.x,(*it).pt.y))};

    //     // (*it).angle = 0;

    //     fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
    // }
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),3);
}

void FeatureExtractor::findFASTGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
{
    fastKeys.reserve(2000);
    std::vector<std::vector <cv::KeyPoint>> allKeys;
    allKeys.reserve((gridCols + 1) * (gridRows + 1));
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);

    const int mnNKey {numberPerCell/2};
    // fastEdge is the Edge Threshold of FAST Keypoints, it does not search for keypoints for a border of 3 pixels around image.

    const int rowJump = (croppedImage.rows) / gridRows;
    const int colJump = (croppedImage.cols ) / gridCols;

    int count {-1};
    int gridFoundCount {0};
    for (int32_t row = 0; row < gridRows + 1; row++)
    {
        
        int imRowStart = row * rowJump - fastEdge;
        int imRowEnd = (row + 1) * rowJump + fastEdge;
        if ( imRowEnd > croppedImage.rows )
            imRowEnd = croppedImage.rows;
        if ( imRowStart < 0)
            imRowStart = 0;

        for (int32_t col = 0; col < gridCols + 1; col++)
        {
            count++;

            int imColStart = col * colJump - fastEdge;
            int  imColEnd = (col + 1) * colJump + fastEdge;
            if ( imColEnd > croppedImage.cols )
                imColEnd = croppedImage.cols;
            if ( imColStart < 0 )
                imColStart = 0;
            // Logging("imRowStart",imRowStart,3);
            // Logging("imRowEnd",imRowEnd,3);
            // Logging("imColStart",imColStart,3);
            // Logging("imColEnd",imColEnd,3);

            std::vector < cv::KeyPoint > temp;

            const cv::Mat& im = croppedImage.colRange(imColStart, imColEnd).rowRange(imRowStart, imRowEnd);

            cv::Mat fl;
            cv::flip(im, fl,-1);
            // Logging("fl",cv::norm(im,fl),3);
            if ( cv::norm(im,fl) < mnContr)
                continue;
            // cv::imshow("c",croppedImage);
            // cv::imshow("im",im);
            // cv::waitKey(0);

            cv::FAST(im,temp,maxFastThreshold,true);

            if (temp.size() < 5)
            {
                cv::FAST(im,temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                gridFoundCount ++;
                // cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                for (it=temp.begin(); it != end; it++)
                {
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    (*it).class_id = count;
                    (*it).angle = 0;
                    // allKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
                }
                allKeys.emplace_back(temp);
            }
        }
    }

    const int featuresCell {(int)round(nFeatures/gridFoundCount)};

    for (size_t i {0}; i < allKeys.size(); i ++)
    {
        cv::KeyPointsFilter::retainBest(allKeys[i],featuresCell);
        std::vector<cv::KeyPoint>::iterator it;
        std::vector<cv::KeyPoint>::const_iterator end(allKeys[i].end());
        for (it = allKeys[i].begin(); it !=end; it++)
        {
            it->angle = computeOrientation(image,it->pt);
            fastKeys.emplace_back(*it);
        }
    }
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    // fastKeys.resize(nFeatures);

    // std::vector < cv::KeyPoint>::iterator it;
    // std::vector < cv::KeyPoint>::const_iterator end(allKeys.end());
    // for (it=allKeys.begin(); it != end; it++)
    // {

    //     // (*it).angle = {computeOrientation(croppedImage, cv::Point2f((*it).pt.x,(*it).pt.y))};

    //     // (*it).angle = 0;

    //     fastKeys.emplace_back(cv::Point2f((*it).pt.x,(*it).pt.y), (*it).size,(*it).angle,(*it).response,(*it).octave,(*it).class_id);
    // }
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::findORBGrids(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, cv::Mat& Desc)
{
    fastKeys.reserve(2000);
    // std::vector <cv::KeyPoint> allKeys;
    // allKeys.reserve(4000);
    const int fastEdge = 3;
    const int edgeWFast = edgeThreshold - fastEdge;
    cv::Mat croppedImage = image.colRange(edgeWFast,image.cols - edgeWFast).rowRange(edgeWFast, image.rows - edgeWFast);
    cv::Ptr<cv::ORB> detector {cv::ORB::create(2000,imScale,nLevels,edgeThreshold,0,2,cv::ORB::HARRIS_SCORE,patchSize,maxFastThreshold)};
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

            // Logging("imRowStart",imRowStart,3);
            // Logging("imRowEnd",imRowEnd,3);
            // Logging("imColStart",imColStart,3);
            // Logging("imColEnd",imColEnd,3);

            std::vector < cv::KeyPoint > temp;

            detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp);

            // cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.size() < mnNKey)
            {
                detector->setFastThreshold(minFastThreshold);
                detector->detect(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp);
                detector->setFastThreshold(maxFastThreshold);

            }
            if (!temp.empty())
            {
                cv::KeyPointsFilter::retainBest(temp,numberPerCell);
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
    Logging("Keypoint Size After removal", fastKeys.size(),1);
    detector->compute(image, fastKeys, Desc);
}

void FeatureExtractor::findFASTGridsClose(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys)
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
    
    for (int32_t row = 0; row < gridRows + 1; row++)
    {
        
        const int imRowStart = row * rowJump;
        int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;
        if ( imRowEnd > croppedImage.rows )
            imRowEnd = croppedImage.rows;

        for (int32_t col = 0; col < gridCols + 1; col++)
        {
            count++;

            const int imColStart = col * colJump;
            int  imColEnd = (col + 1) * colJump + 2 * fastEdge;
            if ( imColEnd > croppedImage.cols )
                imColEnd = croppedImage.cols;

            // Logging("imRowStart",imRowStart,3);
            // Logging("imRowEnd",imRowEnd,3);
            // Logging("imColStart",imColStart,3);
            // Logging("imColEnd",imColEnd,3);

            std::vector < cv::KeyPoint > temp;

            cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.empty())
            {
                cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                cv::KeyPointsFilter::retainBest(temp,numberPerCell);
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
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
    Logging("Keypoint Size After removal", fastKeys.size(),1);
}

void FeatureExtractor::findFASTGridsCloseMask(cv::Mat& image, std::vector <cv::KeyPoint>& fastKeys, const cv::Mat& mask)
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
    
    for (int32_t row = 0; row < gridRows + 1; row++)
    {
        
        const int imRowStart = row * rowJump;
        int imRowEnd = (row + 1) * rowJump + 2 * fastEdge;
        if ( imRowEnd > croppedImage.rows )
            imRowEnd = croppedImage.rows;

        for (int32_t col = 0; col < gridCols + 1; col++)
        {
            count++;

            const int imColStart = col * colJump;
            int  imColEnd = (col + 1) * colJump + 2 * fastEdge;
            if ( imColEnd > croppedImage.cols )
                imColEnd = croppedImage.cols;

            // Logging("imRowStart",imRowStart,3);
            // Logging("imRowEnd",imRowEnd,3);
            // Logging("imColStart",imColStart,3);
            // Logging("imColEnd",imColEnd,3);

            std::vector < cv::KeyPoint > temp;

            cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,maxFastThreshold,true);

            if (temp.empty())
            {
                cv::FAST(croppedImage.colRange(cv::Range(imColStart, imColEnd)).rowRange(cv::Range(imRowStart, imRowEnd)),temp,minFastThreshold,true);
            }
            if (!temp.empty())
            {
                cv::KeyPointsFilter::retainBest(temp,numberPerCell);
                std::vector < cv::KeyPoint>::iterator it;
                std::vector < cv::KeyPoint>::const_iterator end(temp.end());
                for (it=temp.begin(); it != end; it++)
                {
                    (*it).pt.x += imColStart + edgeWFast;
                    (*it).pt.y += imRowStart + edgeWFast;
                    if ( mask.at<uchar>(it->pt) == 0 )
                        continue;
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
    // cv::KeyPointsFilter::retainBest(fastKeys,nFeatures);
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

FeatureExtractor::FeatureExtractor(const int _nfeatures, const int _nLevels, const float _imScale, const int _edgeThreshold, const int _patchSize, const int _maxFastThreshold, const int _minFastThreshold, const bool _nonMaxSuppression) : nFeatures(_nfeatures), nLevels(_nLevels), imScale(_imScale), edgeThreshold(_edgeThreshold), patchSize(_patchSize), maxFastThreshold(_maxFastThreshold), minFastThreshold(_minFastThreshold), nonMaxSuppression(_nonMaxSuppression), detector(cv::ORB::create(_nfeatures,imScale,nLevels,edgeThreshold,0,2,cv::ORB::FAST_SCORE,patchSize,maxFastThreshold))
{
    KeyDestrib = std::vector<std::vector<int>>(gridRows, std::vector<int>(gridCols));
    KeyDestribRight = std::vector<std::vector<int>>(gridRows, std::vector<int>(gridCols));

    scalePyramid.resize(nLevels);
    scaleInvPyramid.resize(nLevels);
    scalePyramid[0] = 1.0f;
    scaleInvPyramid[0] = 1.0f;
    
    for(int i=1; i<nLevels; i++)
    {
        scalePyramid[i]=scalePyramid[i-1]*imScale;
    }

    scaleInvPyramid.resize(nLevels);

    for(int i=0; i<nLevels; i++)
    {
        scaleInvPyramid[i]=1.0f/scalePyramid[i];
    }

    imagePyramid.resize(nLevels);

    featurePerLevel.resize(nLevels);
    float factor = 1.0f / imScale;
    float nDesiredFeaturesPerScale = nFeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nLevels));

    int sumFeatures = 0;
    for( int level = 0; level < nLevels-1; level++ )
    {
        featurePerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += featurePerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    featurePerLevel[nLevels-1] = std::max(nFeatures - sumFeatures, 0);


    const int npoints = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    umax.resize(halfPatchSize + 1);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    const double hp2 = halfPatchSize*halfPatchSize;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
    
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