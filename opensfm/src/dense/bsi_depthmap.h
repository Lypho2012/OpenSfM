#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include <opencv2/opencv.hpp>
#include <random>

namespace bsidense {

class BsiNCCEstimator {
public:
    BsiNCCEstimator();
    ~BsiNCCEstimator();
    void Push(BsiAttribute<uint64_t>* x, BsiAttribute<uint64_t>* y, BsiAttribute<uint64_t>* w);
    BsiAttribute<uint64_t>* Get();

private:
    BsiAttribute<uint64_t>* sumx_;
    BsiAttribute<uint64_t>* sumy_;
    BsiAttribute<uint64_t>* sumxx_;
    BsiAttribute<uint64_t>* sumyy_;
    BsiAttribute<uint64_t>* sumxy_;
    BsiAttribute<uint64_t>* sumw_;
};

std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(BsiAttribute<uint64_t>* other, std::vector<std::vector<BsiAttribute<uint64_t>*>> PIHB_precalc1,
    std::vector<BsiAttribute<uint64_t>*> PIHB_precalc2, const std::vector<BsiAttribute<uint64_t>*> &v, cv::Matx33d front_Kinvs);

std::vector<std::vector<BsiAttribute<uint64_t>*>> multTranspose(const std::vector<BsiAttribute<uint64_t>*> &a, const std::vector<BsiAttribute<uint64_t>*> &b);
std::vector<std::vector<BsiAttribute<uint64_t>*>> multTranspose(const std::vector<int> &a, const std::vector<BsiAttribute<uint64_t>*> &b);
std::vector<std::vector<BsiAttribute<uint64_t>*>> addMatrices(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b);
std::vector<std::vector<BsiAttribute<uint64_t>*>> addMatrices(const std::vector<std::vector<int>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b);
std::vector<std::vector<BsiAttribute<uint64_t>*>> multMatrices(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b);
std::vector<std::vector<BsiAttribute<uint64_t>*>> multMatrixWithConstants(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const cv::Matx33d &b);
std::vector<BsiAttribute<uint64_t>*> multMatrixWithVector(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<BsiAttribute<uint64_t>*> &b);
std::vector<BsiAttribute<uint64_t>*> multVectorWithConstants(const std::vector<BsiAttribute<uint64_t>*> &a, const cv::Matx33d &b);

struct BsiDepthmapEstimatorResult {
    std::vector<BsiAttribute<uint64_t>*> depth; // float
    std::vector<std::vector<BsiAttribute<uint64_t>*>> plane;  // float rows x 3
    // TODO: consider making a plane object instead of using std::vector<BsiAttribute<uint64_t>*> since the vector only has 3 elements
    std::vector<BsiAttribute<uint64_t>*> score;  // float
    std::vector<BsiAttribute<uint64_t>*> nghbr;  // int

    ~BsiDepthmapEstimatorResult();
};

BsiAttribute<uint64_t>* DepthOfPlaneBackprojection(int y,
    const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K,
    const std::vector<BsiAttribute<uint64_t>*> &plane);

class BsiDepthmapEstimator {
public:
~BsiDepthmapEstimator();
void AddView(const double *pK, const double *pR, const double *pt,
    const unsigned char *pimage, const unsigned char *pmask,
    int width, int height);
void ProcessViews();
BsiAttribute<uint64_t>* PatchVariance(int i);
void AssignPixelRow(BsiDepthmapEstimatorResult *result, int i,
                 BsiAttribute<uint64_t>* depth, std::vector<BsiAttribute<uint64_t>*> &plane,
                 BsiAttribute<uint64_t>* score, BsiAttribute<uint64_t>* nghbr, const HybridBitmap<uint64_t> &mask);
void AssignMatrices(BsiDepthmapEstimatorResult *result);
BsiAttribute<uint64_t>* BilateralWeight(BsiAttribute<uint64_t>* dcolor, float dx, float dy);
void RandomInitialization(BsiDepthmapEstimatorResult *result, bool sample);
void ComputeIgnoreMask(BsiDepthmapEstimatorResult *result);
void ComputePatchMatchSample(BsiDepthmapEstimatorResult *result);
void PatchMatchForwardPass(BsiDepthmapEstimatorResult *result, bool sample);
void PatchMatchBackwardPass(BsiDepthmapEstimatorResult *result, bool sample);
void PostProcess(BsiDepthmapEstimatorResult *result);

void PatchMatchUpdatePixelRow(BsiDepthmapEstimatorResult *result, int i, int adjacent[2][2], bool sample);
void CheckPlaneImageCandidate(BsiDepthmapEstimatorResult *result, int i,
                              std::vector<BsiAttribute<uint64_t>*> &plane, BsiAttribute<uint64_t>* nghbr);

BsiAttribute<uint64_t>* ComputePlaneImageScore(int i,
                                               const std::vector<BsiAttribute<uint64_t>*> &plane,
                                               BsiAttribute<uint64_t>* other, int image_width);

void InitializeViews(size_t num_images);
void AddView(const double *pK, const double *pR, const double *pt,
    const unsigned char *pimage, const unsigned char *pmask,
    size_t width, size_t height);
void ProcessViewsToBsi();
void SetDepthRange(double min_depth, double max_depth, int num_depth_planes);
void SetPatchMatchIterations(int n);
void SetPatchSize(int size);
void SetMinPatchSD(float sd);

BsiAttribute<uint64_t>* UniformRand(double low, double high, int size);
BsiAttribute<uint64_t>* exp(BsiAttribute<uint64_t>* bsi);

std::vector<BsiAttribute<uint64_t>*> PlaneFromDepthAndNormal(int y, BsiAttribute<uint64_t>* depth, 
    const std::vector<BsiAttribute<uint64_t>*> normal, const std::vector<BsiAttribute<uint64_t>*> rays_bsi);

private:
std::vector<std::vector<BsiAttribute<uint64_t>*>> images_bsi; // number of images x number of rows matrix
std::vector<HybridBitmap<uint64_t>> mask_; // TODO: only need to store masks_[0]
std::vector<std::vector<BsiAttribute<uint64_t>*>> Ks_bsi; // 3x3 matrix of BsiAttribute
std::vector<std::vector<BsiAttribute<uint64_t>*>> Rs_bsi;
std::vector<BsiAttribute<uint64_t>*> ts_bsi;
std::vector<std::vector<BsiAttribute<uint64_t>*>> Kinvs_bsi;
std::vector<std::vector<BsiAttribute<uint64_t>*>> Qs_bsi;
std::vector<BsiAttribute<uint64_t>*> as_bsi;

std::vector<std::vector<BsiAttribute<uint64_t>*>> PIHB_precalc1;
std::vector<BsiAttribute<uint64_t>*> PIHB_precalc2;

int patchmatch_iterations_;
int patch_size_;
double min_depth_, max_depth_;
int num_depth_planes_;
std::mt19937 rng_;
std::uniform_int_distribution<int> uni_;
float min_patch_variance_;

std::vector<std::vector<std::vector<long>>> images_;
std::vector<std::vector<std::vector<double>>> Ks_;
std::vector<std::vector<std::vector<double>>> Rs_;
std::vector<std::vector<double>> ts_;
std::vector<std::vector<std::vector<double>>> Kinvs_;
std::vector<std::vector<std::vector<double>>> Qs_;
std::vector<std::vector<double>> as_;

cv::Matx33d front_Kinvs;
cv::Matx33d front_R;
cv::Vec3d front_t;
bool front;
int images_processed;
};

}