#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include <opencv2/opencv.hpp>
#include <random>

namespace bsidense {

class BsiNCCEstimator {
public:
    BsiNCCEstimator();
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

std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(const cv::Matx33d &K1inv,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &Q2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &a2,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &v);

std::vector<BsiAttribute<uint64_t>*> PlaneFromDepthAndNormal(int y, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K,
                                                                BsiAttribute<uint64_t>* depth, const std::vector<BsiAttribute<uint64_t>*> &normal);

struct BsiDepthmapEstimatorResult {
    std::vector<BsiAttribute<uint64_t>*> depth; // float
    std::vector<std::vector<BsiAttribute<uint64_t>*>> plane;  // float
    // TODO: consider making a plane object instead of using std::vector<BsiAttribute<uint64_t>*> since the vector only has 3 elements
    std::vector<BsiAttribute<uint64_t>*> score;  // float
    std::vector<BsiAttribute<uint64_t>*> nghbr;  // int
};

BsiAttribute<uint64_t>* DepthOfPlaneBackprojection(int y,
    const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K,
    const std::vector<BsiAttribute<uint64_t>*> &plane);

class BsiDepthmapEstimator {
public:

BsiAttribute<uint64_t>* PatchVariance(int i);
void AssignPixelRow(DepthmapEstimatorResult *result, int i,
                 BsiAttribute<uint64_t>* depth, std::vector<BsiAttribute<uint64_t>*> &plane,
                 BsiAttribute<uint64_t>* score, BsiAttribute<uint64_t>* nghbr, const HybridBitmap<uint64_t> &mask);
void AssignMatrices(DepthmapEstimatorResult *result);
BsiAttribute<uint64_t>* BilateralWeight(BsiAttribute<uint64_t>* dcolor, float dx, float dy);
void RandomInitialization(DepthmapEstimatorResult *result, bool sample);
void ComputeIgnoreMask(DepthmapEstimatorResult *result);
void ComputePatchMatch(DepthmapEstimatorResult *result);
void PatchMatchForwardPass(DepthmapEstimatorResult *result, bool sample);
void PatchMatchBackwardPass(DepthmapEstimatorResult *result, bool sample);
void PostProcess(DepthmapEstimatorResult *result);

void PatchMatchUpdatePixelRow(DepthmapEstimatorResult *result, int i, int adjacent[2][2], bool sample);
void CheckPlaneImageCandidate(DepthmapEstimatorResult *result, int i,
                              std::vector<BsiAttribute<uint64_t>*> &plane, BsiAttribute<uint64_t>* nghbr);

BsiAttribute<uint64_t>* ComputePlaneImageScore(int i,
                                               std::vector<BsiAttribute<uint64_t>*> &plane,
                                               BsiAttribute<uint64_t>* other);


private:
int patchmatch_iterations_;
int patch_size_;
std::vector<std::vector<BsiAttribute<uint64_t>*>> images_;
std::vector<HybridBitmap<uint64_t>> mask_; // TODO: only need to store masks_[0]
double min_depth_, max_depth_;
BsiAttribute<uint64_t>* x; // TODO: should store all coords from 1 to result->depth[0].rows (image width)
std::vector<std::vector<BsiAttribute<uint64_t>*>> Ks_;
std::mt19937 rng_;
float min_patch_variance_;
};

}