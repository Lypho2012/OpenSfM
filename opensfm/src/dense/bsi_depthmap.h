#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include <opencv2/opencv.hpp>

namespace bsidense {

class BsiNCCEstimator {
public:
    BsiNCCEstimator();
    void Push(BsiAttribute<uint64_t>* x, BsiAttribute<uint64_t>* y, BsiAttribute<uint64_t>* w);
    BsiAttribute<uint64_t>* Get();

private:
    BsiAttribute<uint64_t>* sumx_, sumy_;
    BsiAttribute<uint64_t>* sumxx_, sumyy_, sumxy_;
    BsiAttribute<uint64_t>* sumw_;
};

std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(const cv::Matx33d &K1inv,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &Q2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &a2,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &v);

struct DepthmapEstimatorResult {
    std::vector<BsiAttribute<uint64_t>*> depth; // float
    std::vector<std::vector<BsiAttribute<uint64_t>*>> plane;  // float
    // TODO: consider making a plane object instead of using std::vector<BsiAttribute<uint64_t>*> since the vector only has 3 elements
    std::vector<BsiAttribute<uint64_t>*> score;  // float
    std::vector<BsiAttribute<uint64_t>*> nghbr;  // unsigned int
};

class BsiDepthmapEstimator {
public:

void AssignMatrices(DepthmapEstimatorResult *result);
void RandomInitialization(DepthmapEstimatorResult *result, bool sample);
void ComputeIgnoreMask(DepthmapEstimatorResult *result);
void ComputePatchMatch(DepthmapEstimatorResult *result);
void PatchMatchForwardPass(DepthmapEstimatorResult *result, bool sample);
void PatchMatchBackwardPass(DepthmapEstimatorResult *result, bool sample);
void PostProcess(DepthmapEstimatorResult *result);

void PatchMatchUpdatePixelRow(DepthmapEstimatorResult *result, int i, int adjacent[2][2], bool sample);
void CheckPlaneImageCandidate(DepthmapEstimatorResult *result, int i,
                              const std::vector<BsiAttribute<uint64_t>*> &plane, BsiAttribute<uint64_t>* nghbr);

BsiAttribute<uint64_t>* ComputePlaneImageScore(BsiAttribute<uint64_t>* i, BsiAttribute<uint64_t>* j,
                                               const std::vector<BsiAttribute<uint64_t>*> &plane,
                                               BsiAttribute<uint64_t>* other);


private:
int patchmatch_iterations_;
int patch_size_;
std::vector<std::vector<BsiAttribute<uint64_t>*>> images_;
std::vector<std::vector<BsiAttribute<uint64_t>*>> masks_; // TODO: make masks BsiAttribute of bools
};

}