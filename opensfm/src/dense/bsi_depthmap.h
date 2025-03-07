#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include <opencv2/opencv.hpp>

namespace bsidense {

class BsiDepthmapEstimator {
public:
struct DepthmapEstimatorResult {
    // precision is represented as a power of 10
    int depth_precision; 
    int plane_precision; 
    int score_precision; 
    std::vector<BsiAttribute<uint64_t>*> depth; // float
    std::vector<std::vector<BsiAttribute<uint64_t>*>> plane;  // float
    std::vector<BsiAttribute<uint64_t>*> score;  // float
    std::vector<BsiAttribute<uint64_t>*> nghbr;  // unsigned int
};
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
std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(const cv::Matx33d &K1inv,
                                        const std::vector<std::vector<BsiAttribute<uint64_t>*>> &Q2,
                                        const std::vector<BsiAttribute<uint64_t>*> &a2,
                                        const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K2,
                                        const std::vector<BsiAttribute<uint64_t>*> &v);

private:
int patchmatch_iterations_;
int patch_size_;
std::vector<std::vector<BsiAttribute<uint64_t>*>> images_;
std::vector<std::vector<BsiAttribute<uint64_t>*>> masks_;
};

}