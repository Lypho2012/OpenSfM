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

BsiAttribute<uint64_t>* ComputePlaneImageScore();

private:
int patchmatch_iterations_;
int patch_size_;
};

}