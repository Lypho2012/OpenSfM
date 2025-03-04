#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include "../bsi_depthmap.h"

#include <iostream>

namespace bsidense {

std::vector<BsiAttribute<uint64_t>*> PlaneFromDepthAndNormal(BsiAttribute<uint64_t>* x, BsiAttribute<uint64_t>* y,
                                                             const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K,
                                                             BsiAttribute<uint64_t>* depth,
                                                             const std::vector<BsiAttribute<uint64_t>*> &normal) {
    // TODO: implement inverse, replace multiplication operations with matrix mult
    std::vector<BsiAttribute<uint64_t>*> homogeneous_coord;
    BsiSigned<uint64_t> bsi;
    std::vector<int> ones(x->rows, 1);
    homogeneous_coord.push_back(x);
    homogeneous_coord.push_back(y);
    homogeneous_coord.push_back(bsi.buildBsiAttributeFromVectorSigned(ones));
    std::vector<BsiAttribute<uint64_t>*> point = depth * inverse(K) * homogeneous_coord;
    // TODO: replace dot and division, replace max with softmax
    BsiAttribute<uint64_t> denom = -normal.dot(point);
    return normal / std::max(1e-6f, denom);
}

void BsiDepthmapEstimator::AssignMatrices(DepthmapEstimatorResult *result) {
    BsiSigned<uint64_t> bsi;
    std::vector<int> vec(images[0].cols);

    for (int row=0; row<images[0].rows; row++) {
        // assign depth
        result->depth.push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));

        // assign plane
        std::vector<BsiAttribute<uint64_t>*> plane_vec;
        result->plane.push_back(plane_vec);
        result->plane[row].push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));
        result->plane[row].push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));
        result->plane[row].push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));

        // assign score
        result->score.push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));

        // assign nghbr
        result->nghbr.push_back(bsi.buildBsiAttributeFromVectorSigned(vec,0.5));
    }
}

void BsiDepthmapEstimator::RandomInitialization(DepthmapEstimatorResult *result, bool sample) {
    int hpz = (patch_size_ - 1) / 2;
    BsiSigned<uint64_t> bsi;
    std::vector<int> normal_z(result->depth.cols, -1);

    for (int i = hpz; i < result->depth.rows - hpz; ++i) {
        // TODO: substitute UniformRand, uni_, rng_, and exp for bsi functions
        // initialize depth
        BsiAttribute<uint64_t> depth = exp(UniformRand(log(min_depth_), log(max_depth_)));
        result->depth.push_back(depth);

        // generate normal
        std::vector<BsiAttribute<uint64_t>*> normal;
        normal.push_back(UniformRand(-1, 1));
        normal.push_back(UniformRand(-1, 1));
        normal.push_back(bsi.buildBsiAttributeFromVectorSigned(normal_z));
        result->normal.push_back(normal);

        // initialize plane
        std::vector<BsiAttribute<uint64_t>*> plane = PlaneFromDepthAndNormal(j, i, Ks_[0], depth, normal);
        result->plane.push_back(plane);

        // initialize nghbr and score
        BsiAttribute<uint64_t>* nghbr;
        BsiAttribute<uint64_t>* score;
        if (sample) {
            nghbr = uni_(rng_);
            score = ComputePlaneImageScore(i, j, plane, nghbr);
        } else {
            // TODO: don't implement for now, focus on compute patch match sample
            ComputePlaneScore(i, j, plane, &score, &nghbr);
        }
        result->nghbr.push_back(nghbr);
        result->score.push_back(score);
    }
}

void BsiDepthmapEstimator::ComputePatchMatch(DepthmapEstimatorResult *result) {
    AssignMatrices(result);
    RandomInitialization(result, false);
    ComputeIgnoreMask(result); // TODO
    
    for (int i = 0; i < patchmatch_iterations_; ++i) {
        PatchMatchForwardPass(result, false); // TODO
        PatchMatchBackwardPass(result, false); // TODO
    }
    
    PostProcess(result); // TODO
}

// TODO
BsiAttribute<uint64_t>* BsiDepthmapEstimator::ComputePlaneImageScore(BsiAttribute<uint64_t>* i, BsiAttribute<uint64_t>* j,
                                                const std::vector<BsiAttribute<uint64_t>*> &plane,
                                                BsiAttribute<uint64_t>* other) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> H = PlaneInducedHomographyBaked(Kinvs_[0], Qs_[other], as_[other],
                                                Ks_[other], plane);
    int hpz = (patch_size_ - 1) / 2;
    BsiAttribute<uint64_t>* u = H(0, 0) * j + H(0, 1) * i + H(0, 2);
    BsiAttribute<uint64_t>* v = H(1, 0) * j + H(1, 1) * i + H(1, 2);
    BsiAttribute<uint64_t>* w = H(2, 0) * j + H(2, 1) * i + H(2, 2);

    if (w == 0.0) {
        return -1.0f;
    }

    //du/dx
    float dfdx_x = (H(0, 0) * w - H(2, 0) * u) / (w * w);
    //du/dy
    float dfdx_y = (H(1, 0) * w - H(2, 0) * v) / (w * w);
    //dv/dx
    float dfdy_x = (H(0, 1) * w - H(2, 1) * u) / (w * w);
    //dv/dy
    float dfdy_y = (H(1, 1) * w - H(2, 1) * v) / (w * w);

    // homogeneous coordinates of center of corresponding patch in other image
    float Hx0 = u / w;
    float Hy0 = v / w;

    float im1_center = images_[0].at<unsigned char>(i, j);

    NCCEstimator ncc;
    for (int dy = -hpz; dy <= hpz; ++dy) {
        for (int dx = -hpz; dx <= hpz; ++dx) {
            float im1 = images_[0].at<unsigned char>(i + dy, j + dx);
            // subpixel coordinates of (i+dy,j+dx) in images_[other]
            float x2 = Hx0 + dfdx_x * dx + dfdy_x * dy;
            float y2 = Hy0 + dfdx_y * dx + dfdy_y * dy;
            float im2 = LinearInterpolation<unsigned char>(images_[other], y2, x2);
            float weight = BilateralWeight(im1 - im1_center, dx, dy);
            ncc.Push(im1, im2, weight);
        }
    }
    return ncc.Get();
}

}