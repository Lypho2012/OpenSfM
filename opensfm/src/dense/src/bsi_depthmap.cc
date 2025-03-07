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
    // TODO: replace dot, replace max with softmax
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

void BsiDepthmapEstimator::ComputeIgnoreMask(DepthmapEstimatorResult *result) {
    int hpz = (patch_size_ - 1) / 2;
    for (int i = hpz; i < result->depth.rows - hpz; ++i) {
        // masked represents a vector of 1's and 0's, where 0 means not masked
        BsiAttribute<uint64_t>* masked = masks_[0].at<unsigned char>[i];
        // TODO: use softmax to calculate low variance
        bool low_variance = PatchVariance(i, j) < min_patch_variance_;
        // TODO: apply convolution based on where masked + low_variance is not equal to 0
        if (masked || low_variance) {
            AssignPixel(result, i, j, 0.0f, cv::Vec3f(0, 0, 0), 0.0f, 0);
        }
    }
}

void BsiDepthmapEstimator::ComputePatchMatch(DepthmapEstimatorResult *result) {
    AssignMatrices(result);
    RandomInitialization(result, false);
    ComputeIgnoreMask(result);
    
    for (int i = 0; i < patchmatch_iterations_; ++i) {
        PatchMatchForwardPass(result, false); // TODO
        PatchMatchBackwardPass(result, false); // TODO
    }
    
    PostProcess(result); // TODO
}

void BsiDepthmapEstimator::PatchMatchForwardPass(DepthmapEstimatorResult *result,
                                              bool sample) {
    int adjacent[2][2] = {{-1, 0}, {0, -1}};
    int hpz = (patch_size_ - 1) / 2;
    for (int i = hpz; i < result->depth.rows - hpz; ++i) {
        PatchMatchUpdatePixelRow(result, i, adjacent, sample);
    }
}

void BsiDepthmapEstimator::PatchMatchUpdatePixel(DepthmapEstimatorResult *result,
                                              int i, int adjacent[2][2],
                                              bool sample) {
    // Ignore pixels with depth == 0.
    // TODO: mask at the end with softmax
    if (result->depth.at<float>(i, j) == 0.0f) {
        return;
    }

    // Check neighbors and their best match to see if it is also this pixel's best match
    for (int k = 0; k < 2; ++k) {
        int i_adjacent = i + adjacent[k][0];
        int j_adjacent = j + adjacent[k][1];

        // Do not propagate ignored adjacent pixels.
        // TODO: mask
        if (result->depth.at<float>(i_adjacent, j_adjacent) == 0.0f) {
            continue;
        }

        // TODO: plane needs to be shifted by j_adjacent
        std::vector<BsiAttribute<uint64_t>*> plane = result->plane.at(i_adjacent);

        if (sample) {
            // TODO: nghbr needs to be shifted by j_adjacent
            BsiAttribute<uint64_t>* nghbr = result->nghbr.at<int>(i_adjacent);
            CheckPlaneImageCandidate(result, i, plane, nghbr);
        } else {
            // TODO: don't implement for now and focus on sample
            CheckPlaneCandidate(result, i, j, plane);
        }
    }

    // TODO
    // Perturb depth and plane guesses to see which is the best match with the other image
    float depth_range = 0.02;
    float normal_range = 0.5;
    int current_nghbr = result->nghbr.at<int>(i, j);
    for (int k = 0; k < 6; ++k) {
        float current_depth = result->depth.at<float>(i, j);
        // Similar to log normal distribution of perturbation to depth
        float depth = current_depth * exp(depth_range * unit_normal_(rng_));

        cv::Vec3f current_plane = result->plane.at<cv::Vec3f>(i, j);
        if (current_plane(2) == 0.0) {
            continue;
        }
        // normal distribution of perturbation to normal
        cv::Vec3f normal(-current_plane(0) / current_plane(2) +
                         normal_range * unit_normal_(rng_),
                         -current_plane(1) / current_plane(2) +
                         normal_range * unit_normal_(rng_),
                         -1.0f);

        cv::Vec3f plane = PlaneFromDepthAndNormal(j, i, Ks_[0], depth, normal);
        if (sample) {
            CheckPlaneImageCandidate(result, i, j, plane, current_nghbr);
        } else {
            CheckPlaneCandidate(result, i, j, plane);
        }

        depth_range *= 0.3;
        normal_range *= 0.8;
    }

    if (!sample || images_.size() <= 2) {
        return;
    }

    // Check random other image to escape local minima
    int other_nghbr = uni_(rng_);
    while (other_nghbr == current_nghbr) {
        other_nghbr = uni_(rng_);
    }

    cv::Vec3f plane = result->plane.at<cv::Vec3f>(i, j);
    CheckPlaneImageCandidate(result, i, j, plane, other_nghbr);
}

void BsiDepthmapEstimator::CheckPlaneImageCandidate(
        DepthmapEstimatorResult *result, int i, const std::vector<BsiAttribute<uint64_t>*> &plane,
        BsiAttribute<uint64_t>* nghbr) {
    BsiAttribute<uint64_t>* score = ComputePlaneImageScore(i, j, plane, nghbr);
    // TODO: mask using score
    if (score > result->score.at(i)) {
        BsiAttribute<uint64_t>* depth = DepthOfPlaneBackprojection(j, i, Ks_[0], plane);
        // TODO: implement assignpixelrow
        AssignPixel(result, i, j, depth, plane, score, nghbr);
    }
}

BsiAttribute<uint64_t>* BsiDepthmapEstimator::ComputePlaneImageScore(BsiAttribute<uint64_t>* i, BsiAttribute<uint64_t>* j,
                                                const std::vector<BsiAttribute<uint64_t>*> &plane,
                                                BsiAttribute<uint64_t>* other) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> H = PlaneInducedHomographyBaked(Kinvs_[0], Qs_[other], as_[other],
                                                Ks_[other], plane);
    int hpz = (patch_size_ - 1) / 2;
    // TODO: overload multiplication, sum, division
    BsiAttribute<uint64_t>* u = H[0][0] * j + H[0][1] * i + H[0][2];
    BsiAttribute<uint64_t>* v = H[1][0] * j + H[1][1] * i + H[1][2];
    BsiAttribute<uint64_t>* w = H[2][0] * j + H[2][1] * i + H[2][2];

    // TODO: leave this filter step as a softmax at the end?
    if (w == 0.0) {
        return -1.0f;
    }

    //du/dx
    BsiAttribute<uint64_t>* dfdx_x = (H[0][0] * w - H[2][0] * u) / (w * w);
    //du/dy
    BsiAttribute<uint64_t>* dfdx_y = (H[1][0] * w - H[2][0] * v) / (w * w);
    //dv/dx
    BsiAttribute<uint64_t>* dfdy_x = (H[0][1] * w - H[2][1] * u) / (w * w);
    //dv/dy
    BsiAttribute<uint64_t>* dfdy_y = (H[1][1] * w - H[2][1] * v) / (w * w);

    // homogeneous coordinates of center of corresponding patch in other image
    BsiAttribute<uint64_t>* Hx0 = u / w;
    BsiAttribute<uint64_t>* Hy0 = v / w;

    BsiAttribute<uint64_t>* im1_center = images_[0][i];

    //TODO
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

std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(const cv::Matx33d &K1inv,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &Q2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &a2,
                                                                              const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K2,
                                                                              const std::vector<BsiAttribute<uint64_t>*> &v) {
    // TODO: operations with elements in K1inv are scalar multiplication
    return K2 * (Q2 + a2 * v.t()) * K1inv;
}

}