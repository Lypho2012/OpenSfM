/*
 * Note: behavior for borders of images is not defined; i.e. indices with values between [0,hpz)
 */
#include "bsi/BsiAttribute.hpp"
#include "bsi/BsiSigned.hpp"
#include "bsi/BsiUnsigned.hpp"
#include "../bsi_depthmap.h"

#include <iostream>
// TODO: implement remaining methods called from dense.py
namespace bsidense {
BsiDepthmapEstimatorResult::~BsiDepthmapEstimatorResult() {
    for (BsiAttribute<uint64_t>* el: depth) {
        delete el;
    }
    for (BsiAttribute<uint64_t>* el: score) {
        delete el;
    }
    for (BsiAttribute<uint64_t>* el: nghbr) {
        delete el;
    }
    for (std::vector<BsiAttribute<uint64_t>*> row: plane) {
        for (BsiAttribute<uint64_t>* el: row) {
            delete el;
        }
    }
}

BsiDepthmapEstimator::~BsiDepthmapEstimator() {
    for (std::vector<BsiAttribute<uint64_t>*> image: images_bsi) {
        for (BsiAttribute<uint64_t>* el: image) {
            delete el;
        }
    }
    for (std::vector<BsiAttribute<uint64_t>*> row: Ks_bsi) {
        for (BsiAttribute<uint64_t>* el: row) {
            delete el;
        }
    }
    for (std::vector<BsiAttribute<uint64_t>*> row: Rs_bsi) {
        for (BsiAttribute<uint64_t>* el: row) {
            delete el;
        }
    }
    for (BsiAttribute<uint64_t>* el: ts_bsi) {
        delete el;
    }
    for (std::vector<BsiAttribute<uint64_t>*> row: Kinvs_bsi) {
        for (BsiAttribute<uint64_t>* el: row) {
            delete el;
        }
    }
    for (std::vector<BsiAttribute<uint64_t>*> row: Qs_bsi) {
        for (BsiAttribute<uint64_t>* el: row) {
            delete el;
        }
    }
    for (BsiAttribute<uint64_t>* el: as_bsi) {
        delete el;
    }
}

void BsiDepthmapEstimator::InitializeViews(size_t num_images) {
    for (size_t i=0; i<3; i++) {
        Ks_.emplace_back(std::vector<std::vector<double>>());
        Rs_.emplace_back(std::vector<std::vector<double>>());
        ts_.emplace_back(std::vector<double>());
        Kinvs_.emplace_back(std::vector<std::vector<double>>());
        Qs_.emplace_back(std::vector<std::vector<double>>());
        as_.emplace_back(std::vector<double>());
        
        Ks_bsi.emplace_back(std::vector<BsiAttribute<uint64_t>*>());
        Rs_bsi.emplace_back(std::vector<BsiAttribute<uint64_t>*>());
        Kinvs_bsi.emplace_back(std::vector<BsiAttribute<uint64_t>*>());
        Qs_bsi.emplace_back(std::vector<BsiAttribute<uint64_t>*>());
        for (size_t j=0; j<3; j++) {
            Ks_[i].emplace_back(std::vector<double>());
            Rs_[i].emplace_back(std::vector<double>());
            Kinvs_[i].emplace_back(std::vector<double>());
            Qs_[i].emplace_back(std::vector<double>());
        }
    }
    for (size_t i=0; i<num_images; i++) {
        images_.emplace_back(std::vector<std::vector<long>>());
        images_bsi.emplace_back(std::vector<BsiAttribute<uint64_t>*>());
    }
    front = true;
    images_processed = 0;
}
void BsiDepthmapEstimator::AddView(const double *pK, const double *pR, const double *pt,
        const unsigned char *pimage, const unsigned char *pmask,
        size_t width, size_t height) {
    cv::Matx33d curKinvs(pK);
    curKinvs = curKinvs.inv();
    cv::Matx33d curR(pR);
    cv::Vec3d curt(pt);
    cv::Matx33d curQ(curR);
    curQ = curQ * front_R.t();
    cv::Vec3d cura = curQ * front_t - curt;
    for (size_t i=0; i<3; i++) {
        ts_[i].emplace_back(pt[i]);
        as_[i].emplace_back(cura[i]);
        for (size_t j=0; j<3; j++) {
            Ks_[i][j].emplace_back(pK[3*i+j]);
            Rs_[i][j].emplace_back(pR[3*i+j]);
            Kinvs_[i][j].emplace_back(curKinvs(i,j));
            Qs_[i][j].emplace_back(curQ(i,j));
        }
    }
    cv::Mat curimage(height, width, CV_8U, (void *)pimage);
    for (size_t i=0; i<height; i++) {
        images_[images_processed].emplace_back(std::vector<long>());
        for (size_t j=0; j<width; j++) {
            images_[images_processed][i].emplace_back(curimage.at<uchar>(i,j));
        }
    }
    images_processed ++;
    if (front) {
        cv::Mat curmask(height, width, CV_8U, (void *)pmask);
        for (size_t i=0; i<height; i++) {
            HybridBitmap<uint64_t> row(width);
            for (size_t j=0; j<width; j++) {
                row.add(curmask.at<uchar>(i,j)-0);
            }
            mask_.emplace_back(row);
        }
        front_R = curR;
        front_t = curt;
        front_Kinvs = curKinvs;
        front = false;
    }
    std::size_t size = images_processed;
    int a = (size > 1) ? 1 : 0;
    int b = (size > 1) ? size - 1 : 0;
    uni_.param(std::uniform_int_distribution<int>::param_type(a, b));
}

void BsiDepthmapEstimator::ProcessViewsToBsi() {
    BsiSigned<uint64_t> bsi;
    for (size_t i=0; i<3; i++) {
        ts_bsi.emplace_back(bsi.buildBsiAttributeFromVectorSigned(ts_[i]));
        as_bsi.emplace_back(bsi.buildBsiAttributeFromVectorSigned(as_[i]));
        for (size_t j=0; j<3; j++) {
            Ks_bsi[i].emplace_back(bsi.buildBsiAttributeFromVectorSigned(Ks_[i][j]));
            Rs_bsi[i].emplace_back(bsi.buildBsiAttributeFromVectorSigned(Rs_[i][j]));
            Kinvs_bsi[i].emplace_back(bsi.buildBsiAttributeFromVectorSigned(Kinvs_[i][j]));
            Qs_bsi[i].emplace_back(bsi.buildBsiAttributeFromVectorSigned(Qs_[i][j]));
        }
    }
    for (size_t i=0; i<images_.size(); i++) {
        for (size_t j=0; j<images_[i].size(); j++) {
            images_bsi[i].emplace_back(bsi.buildBsiAttributeFromVectorSigned(images_[i][j],0.5));
        }
    }
    PIHB_precalc1 = multMatrixWithConstants(multMatrices(Ks_bsi, Qs_bsi), front_Kinvs);
    PIHB_precalc2 = multMatrixWithVector(Ks_bsi, as_bsi);
}

void BsiDepthmapEstimator::SetDepthRange(double min_depth, double max_depth,
    int num_depth_planes) {
    min_depth_ = min_depth;
    max_depth_ = max_depth;
    num_depth_planes_ = num_depth_planes;
}

void BsiDepthmapEstimator::SetPatchMatchIterations(int n) {
    patchmatch_iterations_ = n;
}

void BsiDepthmapEstimator::SetPatchSize(int size) {
    patch_size_ = size;
}

void BsiDepthmapEstimator::SetMinPatchSD(float sd) {
    min_patch_variance_ = sd * sd;
  }

std::vector<BsiAttribute<uint64_t>*> BsiDepthmapEstimator::PlaneFromDepthAndNormal(int y,
                                                             BsiAttribute<uint64_t>* depth,
                                                             const std::vector<BsiAttribute<uint64_t>*> normal,
                                                             const std::vector<BsiAttribute<uint64_t>*> rays_bsi) {
    std::vector<BsiAttribute<uint64_t>*> point;
    // TODO: remove cast to long when precision is implemented
    point.push_back(rays_bsi[0]->SUM(static_cast<long>(y*front_Kinvs(0,1)))->multiplyWithBsiHorizontal(depth));
    point.push_back(rays_bsi[1]->SUM(static_cast<long>(y*front_Kinvs(1,1)))->multiplyWithBsiHorizontal(depth));
    point.push_back(rays_bsi[2]->SUM(static_cast<long>(y*front_Kinvs(2,1)))->multiplyWithBsiHorizontal(depth));

    std::vector<BsiAttribute<uint64_t>*> res;
    for (int i=0; i<normal.size(); i++) {
        // BsiAttribute<uint64_t>* denom = normal[i]->multiplyWithBsiHorizontal(point[i]);
        // HybridBitmap<u_int64_t> new_sign = denom->sign.Not();
        // denom->sign = new_sign;
        // std::pair<BsiAttribute<uint64_t>*, BsiAttribute<uint64_t>*> div_res = normal[i]->divide(denom);
        // res.push_back(div_res.first);
        // delete div_res.second;
        // delete denom;
        res.push_back(normal[i]->multiplyByConstant(1));
    }
    delete point[0];
    delete point[1];
    delete point[2];
    return res;
}

void BsiDepthmapEstimator::AssignMatrices(BsiDepthmapEstimatorResult *result) {
    BsiSigned<uint64_t> bsi;
    std::vector<long> vec(images_[0][0].size(),0);

    for (int row=0; row<images_[0].size(); row++) {
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

BsiAttribute<uint64_t>* BsiDepthmapEstimator::UniformRand(double low, double high, int size) {
    BsiSigned<uint64_t> bsi;
    int precision = static_cast<int>(pow(10,5));
    BsiAttribute<uint64_t>* res = bsi.createRandomBsi(size,static_cast<int>((high-low)*precision),0.5);
    res = res->SUM(static_cast<long>(low*precision));
    res->decimals = precision;
    return res;
}

// TODO:
BsiAttribute<uint64_t>* BsiDepthmapEstimator::exp(BsiAttribute<uint64_t>* bsi) {
    return bsi;
}

void BsiDepthmapEstimator::RandomInitialization(BsiDepthmapEstimatorResult *result, bool sample) {
    int hpz = (patch_size_ - 1) / 2;
    BsiSigned<uint64_t> bsi;
    std::vector<long> normal_z(result->depth[0]->rows, -1);
    int image_width = result->depth[0]->rows;

    // setup rays_bsi to be used in PlaneFromDepthAndNormal
    std::vector<long> x_coord;
    for (int i=0; i<image_width; i++) {
        x_coord.push_back(i);
    }
    BsiAttribute<uint64_t>* x = bsi.buildBsiAttributeFromVectorSigned(x_coord,0.5);

    std::vector<BsiAttribute<uint64_t>*> rays_bsi; // 3 x width (depth * K.inv() * cv::Vec3d(x, 0, 1))
    // TODO: change precision when multiplying by numbers in front_Kinvs
    rays_bsi.push_back(x->multiplyByConstant(static_cast<int>(front_Kinvs(0,0)))->SUM(static_cast<long>(front_Kinvs(0,2))));
    rays_bsi.push_back(x->multiplyByConstant(static_cast<int>(front_Kinvs(1,0)))->SUM(static_cast<long>(front_Kinvs(1,2))));
    rays_bsi.push_back(x->multiplyByConstant(static_cast<int>(front_Kinvs(2,0)))->SUM(static_cast<long>(front_Kinvs(2,2))));

    for (int i = hpz; i < result->depth.size() - hpz; ++i) {
        // initialize depth
        BsiAttribute<uint64_t>* depth = exp(UniformRand(log(min_depth_), log(max_depth_), image_width));
        delete result->depth[i];
        result->depth[i] = depth;

        // // generate normal
        std::vector<BsiAttribute<uint64_t>*> normal;
        normal.push_back(UniformRand(-1, 1, image_width));
        normal.push_back(UniformRand(-1, 1, image_width));
        normal.push_back(bsi.buildBsiAttributeFromVectorSigned(normal_z,0.5));

        // // initialize plane
        std::vector<BsiAttribute<uint64_t>*> plane = PlaneFromDepthAndNormal(i, result->depth[i], normal, rays_bsi);
        delete result->plane[i][0];
        delete result->plane[i][1];
        delete result->plane[i][2];
        result->plane[i][0] = plane[0];
        result->plane[i][1] = plane[1];
        result->plane[i][2] = plane[2];

        // // initialize nghbr and score
        BsiAttribute<uint64_t>* nghbr;
        BsiAttribute<uint64_t>* score;
        if (sample) {
            nghbr = bsi.createRandomBsi(image_width,images_bsi.size(),0.5);
            // TODO: for now, set nghbr for one row of pixels in an image to an integer, check later if choosing a diff nghbr for each pixel is better
            score = ComputePlaneImageScore(i, plane, nghbr, image_width);
        } else {
            // TODO: don't implement for now, focus on compute patch match sample
        //    ComputePlaneScore(i, plane, &score, &nghbr);
        }
        delete result->nghbr[i];
        delete result->score[i];

        result->nghbr[i] = nghbr;
        result->score[i] = score;

        delete normal[0];
        delete normal[1];
        delete normal[2];
        break;
    }
    delete rays_bsi[0];
    delete rays_bsi[1];
    delete rays_bsi[2];
    delete x;
}

void BsiDepthmapEstimator::ComputeIgnoreMask(BsiDepthmapEstimatorResult *result) {}
/*    int hpz = (patch_size_ - 1) / 2;
    for (int i = hpz; i < result->depth.size() - hpz; ++i) {
        // masked represents a vector of 1's and 0's, where 0 means not masked
        HybridBitmap<uint64_t> masked = mask_[i];
        HybridBitmap<uint64_t> low_variance = PatchVariance(i)->reLU(min_patch_variance_ * (patch_size_ * patch_size_));
        // TODO: check if this correctly initializes
        AssignPixelRow(result, i, new BsiSigned<uint64_t>(result->depth[0]->rows),
                {new BsiSigned<uint64_t>(result->depth[0]->rows),new BsiSigned<uint64_t>(result->depth[0]->rows),new BsiSigned<uint64_t>(result->depth[0]->rows)},
                new BsiSigned<uint64_t>(result->depth[0]->rows), new BsiSigned<uint64_t>(result->depth[0]->rows), masked.Or(low_variance));
    }
}*/

BsiAttribute<uint64_t>* BsiDepthmapEstimator::PatchVariance(int i) {
    BsiAttribute<uint64_t>* patch_sum = images_bsi[0].at(i);
    int hpz = (patch_size_ - 1) / 2;
    for (int u = -hpz; u <= hpz; ++u) {
        for (int v = -hpz; v <= hpz; ++v) {
            patch_sum = (*patch_sum) + images_bsi[0].at(i + u)->shift(v);
        }
    }
    
    BsiAttribute<uint64_t>* mean;
    BsiAttribute<uint64_t>* variance;
    // TODO: uncomment when implemented / for constants
    // BsiAttribute<uint64_t>* mean = *patch_sum / (patch_size_ * patch_size_);
    // TODO: uncomment when implemented - for constants
    // BsiAttribute<uint64_t>* variance = patch_sum - mean;
    BsiAttribute<uint64_t>* variance_squared = *variance * variance;
    return variance_squared;
}

void BsiDepthmapEstimator::AssignPixelRow(BsiDepthmapEstimatorResult *result, int i,
                    BsiAttribute<uint64_t>* depth, std::vector<BsiAttribute<uint64_t>*> &plane,
                    BsiAttribute<uint64_t>* score, BsiAttribute<uint64_t>* nghbr, const HybridBitmap<uint64_t> &mask) {
    BsiAttribute<uint64_t>* prev_depth = result->depth[i];
    BsiAttribute<uint64_t>* prev_score = result->score[i];
    BsiAttribute<uint64_t>* prev_nghbr = result->nghbr[i];
    BsiAttribute<uint64_t>* prev_plane1 = result->plane[i][0];
    BsiAttribute<uint64_t>* prev_plane2 = result->plane[i][1];
    BsiAttribute<uint64_t>* prev_plane3 = result->plane[i][2];
    result->depth[i] = depth->maskAssign(result->depth[i],mask);
    result->score[i] = score->maskAssign(result->score[i],mask);
    result->nghbr[i] = nghbr->maskAssign(result->nghbr[i],mask);
    result->plane[i][0] = plane[0]->maskAssign(result->plane[i][0],mask);
    result->plane[i][1] = plane[1]->maskAssign(result->plane[i][1],mask);
    result->plane[i][2] = plane[2]->maskAssign(result->plane[i][2],mask);
    delete prev_depth;
    delete prev_score;
    delete prev_nghbr;
    delete prev_plane1;
    delete prev_plane2;
    delete prev_plane3;
}

void BsiDepthmapEstimator::ComputePatchMatchSample(BsiDepthmapEstimatorResult *result) {
    AssignMatrices(result);
    RandomInitialization(result, true);
    // ComputeIgnoreMask(result);
    
    // for (int i = 0; i < patchmatch_iterations_; ++i) {
    //     PatchMatchForwardPass(result, true);
    //     PatchMatchBackwardPass(result, true); // TODO
    // }
    
    // PostProcess(result);
}

void BsiDepthmapEstimator::PatchMatchForwardPass(BsiDepthmapEstimatorResult *result,
                                              bool sample) {
    int adjacent[2][2] = {{-1, 0}, {0, -1}};
    int hpz = (patch_size_ - 1) / 2;
    for (int i = hpz; i < result->depth.size() - hpz; ++i) {
        PatchMatchUpdatePixelRow(result, i, adjacent, sample);
    }
}

void BsiDepthmapEstimator::PatchMatchBackwardPass(BsiDepthmapEstimatorResult *result,
    bool sample) {}

void BsiDepthmapEstimator::PatchMatchUpdatePixelRow(BsiDepthmapEstimatorResult *result,
                                              int i, int adjacent[2][2],
                                              bool sample) {}
/*    // Ignore pixels with depth == 0.
    // TODO: mask convolution at the end with relu
    HybridBitmap<uint64_t> mask = result->depth.at(i)->reLU(0).Not();

    // Check neighbors and their best match to see if it is also this pixel's best match
    for (int k = 0; k < 2; ++k) {
        int i_adjacent = i + adjacent[k][0];

        // Do not propagate ignored adjacent pixels.
        // TODO: figure out how to mask convolution at the end with relu
        // result->depth.at(i_adjacent)->relu(0)->Not();
        // if (result->depth[i_adjacent] == 0.0f) {
        //     continue;
        // }

        std::vector<BsiAttribute<uint64_t>*> plane;
        for (auto p: result->plane[i_adjacent]) {
            plane.push_back(p->shift(adjacent[k][1]));
        }

        if (sample) {
            BsiAttribute<uint64_t>* nghbr = result->nghbr[i_adjacent]->shift(adjacent[k][1]);
            CheckPlaneImageCandidate(result, i, plane, nghbr);
        } else {
            // TODO: don't implement for now and focus on sample
//            CheckPlaneCandidate(result, i, j, plane);
        }
    }

    // Perturb depth and plane guesses to see which is the best match with the other image
    float depth_range = 0.02;
    float normal_range = 0.5;
    BsiAttribute<uint64_t>* current_nghbr = result->nghbr.at(i);
    for (int k = 0; k < 6; ++k) {
        BsiAttribute<uint64_t>* current_depth = result->depth.at(i);
        // Similar to log normal distribution of perturbation to depth
        // TODO: replace exp, unit_normal_, rng_
        // TODO: uncomment
        BsiAttribute<uint64_t>* depth;
        // BsiAttribute<uint64_t>* depth = current_depth * exp(depth_range * unit_normal_(rng_));

        std::vector<BsiAttribute<uint64_t>*> current_plane = result->plane.at(i);
        // TODO: mask with relu
        // if (current_plane[2] == 0.0) {
        //     continue;
        // }
        // normal distribution of perturbation to normal
        // TODO: uncomment
        std::vector<BsiAttribute<uint64_t>*> normal;
        // std::vector<BsiAttribute<uint64_t>*> normal(-current_plane[0] / current_plane[2] +
        //                  normal_range * unit_normal_(rng_),
        //                  -current_plane[1] / current_plane[2] +
        //                  normal_range * unit_normal_(rng_),
        //                  -1.0f);

        std::vector<BsiAttribute<uint64_t>*> plane = PlaneFromDepthAndNormal(i, Ks_[0], depth, normal);
        if (sample) {
            CheckPlaneImageCandidate(result, i, plane, current_nghbr);
        } else {
//            CheckPlaneCandidate(result, i, j, plane);
        }

        depth_range *= 0.3;
        normal_range *= 0.8;
    }

    if (!sample || images_.size() <= 2) {
        return;
    }

    // Check random other image to escape local minima
    BsiAttribute<uint64_t>* other_nghbr = uni_(rng_);
    // TODO: subtract other_nghbr and current_nghbr and check if any of the elements are zero (NAND)
    while (other_nghbr == current_nghbr) {
        // other_nghbr = uni_(rng_);
    }

    std::vector<BsiAttribute<uint64_t>*> plane = result->plane.at(i);
    CheckPlaneImageCandidate(result, i, plane, other_nghbr);
}*/

/*void BsiDepthmapEstimator::CheckPlaneImageCandidate(
        BsiDepthmapEstimatorResult *result, int i, std::vector<BsiAttribute<uint64_t>*> &plane,
        BsiAttribute<uint64_t>* nghbr) {
    BsiAttribute<uint64_t>* score = ComputePlaneImageScore(i, plane, nghbr);
    // TODO: implement relu between bsi
    if (score > result->score.at(i)) {
        BsiAttribute<uint64_t>* depth = DepthOfPlaneBackprojection(i, Ks_[0], plane);
        HybridBitmap<uint64_t> mask;
        AssignPixelRow(result, i, depth, plane, score, nghbr,mask);
    }
}*/

/*BsiAttribute<uint64_t>* DepthOfPlaneBackprojection(int y,
                                                   const std::vector<std::vector<BsiAttribute<uint64_t>*>> &K,
                                                   const std::vector<BsiAttribute<uint64_t>*> &plane) {
    std::vector<BsiAttribute<uint64_t>*> homogeneous_coord;
    BsiSigned<uint64_t> bsi;
    std::vector<long> ones(x->rows, 1);
    homogeneous_coord.push_back(x);
    homogeneous_coord.push_back(y);
    homogeneous_coord.push_back(bsi.buildBsiAttributeFromVectorSigned(ones,0.5));
    // TODO: implement matrix mult
    BsiAttribute<uint64_t>* denom = -(plane.t() * K.inv() * homogeneous_coord)[0];
    return 1.0f / std::max(1e-6f, denom);
}*/

BsiAttribute<uint64_t>* BsiDepthmapEstimator::ComputePlaneImageScore(int i,
                                                const std::vector<BsiAttribute<uint64_t>*> &plane,
                                                BsiAttribute<uint64_t>* other, int image_width) {
    // calculate all homography matrices for the ith row
    std::vector<std::vector<BsiAttribute<uint64_t>*>> H = PlaneInducedHomographyBaked(other, PIHB_precalc1, PIHB_precalc2, plane, front_Kinvs);
    int hpz = (patch_size_ - 1) / 2;
    BsiSigned<uint64_t> bsi;
    std::vector<long> x_coord;
    for (int i=0; i<image_width; i++) {
        x_coord.push_back(i);
    }
    BsiAttribute<uint64_t>* j = bsi.buildBsiAttributeFromVectorSigned(x_coord,0.5);
    
    BsiAttribute<uint64_t>* u = ((*H[0][0]) * j)->SUM ((*H[0][1]) * i) ->SUM (H[0][2]);
    BsiAttribute<uint64_t>* v = ((*H[1][0]) * j)->SUM ((*H[1][1]) * i) ->SUM (H[1][2]);
    BsiAttribute<uint64_t>* w = ((*H[2][0]) * j)->SUM ((*H[2][1]) * i) ->SUM (H[2][2]);
    return j;

    // TODO: use relu to make mask, then use mask to avoid adding to ncc result
    /*if (w == 0.0) {
        return -1.0f;
    }

    //du/dx
    BsiAttribute<uint64_t>* dfdx_x = ((*H[0][0]) * w - (*H[2][0]) * u) / (w * w);
    //du/dy
    BsiAttribute<uint64_t>* dfdx_y = ((*H[1][0]) * w - (*H[2][0]) * v) / (w * w);
    //dv/dx
    BsiAttribute<uint64_t>* dfdy_x = ((*H[0][1]) * w - (*H[2][1]) * u) / (w * w);
    //dv/dy
    BsiAttribute<uint64_t>* dfdy_y = ((*H[1][1]) * w - (*H[2][1]) * v) / (w * w);

    // homogeneous coordinates of center of corresponding patch in other image
    BsiAttribute<uint64_t>* Hx0 = u / w;
    BsiAttribute<uint64_t>* Hy0 = v / w;

    BsiAttribute<uint64_t>* im1_center = images_[0][i];

    BsiNCCEstimator ncc;
    for (int dy = -hpz; dy <= hpz; ++dy) {
        for (int dx = -hpz; dx <= hpz; ++dx) {
            BsiAttribute<uint64_t>* im1 = images_[0].at(i + dy).shift(dx);
            // subpixel coordinates of (i+dy,j+dx) in images_[other]
            BsiAttribute<uint64_t>* x2 = Hx0 + dfdx_x * dx + dfdy_x * dy;
            BsiAttribute<uint64_t>* y2 = Hy0 + dfdx_y * dx + dfdy_y * dy;
            BsiAttribute<uint64_t>* im2 = LinearInterpolation<unsigned char>(images_[other], y2, x2);
            BsiAttribute<uint64_t>* weight = BilateralWeight(im1 - im1_center, dx, dy);
            ncc.Push(im1, im2, weight);
        }
    }
    // TODO: mask for hpz size borders
    return ncc.Get();*/
}

/*
Multiply 2 vectors of BSI's by treating the first as 3x1, and the second as 1x3
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> multTranspose(const std::vector<BsiAttribute<uint64_t>*> &a, const std::vector<BsiAttribute<uint64_t>*> &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<a.size();i++) {
        BsiAttribute<uint64_t>* a_bsi = a[i];
        res.push_back({});
        for (BsiAttribute<uint64_t>* b_bsi: b) {
            res[i].push_back(a_bsi->multiplyWithBsiHorizontal(b_bsi));
        }
    }
    return res;
}

/*
Multiply a vector of constants with a vector of BSI's by treating the first as 3x1, and the second as 1x3
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> multTranspose(const std::vector<int> &a, const std::vector<BsiAttribute<uint64_t>*> &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<a.size();i++) {
        res.push_back({});
        for (BsiAttribute<uint64_t>* b_bsi: b) {
            res[i].push_back(b_bsi->multiplyByConstant(a[i]));
        }
    }
    return res;
}

/*
Add a 3x3 matrix of BSI's with a 3x3 matrix of BSI's
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> addMatrices(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<a.size();i++) {
        res.push_back({});
        for (int j=0;j<a[i].size();j++) {
            res[i].push_back(a[i][j]->SUM(b[i][j]));
        }
    }
    return res;
}

/*
Add a 3x3 matrix of constants with a 3x3 matrix of BSI's
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> addMatrices(const std::vector<std::vector<int>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<a.size();i++) {
        res.push_back({});
        for (int j=0;j<3;j++) {
            res[i].push_back(b[i][j]->SUM(static_cast<long>(a[i][j])));
        }
    }
    return res;
}

/*
Multiply a 3x3 matrix of BSI's with a 3x3 matrix of BSI's
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> multMatrices(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<std::vector<BsiAttribute<uint64_t>*>> &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<a.size();i++) {
        res.push_back({});
        for (int j=0;j<b[i].size();j++) {
            BsiAttribute<uint64_t>* res_ij = a[i][0]->multiplyWithBsiHorizontal(b[0][j]);
            for (int k=1; k<a[i].size(); k++) {
                res_ij = res_ij->SUM(a[i][k]->multiplyWithBsiHorizontal(b[k][j]));
            }
            res[i].push_back(res_ij);
        }
    }
    return res;
}

/*
Multiply a 3x3 matrix of BSI's with a 3x3 matrix of constants
*/
std::vector<std::vector<BsiAttribute<uint64_t>*>> multMatrixWithConstants(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const cv::Matx33d &b) {
    std::vector<std::vector<BsiAttribute<uint64_t>*>> res;
    for (int i=0;i<3;i++) {
        res.push_back({});
        for (int j=0;j<3;j++) {
            BsiAttribute<uint64_t>* res_ij = a[i][0]->multiplyByConstant(b(0,j));
            for (int k=1; k<a[i].size(); k++) {
                res_ij = res_ij->SUM(a[i][k]->multiplyByConstant(static_cast<int>(b(k,j))));
            }
            res[i].push_back(res_ij);
        }
    }
    return res;
}

/*
Multiply a 3x3 matrix of BSI's with a 3x1 vector of BSI's
*/
std::vector<BsiAttribute<uint64_t>*> multMatrixWithVector(const std::vector<std::vector<BsiAttribute<uint64_t>*>> &a, const std::vector<BsiAttribute<uint64_t>*> &b) {
    std::vector<BsiAttribute<uint64_t>*> res;
    for (int i=0;i<3;i++) {
        BsiAttribute<uint64_t>* res_i = a[i][0]->multiplyWithBsiHorizontal(b[0]);
        for (int k=1; k<3; k++) {
            res_i = res_i->SUM(a[i][k]->multiplyWithBsiHorizontal(b[k]));
        }
        res.push_back(res_i);
    }
    return res;
}

/*
Multiply a 1x3 vector of BSI's with a 3x3 matrix of constants
*/
std::vector<BsiAttribute<uint64_t>*> multVectorWithConstants(const std::vector<BsiAttribute<uint64_t>*> &a, const cv::Matx33d &b) {
    std::vector<BsiAttribute<uint64_t>*> res;
    for (int i=0;i<3;i++) {
        BsiAttribute<uint64_t>* res_i = a[0]->multiplyByConstant(b(0,i));
        for (int j=1;j<3;j++) {
            res_i = res_i->SUM(a[j]->multiplyByConstant(static_cast<int>(b(j,i))));
        }
        res.push_back(res_i);
    }
    return res;
}

std::vector<std::vector<BsiAttribute<uint64_t>*>> PlaneInducedHomographyBaked(BsiAttribute<uint64_t>* other, std::vector<std::vector<BsiAttribute<uint64_t>*>> PIHB_precalc1,
    std::vector<BsiAttribute<uint64_t>*> PIHB_precalc2, const std::vector<BsiAttribute<uint64_t>*> &v, cv::Matx33d front_Kinvs) {
    /*
    K2 * (Q2 + a2*v.T) * K1inv can be broken down into parts that are precalculated, then just need to calculate the part with v
    PIHB_precalc1 = K2 * Q2 * K1inv
    PIHB_precalc2 = K2 * a2
    K2 * (Q2 + a2*v.T) * K1inv = PIHB_precalc1 + PIHB_precalc2 * (v.T * K1inv)
    */
    return addMatrices(PIHB_precalc1,multTranspose(PIHB_precalc2,multVectorWithConstants(v,front_Kinvs)));
}

/*BsiAttribute<uint64_t>* LinearInterpolation(std::vector<BsiAttribute<uint64_t>*> &image, BsiAttribute<uint64_t>* y, BsiAttribute<uint64_t>* x) {
    // TODO: mask with relu
    if (x < 0.0f || x >= image[0]->rows - 1 || y < 0.0f || y >= image.size() - 1) {
        return nullptr;
    }
    // TODO: implement getting integer (shifting by precision) and decimal (chop off) part of bsi
    BsiAttribute<uint64_t>* iy = get_int_part(y);
    BsiAttribute<uint64_t>* dx = get_decimal_part(x);
    BsiAttribute<uint64_t>* dy = get_decimal_part(y);
    BsiAttribute<uint64_t>* im00 = image.at(iy);
    BsiAttribute<uint64_t>* im01 = image.at(iy).shift(1);
    BsiAttribute<uint64_t>* im10 = image.at(iy + 1);
    BsiAttribute<uint64_t>* im11 = image.at(iy + 1).shift(1);
    BsiAttribute<uint64_t>* im0 = (1 - dx) * im00 + dx * im01;
    BsiAttribute<uint64_t>* im1 = (1 - dx) * im10 + dx * im11;
    return (1 - dy) * im0 + dy * im1;
}*/

/*BsiAttribute<uint64_t>* BsiDepthmapEstimator::BilateralWeight(BsiAttribute<uint64_t>* dcolor, float dx, float dy) {
    const float dcolor_sigma = 50.0f;
    const float dx_sigma = 5.0f;
    const float dcolor_factor = 1.0f / (2 * dcolor_sigma * dcolor_sigma);
    const float dx_factor = 1.0f / (2 * dx_sigma * dx_sigma);
    // TODO: replace exp
    return exp(-dcolor * dcolor * dcolor_factor -
               (dx * dx + dy * dy) * dx_factor);
}*/


BsiNCCEstimator::BsiNCCEstimator()
        : sumx_(0), sumy_(0), sumxx_(0), sumyy_(0), sumxy_(0), sumw_(0) {} // TODO: initializer list with empty bsi

BsiNCCEstimator::~BsiNCCEstimator() {
    delete sumx_;
    delete sumy_;
    delete sumxx_;
    delete sumxy_;
    delete sumyy_;
    delete sumw_;
}

/*void BsiNCCEstimator::Push(BsiAttribute<uint64_t>* x, BsiAttribute<uint64_t>* y, BsiAttribute<uint64_t>* w) {
    sumx_ += w * x;
    sumy_ += w * y;
    sumxx_ += w * x * x;
    sumyy_ += w * y * y;
    sumxy_ += w * x * y;
    sumw_ += w;
}*/

/*BsiAttribute<uint64_t>* BsiNCCEstimator::Get() {
    // TODO: mask with relu
    if (sumw_ == 0.0) {
        return -1;
    }
    BsiAttribute<uint64_t>* meanx = sumx_ / sumw_;
    BsiAttribute<uint64_t>* meany = sumy_ / sumw_;
    BsiAttribute<uint64_t>* meanxx = sumxx_ / sumw_;
    BsiAttribute<uint64_t>* meanyy = sumyy_ / sumw_;
    BsiAttribute<uint64_t>* meanxy = sumxy_ / sumw_;
    BsiAttribute<uint64_t>* varx = meanxx - meanx * meanx;
    BsiAttribute<uint64_t>* vary = meanyy - meany * meany;
    // TODO: mask with relu
    if (varx < 0.1 || vary < 0.1) {
        return -1;
    } else {
        // covariance between intensities in the overlap of x and y / standard deviation of intensities
        // normalized by standard deviation to prevent change in brightness from affecting score
        return (meanxy - meanx * meany) / sqrt(varx * vary);
    }
}*/

void BsiDepthmapEstimator::PostProcess(BsiDepthmapEstimatorResult *result) {}
/*    std::vector<BsiAttribute<uint64_t>*> depth_filtered;
    cv::medianBlur(result->depth, depth_filtered, 5); // TODO: convolution - take median of kernel size 5

    for (int i = 0; i < result->depth.size(); ++i) {
        BsiAttribute<uint64_t>* d = result->depth.at(i);
        BsiAttribute<uint64_t>* m = depth_filtered.at(i);
        // TODO: mask with relu
        if (d == 0.0 || fabs(d - m) / d > 0.05) {
            result->depth.at(i) = 0;
        }
    }
}*/

}