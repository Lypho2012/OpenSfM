#pragma once

#include <dense/bsi_depthmap.h>
#include <foundation/python_types.h>

using namespace foundation;

namespace bsidense {

class BsiDepthmapEstimatorWrapper {
 public:
  void AddView(pyarray_d K, pyarray_d R, pyarray_d t, pyarray_uint8 image,
               pyarray_uint8 mask) {
    if ((image.shape(0) != mask.shape(0)) || (image.shape(1) != mask.shape(1))){
      throw std::invalid_argument("image and mask must have matching shapes.");
    }
    double *pK = K.data();
    double *pR = R.data();
    double *pt = t.data();
    unsigned char *pimage = image.data();
    unsigned char *pmask = mask.data();
    std::vector<double> K_vec(pK,pK+K.shape(0));
    std::vector<double> R_vec(pR,pR+R.shape(0));
    std::vector<double> t_vec(pt,pt+t.shape(0));
    std::vector<unsigned char> image_vec(pimage,pimage+image.shape(0));
    std::vector<unsigned char> mask_vec(pmask,pmask+mask.shape(0));
    bde_.AddView(K_vec, R_vec, t_vec, image_vec, mask_vec,
                image.shape(1), image.shape(0));
    delete pK;
    delete pR;
    delete pt;
    delete pimage;
    delete pmask;
  }

  void SetDepthRange(double min_depth, double max_depth, int num_depth_planes) {
    bde_.SetDepthRange(min_depth, max_depth, num_depth_planes);
  }

 private:
  BsiDepthmapEstimator bde_;
};


}  // namespace bsidense
