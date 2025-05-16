#pragma once

#include <dense/bsi_depthmap.h>
#include <foundation/python_types.h>

using namespace foundation;

namespace bsidense {

class BsiDepthmapEstimatorWrapper {
 public:
  void InitializeViews(size_t num_images) { bde_.InitializeViews(num_images); }

  void AddView(pyarray_d K, pyarray_d R, pyarray_d t, pyarray_uint8 image,
               pyarray_uint8 mask) {
    if ((image.shape(0) != mask.shape(0)) || (image.shape(1) != mask.shape(1))){
      throw std::invalid_argument("image and mask must have matching shapes.");
    }
    const double *pK = K.data();
    const double *pR = R.data();
    const double *pt = t.data();
    const unsigned char *pimage = image.data();
    const unsigned char *pmask = mask.data();
    bde_.AddView(pK, pR, pt, pimage, pmask,
                (size_t)image.shape(1), (size_t)image.shape(0));
    delete pK;
    delete pR;
    delete pt;
    delete pimage;
    delete pmask;
  }

  void ProcessViews() { bde_.ProcessViews(); }

  void SetDepthRange(double min_depth, double max_depth, int num_depth_planes) {
    bde_.SetDepthRange(min_depth, max_depth, num_depth_planes);
  }

  void SetPatchMatchIterations(int n) { bde_.SetPatchMatchIterations(n); }

  void SetPatchSize(int size) { bde_.SetPatchSize(size); }

  void SetMinPatchSD(float sd) { bde_.SetMinPatchSD(sd); }

 private:
  BsiDepthmapEstimator bde_;
};


}  // namespace bsidense
