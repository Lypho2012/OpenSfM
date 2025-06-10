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
  }

  void ProcessViews() { bde_.ProcessViewsToBsi(); }

  void SetDepthRange(double min_depth, double max_depth, int num_depth_planes) {
    bde_.SetDepthRange(min_depth, max_depth, num_depth_planes);
  }

  void SetPatchMatchIterations(int n) { bde_.SetPatchMatchIterations(n); }

  void SetPatchSize(int size) { bde_.SetPatchSize(size); }

  void SetMinPatchSD(float sd) { bde_.SetMinPatchSD(sd); }

  py::object ComputePatchMatchSample() {
    BsiDepthmapEstimatorResult result;
    {
      py::gil_scoped_release release;
      bde_.ComputePatchMatchSample(&result);
    }
    return ComputeReturnValues(result);
  }

  py::object ComputeReturnValues(const BsiDepthmapEstimatorResult &result) {
    py::list retn;
    py::array_t<long> depth({static_cast<size_t>(result.depth.size()),static_cast<size_t>(result.depth[0]->rows)});
    py::array_t<long> plane({static_cast<size_t>(result.plane.size()),static_cast<size_t>(result.plane[0][0]->rows),static_cast<size_t>(3)});
    py::array_t<long> score({static_cast<size_t>(result.score.size()),static_cast<size_t>(result.score[0]->rows)});
    py::array_t<long> nghbr({static_cast<size_t>(result.nghbr.size()),static_cast<size_t>(result.nghbr[0]->rows)});

    for (int i=0; i<result.depth.size(); i++) {
        for (int j=0; j<result.depth[i]->rows; j++) {
            depth.mutable_at(i,j) = result.depth[i]->getValue(j);
        }
    }
    for (int i=0; i<result.plane.size(); i++) {
        for (int k=0; k<3; k++) {
            for (int j=0; j<result.plane[i][k]->rows; j++) {
                plane.mutable_at(i,j,k) = result.plane[i][k]->getValue(j);
            }
        }
    }
    for (int i=0; i<result.score.size(); i++) {
        for (int j=0; j<result.score[i]->rows; j++) {
            score.mutable_at(i,j) = result.score[i]->getValue(j);
        }
    }
    for (int i=0; i<result.nghbr.size(); i++) {
        for (int j=0; j<result.nghbr[i]->rows; j++) {
            nghbr.mutable_at(i,j) = result.nghbr[i]->getValue(j);
        }
    }
    retn.append(depth);
    retn.append(plane);
    retn.append(score);
    retn.append(nghbr);
    return std::move(retn);
  }

 private:
  BsiDepthmapEstimator bde_;
};


}  // namespace bsidense
