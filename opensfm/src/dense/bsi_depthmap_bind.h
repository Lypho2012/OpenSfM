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

    cv::Mat depth = cv::Mat(result.depth.size(), result.depth[0]->rows, CV_32F, 0.0f);
    cv::Mat plane = cv::Mat(result.plane.size(), result.plane[0][0]->rows, CV_32FC3, 0.0f);
    cv::Mat score = cv::Mat(result.score.size(), result.score[0]->rows, CV_32F, 0.0f);
    cv::Mat nghbr = cv::Mat(result.nghbr.size(), result.nghbr[0]->rows, CV_32S, cv::Scalar(0));

    for (int i=0; i<result.depth.size(); i++) {
        for (int j=0; j<result.depth[i]->rows; j++) {
            depth.at<float>(i,j) = result.depth[i]->getValue(j);
        }
    }
    for (int i=0; i<result.plane.size(); i++) {
        for (int k=0; k<3; k++) {
            for (int j=0; j<result.plane[i][k]->rows; j++) {
                plane.at<float>(i,j,k) = result.plane[i][k]->getValue(j);
            }
        }
    }
    for (int i=0; i<result.score.size(); i++) {
        for (int j=0; j<result.score[i]->rows; j++) {
            score.at<float>(i,j) = result.score[i]->getValue(j);
        }
    }
    for (int i=0; i<result.nghbr.size(); i++) {
        for (int j=0; j<result.nghbr[i]->rows; j++) {
            nghbr.at<int>(i,j) = result.nghbr[i]->getValue(j);
        }
    }

    retn.append(py_array_from_data(depth.ptr<float>(0),
                                   depth.rows, depth.cols));
    retn.append(py_array_from_data(plane.ptr<float>(0),
                                   plane.rows, plane.cols, 3));
    retn.append(py_array_from_data(score.ptr<float>(0),
                                   score.rows, score.cols));
    retn.append(py_array_from_data(nghbr.ptr<int>(0), nghbr.rows,
                                   nghbr.cols));
    return std::move(retn);
  }

 private:
  BsiDepthmapEstimator bde_;
};


}  // namespace bsidense
