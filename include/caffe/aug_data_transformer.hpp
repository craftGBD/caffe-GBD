#ifndef CAFFE_AUG_DATA_TRANSFORMER_HPP
#define CAFFE_AUG_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
class AugDataTransformer {
 public:
  explicit AugDataTransformer(const AugTransformationParameter& param, Phase phase);
  virtual ~AugDataTransformer() {}

  void InitRand();

  void Transform(const cv::Mat& cv_img, const vector<Dtype>& cv_points, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_blob2);
  
  protected:
   virtual unsigned Rand(unsigned n);
   virtual double m_rand_gen(const MRNGParameter& para, double &value);
   
   AugTransformationParameter param_;

   shared_ptr<Caffe::RNG> rng_;

   Phase phase_;
};

}
#endif

