#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ExtractLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const ExtractParameter& param = this->layer_param_.extract_param();
  CHECK_GE(param.x(), 0) << "x should not be negative";
  CHECK_GE(param.y(), 0) << "y should not be negative";
  CHECK_GE(param.h(), 1) << "h should be greater than 1";
  CHECK_GE(param.w(), 1) << "w should be greater than 1";
  hstart_ = param.y();
  wstart_ = param.x();
  hend_ = hstart_ + param.h();
  wend_ = wstart_ + param.w();
  // fixed-size allocations
  vector<int> new_shape;
  new_shape.push_back(bottom[0]->num());
  new_shape.push_back(bottom[0]->channels());
  new_shape.push_back(param.h());
  new_shape.push_back(param.w());
  top[0]->Reshape(new_shape);
}

template <typename Dtype>
void ExtractLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_LE(hend_, bottom[0]->height()) << "Exceeds boundaries";
  CHECK_LE(wend_, bottom[0]->width()) << "Exceeds boundaries";
}

template <typename Dtype>
void ExtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int w = 0; w < bottom[0]->width(); ++w) {
          if (h >= hstart_ && h < hend_ && w >= wstart_ && w < wend_) {
            int bottom_index = h * bottom[0]->width() + w;
            int top_index = (h - hstart_) * (wend_ - wstart_) + (w - wstart_);
            top_data[top_index] = bottom_data[bottom_index];
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void ExtractLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int w = 0; w < bottom[0]->width(); ++w) {
          if (h >= hstart_ && h < hend_ && w >= wstart_ && w < wend_) {
            int bottom_index = h * bottom[0]->width() + w;
            int top_index = (h - hstart_) * (wend_ - wstart_) + (w - wstart_);
            bottom_diff[bottom_index] = top_diff[top_index];
          }
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ExtractLayer);
#endif

INSTANTIATE_CLASS(ExtractLayer);
REGISTER_LAYER_CLASS(Extract);

}  // namespace caffe
