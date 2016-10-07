#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
#ifdef USE_MPI
  //advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
  BaseDataLayer<Dtype>::OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
#endif
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void FrcnnPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
    // Reshape to loaded rois.
    top[2]->ReshapeLike(prefetch_rois_);
    // Copy the rois.
    caffe_copy(prefetch_rois_.count(), prefetch_rois_.cpu_data(),
        top[2]->mutable_gpu_data());
    // Reshape to loaded bbox_targets.
    top[3]->ReshapeLike(prefetch_bbox_targets_);
    // Copy the bbox_targets.
    caffe_copy(prefetch_bbox_targets_.count(), prefetch_bbox_targets_.cpu_data(),
        top[3]->mutable_gpu_data());
    // Reshape to loaded bbox_weights.
    top[4]->ReshapeLike(prefetch_bbox_weights_);
    // Copy the bbox_weights.
    caffe_copy(prefetch_bbox_weights_.count(), prefetch_bbox_weights_.cpu_data(),
        top[4]->mutable_gpu_data());
  }
  
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(FrcnnPrefetchingDataLayer);

}  // namespace caffe

