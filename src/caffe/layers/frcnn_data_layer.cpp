///////////////////////////////////////////////////////////////////
// Fast RCNN DataLayer
//
// Feature         Support multigpu training
// Usage           Load roidb, bbox_targets from rois_file to
//                 avoid expensive computation.
//                 Roidb and bbox_targets can be computed using
//                 Fast-RCNN python codes and saved to rois_file
//                 with following format:
//
//                 rois_file format
//                 repeated:
//                    # image_index
//                    img_path
//                    channels
//                    height
//                    width
//                    flipped
//                    num_windows
//                    class_index overlap x1 y1 x2 y2 dx dy dw dh
//
//                  Note: image_index, class_index and x1, y1, x2,
//                  y2 are 0-indexed.
//
// Written by      Yang Bin, Wang Kun
// Last edit       2016.3.14
///////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <stdint.h>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > FrcnnDataParameter
//   'source'          where the rois_file is
//   'root_folder'     root folder where images data store
//   'min_size'        minimum size of re-scaled image
//   'max_size'        maximum size of re-scaled image
//   'fg_thresh'       overlap in [fg_thresh,1] is positive samples
//   'bg_thresh_hi'    overlap in [bg_thresh_lo, bg_thresh_hi) is negative samples
//   'bg_thresh_lo'    overlap in [bg_thresh_lo, bg_thresh_hi) is negative samples
//   'fg_fraction'     proportion of positive samples in a mini-batch
//   'batch_size'      number of rois on a gpu
//   'ims_per_batch'   number of ims on a gpu
//   'num_class'       number of classes
//   'rand_seed'       random seed for data shuffle
//   'rand_skip'       random skip some data from the head of the dataset

namespace caffe {

template <typename Dtype>
FrcnnDataLayer<Dtype>::~FrcnnDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void FrcnnDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the rois_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // 0. read parameters
  // thread
  const int thread_id = Caffe::MPI_my_rank();
  const string source_file = this->layer_param_.frcnn_data_param().source();
  const string root_folder = this->layer_param_.frcnn_data_param().root_folder();
  const int max_size = this->layer_param_.frcnn_data_param().max_size();
  const int ims_per_batch = this->layer_param_.frcnn_data_param().ims_per_batch();
  const int ims_per_gpu = ims_per_batch;
  const int batch_size = this->layer_param_.frcnn_data_param().batch_size();
  const int num_class = this->layer_param_.frcnn_data_param().num_class();
  const int rand_seed = this->layer_param_.frcnn_data_param().rand_seed();

  // 1. read rois_file
  std::ifstream infile(source_file.c_str());
  CHECK(infile.good()) << "Failed to open window file " << source_file << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index;
  int channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // get image info
    vector<int> image_info(4);
    infile >> image_info[0] >> image_info[1] >> image_info[2] >> image_info[3];
    image_database_.push_back(std::make_pair(image_path, image_info));
    channels = image_info[0];

    // read each box
    int num_windows;
    infile >> num_windows;
    // skip the image that does not have proposals
    if (num_windows <= 0) {
      LOG(INFO) << image_path << " does not have proposals!";
      continue;
    }
    vector<vector<float> > rois_list;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap, dx, dy, dw, dh;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >> dx >> dy >> dw >> dh;

      vector<float> window(FrcnnDataLayer<Dtype>::NUM);
      window[FrcnnDataLayer<Dtype>::LABEL] = label;
      window[FrcnnDataLayer<Dtype>::OVERLAP] = overlap;
      window[FrcnnDataLayer<Dtype>::X1] = x1;
      window[FrcnnDataLayer<Dtype>::Y1] = y1;
      window[FrcnnDataLayer<Dtype>::X2] = x2;
      window[FrcnnDataLayer<Dtype>::Y2] = y2;
      window[FrcnnDataLayer<Dtype>::DX] = dx;
      window[FrcnnDataLayer<Dtype>::DY] = dy;
      window[FrcnnDataLayer<Dtype>::DW] = dw;
      window[FrcnnDataLayer<Dtype>::DH] = dh;

      // add box to rois list
      label = window[FrcnnDataLayer<Dtype>::LABEL];
      CHECK_GE(label, 0.0);
      CHECK_LT(label, num_class);
      rois_list.push_back(window);
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }

    // add rois to roidb
    roidb_.push_back(rois_list);

    if ((image_index+1) % 5000 == 0) {
      LOG(INFO) << image_index+1 << " images processed.";
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << it->second << " samples";
  }

  // 2. shuffle image_database_, roidb_
  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = static_cast<unsigned int>(rand_seed);
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleData();
  cur_ = ims_per_gpu * thread_id;

  // 3. reshape output blobs
  // data
  top[0]->Reshape(ims_per_gpu, channels, max_size, max_size);
  this->prefetch_data_.Reshape(ims_per_gpu, channels, max_size, max_size);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size * ims_per_gpu);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
  // rois
  vector<int> rois_shape;
  rois_shape.push_back(batch_size * ims_per_gpu);
  rois_shape.push_back(5);  // R = [batch_index x1 y1 x2 y2]
  top[2]->Reshape(rois_shape);
  this->prefetch_rois_.Reshape(rois_shape);
  // bbox_targets
  vector<int> bbox_targets_shape(2, batch_size * ims_per_gpu);
  bbox_targets_shape[1] = 4 * num_class;
  top[3]->Reshape(bbox_targets_shape);
  this->prefetch_bbox_targets_.Reshape(bbox_targets_shape);
  // bbox_weights
  vector<int> bbox_weights_shape(2, batch_size * ims_per_gpu);
  bbox_weights_shape[1] = 4 * num_class;
  top[4]->Reshape(bbox_weights_shape);
  this->prefetch_bbox_weights_.Reshape(bbox_weights_shape);

  // 4. get data mean
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_values_) {
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

// For shuffling data
template <typename Dtype>
void FrcnnDataLayer<Dtype>::ShuffleData() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  int length = image_database_.size();
  vector<int> shuff_inds = shuffle_index(length, prefetch_rng);
  for (int i = length - 1; i > 0; --i) {
    std::iter_swap(image_database_.begin() + i, image_database_.begin() + shuff_inds[length-i-1]);
    std::iter_swap(roidb_.begin() + i, roidb_.begin() + shuff_inds[length-i-1]);
  }
}

bool cmp(const pair<int, int> &x, const pair<int, int> &y) {
  return x.first > y.first;
}

template <typename Dtype>
vector<int> FrcnnDataLayer<Dtype>::Randperm(int total, int pick) {
  srand((unsigned)time(0));
  vector<pair<int, int> > rand_list;
  for (int i = 0; i < total; ++i) {
    rand_list.push_back(std::make_pair(rand(), i));
  }
  sort(rand_list.begin(), rand_list.end(), cmp);
  vector<int> picks;
  for (int i = 0; i < pick; ++i) {
    picks.push_back(rand_list[i].second);
  }
  return picks;
}

// Thread fetching the data
template <typename Dtype>
void FrcnnDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample one mini-batch
  // 0. read parameters
  CPUTimer batch_timer;
  batch_timer.Start();
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_rois = this->prefetch_rois_.mutable_cpu_data();
  Dtype* top_bbox_targets = this->prefetch_bbox_targets_.mutable_cpu_data();
  Dtype* top_bbox_weights = this->prefetch_bbox_weights_.mutable_cpu_data();
  const int thread_id = Caffe::MPI_my_rank();
  const int thread_num = Caffe::MPI_all_rank();
  const int min_size = this->layer_param_.frcnn_data_param().min_size();
  const int max_size = this->layer_param_.frcnn_data_param().max_size();
  const int ims_per_batch = this->layer_param_.frcnn_data_param().ims_per_batch();
  const int ims_per_gpu = ims_per_batch;
  const float bg_thresh_hi = this->layer_param_.frcnn_data_param().bg_thresh_hi();
  const float bg_thresh_lo = this->layer_param_.frcnn_data_param().bg_thresh_lo();
  const float fg_fraction = this->layer_param_.frcnn_data_param().fg_fraction();
  const int num_class = this->layer_param_.frcnn_data_param().num_class();
  bool has_part = false;
  std::string part_name;
  if (this->layer_param_.frcnn_data_param().has_part_name()) {
    has_part = true;
    part_name = this->layer_param_.frcnn_data_param().part_name();
  }

  // 1. sample one mini-batch
  if ((cur_ + ims_per_gpu * thread_num) > image_database_.size()) {
    ShuffleData();
    cur_ = ims_per_gpu * thread_id;
  }
  vector<std::pair<std::string, vector<int> > > imdb;
  vector<vector<vector<float> > > roidb;
  for (int i = 0; i < ims_per_gpu; ++i) {
    imdb.push_back(image_database_[cur_ + i]);
    roidb.push_back(roidb_[cur_ + i]);
  }
  cur_ += ims_per_batch * thread_num;

  // 2. zero out top data
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);
  caffe_set(this->prefetch_label_.count(), Dtype(0), top_label);
  caffe_set(this->prefetch_rois_.count(), Dtype(0), top_rois);
  caffe_set(this->prefetch_bbox_targets_.count(), Dtype(0), top_bbox_targets);
  caffe_set(this->prefetch_bbox_weights_.count(), Dtype(0), top_bbox_weights);

  // 3. arrange rois on each im in imdb
  for (int im_id = 0; im_id < ims_per_gpu; ++im_id) {
    std::pair<std::string, vector<int> > im = imdb[im_id];
    vector<vector<float> > rois = roidb[im_id];
    DLOG(INFO) << "thread id: " << thread_id << " batch_im: " << im_id;

    // rescale im
    std::string im_path = im.first;
    float fg_thresh;
    int batch_size;
    if (has_part && im_path.find(part_name) != std::string::npos) {
      fg_thresh = this->layer_param_.frcnn_data_param().part_fg_thresh();
      batch_size = this->layer_param_.frcnn_data_param().part_batch_size();
    } else {
      fg_thresh = this->layer_param_.frcnn_data_param().fg_thresh();
      batch_size = this->layer_param_.frcnn_data_param().batch_size();
    }
    vector<int> im_info = im.second;
    int channels = im_info[0];
    bool is_color = channels == 3 ? true:false;
    int height = im_info[1];
    int width = im_info[2];
    bool flipped = im_info[3] > 0 ? true:false;
    float im_scale_min = static_cast<float>(min_size) / static_cast<float>(std::min(height, width));
    float im_scale_max = static_cast<float>(max_size) / static_cast<float>(std::max(height, width));
    float im_scale = std::min(im_scale_min, im_scale_max);
    int new_height = static_cast<int>(im_scale * static_cast<float>(height) + 0.5);
    new_height = new_height > max_size ? max_size : new_height;
    int new_width = static_cast<int>(im_scale * static_cast<float>(width) + 0.5);
    new_width = new_width > max_size ? max_size : new_width;
    cv::Mat cv_img = ReadImageToCVMat(im_path, new_height, new_width, is_color);
    DLOG(INFO) << "image rescale size: "<< new_height << ", " << new_width << ", " << is_color;

    // flip im
    if (flipped) {
      cv::flip(cv_img, cv_img, 1);
    }

    // copy im to top_data
    for (int h = 0; h < cv_img.rows; ++h) {
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_img.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          int top_index = ((im_id * channels + c) * max_size + h)
                   * max_size + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (this->has_mean_values_) {
            top_data[top_index] = (pixel - this->mean_values_[c]);
          } else {
            top_data[top_index] = pixel;
          }
        }
      }
    }
    DLOG(INFO) << "copy image";

    // sample fg and bg inds
    int rois_per_image = batch_size;
    int fg_per_image = static_cast<int>(static_cast<float>(rois_per_image) * fg_fraction + 0.5);
    int bg_per_image = rois_per_image - fg_per_image;

    vector<int> fg_inds_ori;
    vector<int> bg_inds_ori;
    for (int roi_id = 0; roi_id < rois.size(); ++roi_id) {
      vector<float> roi = rois[roi_id];
      float ov = roi[FrcnnDataLayer<Dtype>::OVERLAP];
      if (ov > fg_thresh) {
        fg_inds_ori.push_back(roi_id);
      } else if (ov >= bg_thresh_lo && ov < bg_thresh_hi) {
        bg_inds_ori.push_back(roi_id);
      }
    }

    // filter fg rois
    int fg_per_this_image = fg_inds_ori.size();
    vector<int> fg_inds;
    // if num of fg rois exceeds, pick from it
    if (fg_per_this_image > 0 && fg_per_this_image > fg_per_image) {
      vector<int> fg_picks = Randperm(fg_per_this_image, fg_per_image);
      for (int i = 0; i < fg_per_image; ++i) {
        fg_inds.push_back(fg_inds_ori[fg_picks[i]]);
      }
    } else {
      fg_inds = fg_inds_ori;
    }
    fg_per_this_image = fg_inds.size();
    // currently we allow no fg rois
    if (fg_per_this_image == 0) {
      LOG(INFO) << im_path << ": Oops! No foreground RoIs!";
    }

    // filter bg rois
    int bg_per_this_image = bg_inds_ori.size();
    vector<int> bg_inds;
    // if num of bg rois exceeds, pick from it
    if (bg_per_this_image > 0 && bg_per_this_image > bg_per_image) {
      vector<int> bg_picks = Randperm(bg_per_this_image, bg_per_image);
      for (int i = 0; i < bg_per_image; ++i) {
        bg_inds.push_back(bg_inds_ori[bg_picks[i]]);
      }
    } else {
      bg_inds = bg_inds_ori;
    }
    bg_per_this_image = bg_inds.size();
    // currently we allow no bg rois
    if (bg_per_this_image == 0) {
      LOG(INFO) << im_path << ": Oops! No background RoIs!";
    }

    DLOG(INFO) << "fp_num: " << fg_per_this_image << ", bg_num: " << bg_per_this_image;

    // augment it to rois_per_image with fg or bg rois
    int real_rois_size = fg_per_this_image + bg_per_this_image;
    if (real_rois_size < rois_per_image) {
      int aug_num = rois_per_image - real_rois_size;
      if (fg_per_this_image) {
        int loop = aug_num / fg_inds_ori.size();
        int concated_num = aug_num - loop * fg_inds_ori.size();
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < fg_inds_ori.size(); ++j)
            fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
        }
        for (int j = 0; j < concated_num; ++j)
          fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
        fg_per_this_image = fg_inds.size();
      } else if (bg_per_this_image) {
        int loop = aug_num / bg_inds_ori.size();
        int concated_num = aug_num - loop * bg_inds_ori.size();
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < bg_inds_ori.size(); ++j)
            bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
        }
        for (int j = 0; j < concated_num; ++j)
          bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
        bg_per_this_image = bg_inds.size();
      } else {
        LOG(ERROR) << im_path << " has no FG or BG objects!";
      }
    }
    CHECK_EQ(fg_per_this_image + bg_per_this_image, rois_per_image) << "num of RoIs doesn't much";
    DLOG(INFO) << "copy rois";

    // copy rois to top_rois
    for (int i = 0; i < fg_per_this_image; ++i) {
      DLOG(INFO) << "fg rois: " << i;
      int offset = 5 * (rois_per_image * im_id + i);
      top_rois[offset] = static_cast<Dtype>(im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    for (int i = 0; i < bg_per_this_image; ++i) {
      DLOG(INFO) << "bg rois: " << i;
      int offset = 5 * (rois_per_image * im_id + fg_per_this_image + i);
      top_rois[offset] = static_cast<Dtype>(im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[bg_inds[i]][FrcnnDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    DLOG(INFO) << "copy rois";

    // copy label to top_label
    for (int i = 0; i < fg_per_this_image; ++i) {
      int offset = 1 * (rois_per_image * im_id + i);
      float label = rois[fg_inds[i]][FrcnnDataLayer<Dtype>::LABEL];
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      top_label[offset] = static_cast<Dtype>(label);
    }
    DLOG(INFO) << "copy label";

    // get bbox targets and weights and copy to top
    for (int i = 0; i < fg_per_this_image; ++i) {
      int label = static_cast<int>(rois[fg_inds[i]][FrcnnDataLayer<Dtype>::LABEL]);
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      int stride = 4 * num_class;
      int offset = stride * (rois_per_image * im_id + i) + 4 * label;
      for (int j = 0; j < 4; ++j) {
        top_bbox_targets[offset+j] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnDataLayer<Dtype>::DX + j]);
        top_bbox_weights[offset+j] = static_cast<Dtype>(1);
      }
    }
    DLOG(INFO) << "copy bbox";
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(FrcnnDataLayer);
REGISTER_LAYER_CLASS(FrcnnData);

}  // namespace caffe
