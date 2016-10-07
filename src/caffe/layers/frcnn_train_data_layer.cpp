///////////////////////////////////////////////////////////////////
// Fast RCNN Train DataLayer
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
// Related file    data_layers.hpp
//                 base_data_layer.cpp
//                 base_data_layer.cu
//                 rng.hpp
//                 caffe.proto
// Reference file  image_data_layer.cpp
//                 window_data_layer.cpp
//                 fast-rcnn/lib/roi_data_layer/layer.py
//                 fast-rcnn/lib/roi_data_layer/minibatch.py
// Written by      Bin Yang
// Last edit       2015.10.15
///////////////////////////////////////////////////////////////////

#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > FrcnnTrainDataParameter
//   'source'          where the rois_file is
//   'root_folder'     root folder where images data store
//   'scales'          minimum size of re-scaled image (support multi-scale training)
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
FrcnnTrainDataLayer<Dtype>::~FrcnnTrainDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void FrcnnTrainDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the rois_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // 0. read parameters
  // thread
  const int thread_id = Caffe::MPI_my_rank();
  int sz;
  const string source_file = this->layer_param_.frcnn_train_data_param().source(0);
  const string source_file_2 = this->layer_param_.frcnn_train_data_param().source(1);
  const string root_folder = this->layer_param_.frcnn_train_data_param().root_folder(0);
  const string root_folder_2 = this->layer_param_.frcnn_train_data_param().root_folder(1);
  const int max_size = this->layer_param_.frcnn_train_data_param().max_size();
  sz = this->layer_param_.frcnn_train_data_param().ims_per_batch_size();
  int ims_per_batch = this->layer_param_.frcnn_train_data_param().ims_per_batch(std::min(sz-1, thread_id));
  int gt_per_batch = this->layer_param_.frcnn_train_data_param().gt_per_batch();
  sz = this->layer_param_.frcnn_train_data_param().batch_size_size();
  int batch_size = this->layer_param_.frcnn_train_data_param().batch_size(0);
  int gt_batch_size = this->layer_param_.frcnn_train_data_param().batch_size(1);
  sz = this->layer_param_.frcnn_train_data_param().rand_seed_size();
  const int num_class = this->layer_param_.frcnn_train_data_param().num_class();
  const int rand_seed = this->layer_param_.frcnn_train_data_param().rand_seed(std::min(sz-1, thread_id));
  sz = this->layer_param_.frcnn_train_data_param().rand_skip_size();
  skip_ = this->layer_param_.frcnn_train_data_param().rand_skip(std::min(sz-1, thread_id));
  bool split_source = this->layer_param_.frcnn_train_data_param().split_source();

  // 1. read rois_file
  std::ifstream infile(source_file.c_str());
  std::ifstream infile_2(source_file_2.c_str());
  CHECK(infile.good()) << "Failed to open window file " << source_file << std::endl;

  map<int, int> label_hist, label_hist_2;
  label_hist.insert(std::make_pair(0, 0));
  label_hist_2.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
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
    vector<vector<float> > rois_list;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap, dx, dy, dw, dh;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >> dx >> dy >> dw >> dh;

      vector<float> window(FrcnnTrainDataLayer<Dtype>::NUM);
      window[FrcnnTrainDataLayer<Dtype>::LABEL] = label;
      window[FrcnnTrainDataLayer<Dtype>::OVERLAP] = overlap;
      window[FrcnnTrainDataLayer<Dtype>::X1] = x1;
      window[FrcnnTrainDataLayer<Dtype>::Y1] = y1;
      window[FrcnnTrainDataLayer<Dtype>::X2] = x2;
      window[FrcnnTrainDataLayer<Dtype>::Y2] = y2;
      window[FrcnnTrainDataLayer<Dtype>::DX] = dx;
      window[FrcnnTrainDataLayer<Dtype>::DY] = dy;
      window[FrcnnTrainDataLayer<Dtype>::DW] = dw;
      window[FrcnnTrainDataLayer<Dtype>::DH] = dh;

      // add box to rois list
      label = window[FrcnnTrainDataLayer<Dtype>::LABEL];
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

  if (!(infile_2 >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile_2 >> image_path;
    image_path = root_folder_2 + image_path;
    // get image info
    vector<int> image_info(4);
    infile_2 >> image_info[0] >> image_info[1] >> image_info[2] >> image_info[3];
    image_database_2_.push_back(std::make_pair(image_path, image_info));
    channels = image_info[0];

    // read each box
    int num_windows;
    infile_2 >> num_windows;
    vector<vector<float> > rois_list;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap, dx, dy, dw, dh;
      infile_2 >> label >> overlap >> x1 >> y1 >> x2 >> y2 >> dx >> dy >> dw >> dh;

      vector<float> window(FrcnnTrainDataLayer<Dtype>::NUM);
      window[FrcnnTrainDataLayer<Dtype>::LABEL] = label;
      window[FrcnnTrainDataLayer<Dtype>::OVERLAP] = overlap;
      window[FrcnnTrainDataLayer<Dtype>::X1] = x1;
      window[FrcnnTrainDataLayer<Dtype>::Y1] = y1;
      window[FrcnnTrainDataLayer<Dtype>::X2] = x2;
      window[FrcnnTrainDataLayer<Dtype>::Y2] = y2;
      window[FrcnnTrainDataLayer<Dtype>::DX] = dx;
      window[FrcnnTrainDataLayer<Dtype>::DY] = dy;
      window[FrcnnTrainDataLayer<Dtype>::DW] = dw;
      window[FrcnnTrainDataLayer<Dtype>::DH] = dh;

      // add box to rois list
      label = window[FrcnnTrainDataLayer<Dtype>::LABEL];
      CHECK_GE(label, 0.0);
      CHECK_LT(label, num_class);
      rois_list.push_back(window);
      label_hist_2.insert(std::make_pair(label, 0));
      label_hist_2[label]++;
    }

    // add rois to roidb
    roidb_2_.push_back(rois_list);

    if ((image_index+1) % 5000 == 0) {
      LOG(INFO) << image_index+1 << " images processed.";
    }
  } while (infile_2 >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;
  for (map<int, int>::iterator it = label_hist_2.begin();
      it != label_hist_2.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << it->second << " samples";
  }

  // 2. shuffle image_database_, roidb_
  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = static_cast<unsigned int>(rand_seed);
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleData();
  cur_ = ims_per_batch * thread_id;
  ShuffleData2();
  cur_2_ = gt_per_batch * thread_id;

  // 3. reshape output blobs
  // data
  if (split_source) {
    if (thread_id % 2) {
      ims_per_batch = 0;
      batch_size = 0;
    } else {
      gt_per_batch = 0;
      gt_batch_size = 0;
    }
  }
  top[0]->Reshape(ims_per_batch + gt_per_batch, channels, max_size, max_size);
  this->prefetch_data_.Reshape(ims_per_batch + gt_per_batch, channels, max_size, max_size);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size*ims_per_batch + gt_batch_size*gt_per_batch);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
  // rois
  vector<int> rois_shape(2, batch_size*ims_per_batch + gt_batch_size*gt_per_batch);
  rois_shape[1] = 5;
  top[2]->Reshape(rois_shape);
  this->prefetch_rois_.Reshape(rois_shape);
  // bbox_targets
  vector<int> bbox_targets_shape(2, batch_size*ims_per_batch + gt_batch_size*gt_per_batch);
  bbox_targets_shape[1] = 4*num_class;
  top[3]->Reshape(bbox_targets_shape);
  this->prefetch_bbox_targets_.Reshape(bbox_targets_shape);
  // bbox_weights
  vector<int> bbox_weights_shape(2, batch_size*ims_per_batch + gt_batch_size*gt_per_batch);
  bbox_weights_shape[1] = 4*num_class;
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
void FrcnnTrainDataLayer<Dtype>::ShuffleData() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  int length = image_database_.size();
  vector<int> shuff_inds = shuffle_index(length, prefetch_rng);
  for (int i = length - 1; i > 0; --i) {
    std::iter_swap(image_database_.begin() + i, image_database_.begin() + shuff_inds[length-i-1]);
    std::iter_swap(roidb_.begin() + i, roidb_.begin() + shuff_inds[length-i-1]);
  }
}

template <typename Dtype>
void FrcnnTrainDataLayer<Dtype>::ShuffleData2() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  int length = image_database_2_.size();
  vector<int> shuff_inds = shuffle_index(length, prefetch_rng);
  for (int i = length - 1; i > 0; --i) {
    std::iter_swap(image_database_2_.begin() + i, image_database_2_.begin() + shuff_inds[length-i-1]);
    std::iter_swap(roidb_2_.begin() + i, roidb_2_.begin() + shuff_inds[length-i-1]);
  }
}

bool Cmp(const pair<int, int> &x, const pair<int, int> &y) {
  return x.first > y.first;
}

template <typename Dtype>
vector<int> FrcnnTrainDataLayer<Dtype>::Randperm(int total, int pick) {
  srand((unsigned)time(0));
  vector<pair<int, int> > rand_list;
  for (int i = 0; i < total; ++i) {
    rand_list.push_back(std::make_pair(rand(), i));
  }
  sort(rand_list.begin(), rand_list.end(), Cmp);
  vector<int> picks;
  for (int i = 0; i < pick; ++i) {
    picks.push_back(rand_list[i].second);
  }
  return picks;
}

// Thread fetching the data
template <typename Dtype>
void FrcnnTrainDataLayer<Dtype>::InternalThreadEntry() {
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
  int sz;
  sz = this->layer_param_.frcnn_train_data_param().scales_size();
  vector<int> rand_scale_id = Randperm(sz, 1);
  const int scale = this->layer_param_.frcnn_train_data_param().scales(rand_scale_id[0]);
  const int max_size = this->layer_param_.frcnn_train_data_param().max_size();
  sz = this->layer_param_.frcnn_train_data_param().ims_per_batch_size();
  int ims_per_batch = this->layer_param_.frcnn_train_data_param().ims_per_batch(std::min(sz-1, thread_id));
  sz = this->layer_param_.frcnn_train_data_param().batch_size_size();
  int batch_size = this->layer_param_.frcnn_train_data_param().batch_size(0);
  int gt_batch_size = this->layer_param_.frcnn_train_data_param().batch_size(1);
  int gt_per_batch = this->layer_param_.frcnn_train_data_param().gt_per_batch();
  sz = this->layer_param_.frcnn_train_data_param().fg_thresh_size();
  const float fg_thresh = this->layer_param_.frcnn_train_data_param().fg_thresh(0);
  const float fg_thresh_2 = this->layer_param_.frcnn_train_data_param().fg_thresh(1);
  sz = this->layer_param_.frcnn_train_data_param().bg_thresh_hi_size();
  const float bg_thresh_hi = this->layer_param_.frcnn_train_data_param().bg_thresh_hi(std::min(sz-1, thread_id));
  sz = this->layer_param_.frcnn_train_data_param().bg_thresh_lo_size();
  const float bg_thresh_lo = this->layer_param_.frcnn_train_data_param().bg_thresh_lo(std::min(sz-1, thread_id));
  sz = this->layer_param_.frcnn_train_data_param().fg_fraction_size();
  const float fg_fraction = this->layer_param_.frcnn_train_data_param().fg_fraction(std::min(sz-1, thread_id));
  const int num_class = this->layer_param_.frcnn_train_data_param().num_class();
  bool split_source = this->layer_param_.frcnn_train_data_param().split_source();

  if (split_source) {
    if (thread_id % 2) {
      ims_per_batch = 0;
      batch_size = 0;
    } else {
      gt_per_batch = 0;
      gt_batch_size = 0;
    }
  }

  // 1. sample one mini-batch
  if ((cur_ + ims_per_batch * thread_num) > image_database_.size()) {
    ShuffleData();
    cur_ = ims_per_batch * thread_id;
  }
  vector<std::pair<std::string, vector<int> > > imdb;
  vector<vector<vector<float> > > roidb;
  for (int i = 0; i < ims_per_batch; ++i) {
    imdb.push_back(image_database_[cur_ + i]);
    roidb.push_back(roidb_[cur_ + i]);
  }
  cur_ += ims_per_batch * thread_num;

  if ((cur_2_ + gt_per_batch * thread_num) > image_database_2_.size()) {
    ShuffleData2();
    cur_2_ = gt_per_batch * thread_id;
  }
  vector<std::pair<std::string, vector<int> > > imdb2;
  vector<vector<vector<float> > > roidb2;
  for (int i = 0; i < gt_per_batch; ++i) {
    imdb2.push_back(image_database_2_[cur_2_ + i]);
    roidb2.push_back(roidb_2_[cur_2_ + i]);
  }
  cur_2_ += gt_per_batch * thread_num;

  // 2. zero out top data
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);
  caffe_set(this->prefetch_label_.count(), Dtype(0), top_label);
  caffe_set(this->prefetch_rois_.count(), Dtype(0), top_rois);
  caffe_set(this->prefetch_bbox_targets_.count(), Dtype(0), top_bbox_targets);
  caffe_set(this->prefetch_bbox_weights_.count(), Dtype(0), top_bbox_weights);

  // 3. arrange rois on each im in imdb
  for (int im_id = 0; im_id < ims_per_batch; ++im_id) {
    std::pair<std::string, vector<int> > im = imdb[im_id];
    vector<vector<float> > rois = roidb[im_id];
    DLOG(INFO) << "thread id: " << thread_id << " batch_im: " << im_id;

    // rescale im
    std::string im_path = im.first;
    vector<int> im_info = im.second;
    int channels = im_info[0];
    bool is_color = channels == 3 ? true:false;
    int height = im_info[1];
    int width = im_info[2];
    bool flipped = im_info[3] > 0 ? true:false;
    float im_scale_min = static_cast<float>(scale) / static_cast<float>(std::min(height, width));
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
          // int top_index = (c * height + h) * width + w;
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
    int bg_per_image, fg_per_this_image, bg_per_this_image;
    vector<int> fg_inds_ori, bg_inds_ori, fg_inds, bg_inds, fg_picks, bg_picks;
    for (int roi_id = 0; roi_id < rois.size(); ++roi_id) {
      vector<float> roi = rois[roi_id];
      float ov = roi[FrcnnTrainDataLayer<Dtype>::OVERLAP];
      if (ov > fg_thresh) {
        fg_inds_ori.push_back(roi_id);
      } else if (ov >= bg_thresh_lo && ov < bg_thresh_hi) {
        bg_inds_ori.push_back(roi_id);
      }
    }
    fg_per_this_image = fg_inds_ori.size();                        // fg
    if (fg_per_this_image > 0 && fg_per_this_image > fg_per_image) {
      fg_picks = Randperm(fg_per_this_image, fg_per_image);
      fg_per_this_image = fg_per_image;
      CHECK_EQ(fg_per_this_image, fg_picks.size());
      for (int i = 0; i < fg_per_this_image; ++i) {
        fg_inds.push_back(fg_inds_ori[fg_picks[i]]);
      }
    } else {
      fg_inds = fg_inds_ori;
    }
    bg_per_image = rois_per_image - fg_per_this_image;         // bg
    bg_per_this_image = bg_inds_ori.size();
    if (bg_per_this_image > 0 && bg_per_this_image > bg_per_image) {
      bg_picks = Randperm(bg_per_this_image, bg_per_image);
      bg_per_this_image = bg_per_image;
      CHECK_EQ(bg_per_this_image, bg_picks.size());
      for (int i = 0; i < bg_per_this_image; ++i) {
        bg_inds.push_back(bg_inds_ori[bg_picks[i]]);
      }
    } else {
      bg_inds = bg_inds_ori;
    }
    DLOG(INFO) << "fp_num: " << fg_per_this_image << ", bg_num: " << bg_per_this_image;

    // augment it to rois_per_image with fg rois
    int real_rois_size = fg_per_this_image + bg_per_this_image;
    DLOG(INFO) << "real_rois_num" << real_rois_size;
    if (real_rois_size < rois_per_image) {
      int aug_num = rois_per_image - real_rois_size;
      // prefer fg if there is at least one sample
      if (fg_inds_ori.size() > 0) {
        int loop = aug_num / fg_inds_ori.size();
        int concated_num = aug_num - loop * fg_inds_ori.size();
        DLOG(INFO) << aug_num << loop << concated_num;
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < fg_inds_ori.size(); ++j) {
            fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
          }
        }
        for (int j = 0; j < concated_num; ++j) {
            fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
        }
        fg_per_this_image = fg_inds.size();
      } else if (bg_inds_ori.size() > 0) {
        int loop = aug_num / bg_inds_ori.size();
        int concated_num = aug_num - loop * bg_inds_ori.size();
        DLOG(INFO) << aug_num << loop << concated_num;
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < bg_inds_ori.size(); ++j) {
            bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
          }
        }
        for (int j = 0; j < concated_num; ++j) {
            bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
        }
        bg_per_this_image = bg_inds.size();
      } else {
        LOG(FATAL) << im_path << ": Oops! No fg or bg boxes at all!";
      }
    }
    CHECK_EQ(fg_per_this_image + bg_per_this_image, rois_per_image) << im_path << ": Roi Num Doesn't Match";

    DLOG(INFO) << "copy rois";

    // copy rois to top_rois
    for (int i = 0; i < fg_per_this_image; ++i) {
      DLOG(INFO) << "fg rois: " << i;
      int offset = 5 * (rois_per_image * im_id + i);
      top_rois[offset] = static_cast<Dtype>(im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    for (int i = 0; i < bg_per_this_image; ++i) {
      DLOG(INFO) << "bg rois: " << i;
      int offset = 5 * (rois_per_image * im_id + fg_per_this_image + i);
      top_rois[offset] = static_cast<Dtype>(im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[bg_inds[i]][FrcnnTrainDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    DLOG(INFO) << "copy rois";

    // copy label to top_label
    for (int i = 0; i < fg_per_this_image; ++i) {
      int offset = 1 * (rois_per_image * im_id + i);
      float label = rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::LABEL];
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      top_label[offset] = static_cast<Dtype>(label);
    }
    DLOG(INFO) << "copy label";

    // get bbox targets and weights and copy to top
    for (int i = 0; i < fg_per_this_image; ++i) {
      int label = static_cast<int>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::LABEL]);
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      int stride = 4 * num_class;
      int offset = stride * (rois_per_image * im_id + i) + 4 * label;
      for (int j = 0; j < 4; ++j) {
        top_bbox_targets[offset+j] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::DX + j]);
        top_bbox_weights[offset+j] = static_cast<Dtype>(1);
      }
    }
    DLOG(INFO) << "copy bbox";
  }

  // 4. arrange gts on each im in imdb
  for (int im_id = 0; im_id < gt_per_batch; ++im_id) {
    std::pair<std::string, vector<int> > im = imdb2[im_id];
    vector<vector<float> > rois = roidb2[im_id];
    DLOG(INFO) << "thread id: " << thread_id << " batch_im: " << im_id;

    // rescale im
    std::string im_path = im.first;
    vector<int> im_info = im.second;
    int channels = im_info[0];
    bool is_color = channels == 3 ? true : false;
    int height = im_info[1];
    int width = im_info[2];
    bool flipped = im_info[3] > 0 ? true : false;
    float im_scale_min = static_cast<float>(scale) / static_cast<float>(std::min(height, width));
    float im_scale_max = static_cast<float>(max_size) / static_cast<float>(std::max(height, width));
    float im_scale = std::min(im_scale_min, im_scale_max);
    int new_height = static_cast<int>(im_scale * static_cast<float>(height) + 0.5);
    new_height = new_height > max_size ? max_size : new_height;
    int new_width = static_cast<int>(im_scale * static_cast<float>(width) + 0.5);
    new_width = new_width > max_size ? max_size : new_width;
    cv::Mat cv_img = ReadImageToCVMat(im_path, new_height, new_width, is_color);

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
          int top_index = (((ims_per_batch + im_id) * channels + c) * max_size + h)
                   * max_size + w;
          // int top_index = (c * height + h) * width + w;
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
    int rois_per_image = gt_batch_size;
    int fg_per_image = static_cast<int>(static_cast<float>(rois_per_image) * fg_fraction + 0.5);
    int bg_per_image, fg_per_this_image, bg_per_this_image;
    vector<int> fg_inds_ori, bg_inds_ori, fg_inds, bg_inds, fg_picks, bg_picks;
    for (int roi_id = 0; roi_id < rois.size(); ++roi_id) {
      vector<float> roi = rois[roi_id];
      float ov = roi[FrcnnTrainDataLayer<Dtype>::OVERLAP];
      if (ov >= fg_thresh_2) {
        fg_inds_ori.push_back(roi_id);
      } else if (ov >= bg_thresh_lo && ov < bg_thresh_hi) {
        bg_inds_ori.push_back(roi_id);
      }
    }
    fg_per_this_image = fg_inds_ori.size();                        // fg
    if (fg_per_this_image > 0 && fg_per_this_image > fg_per_image) {
      fg_picks = Randperm(fg_per_this_image, fg_per_image);
      fg_per_this_image = fg_per_image;
      CHECK_EQ(fg_per_this_image, fg_picks.size());
      for (int i = 0; i < fg_per_this_image; ++i) {
        fg_inds.push_back(fg_inds_ori[fg_picks[i]]);
      }
    } else {
      fg_inds = fg_inds_ori;
    }

    bg_per_image = rois_per_image - fg_per_this_image;         // bg
    bg_per_this_image = bg_inds_ori.size();
    if (bg_per_this_image > 0 && bg_per_this_image > bg_per_image) {
      bg_picks = Randperm(bg_per_this_image, bg_per_image);
      bg_per_this_image = bg_per_image;
      CHECK_EQ(bg_per_this_image, bg_picks.size());
      for (int i = 0; i < bg_per_this_image; ++i) {
        bg_inds.push_back(bg_inds_ori[bg_picks[i]]);
      }
    } else {
      bg_inds = bg_inds_ori;
    }
    DLOG(INFO) << "fp_num: " << fg_per_this_image << ", bg_num: " << bg_per_this_image;

    // augment it to rois_per_image with fg rois
    int real_rois_size = fg_per_this_image + bg_per_this_image;
    DLOG(INFO) << "real_rois_num" << real_rois_size;
    if (real_rois_size < rois_per_image) {
      int aug_num = rois_per_image - real_rois_size;
      // prefer fg if there is at least one sample
      if (fg_inds_ori.size() > 0) {
        int loop = aug_num / fg_inds_ori.size();
        int concated_num = aug_num - loop * fg_inds_ori.size();
        DLOG(INFO) << aug_num << loop << concated_num;
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < fg_inds_ori.size(); ++j) {
            fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
          }
        }
        for (int j = 0; j < concated_num; ++j) {
            fg_inds.push_back(fg_inds_ori[fg_inds_ori.size()-j-1]);  // prefer gt
        }
        fg_per_this_image = fg_inds.size();
      } else if (bg_inds_ori.size() > 0) {
        int loop = aug_num / bg_inds_ori.size();
        int concated_num = aug_num - loop * bg_inds_ori.size();
        DLOG(INFO) << aug_num << loop << concated_num;
        for (int i = 0; i < loop; ++i) {
          for (int j = 0; j < bg_inds_ori.size(); ++j) {
            bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
          }
        }
        for (int j = 0; j < concated_num; ++j) {
            bg_inds.push_back(bg_inds_ori[bg_inds_ori.size()-j-1]);
        }
        bg_per_this_image = bg_inds.size();
      } else {
        LOG(FATAL) << im_path << ": Oops! No fg or bg boxes at all!";
      }
    }
    CHECK_EQ(fg_per_this_image + bg_per_this_image, rois_per_image) << im_path << ": Roi Num Doesn't Match";

    // copy rois to top_rois
    for (int i = 0; i < fg_per_this_image; ++i) {
      DLOG(INFO) << "fg rois: " << i;
      int offset = 5 * (batch_size * ims_per_batch + rois_per_image * im_id + i);
      top_rois[offset] = static_cast<Dtype>(ims_per_batch + im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    for (int i = 0; i < bg_per_this_image; ++i) {
      DLOG(INFO) << "bg rois: " << i;
      int offset = 5 * (batch_size * ims_per_batch + rois_per_image * im_id + fg_per_this_image + i);
      top_rois[offset] = static_cast<Dtype>(ims_per_batch + im_id);
      for (int j = 0; j < 4; ++j) {
        top_rois[offset+j+1] = static_cast<Dtype>(rois[bg_inds[i]][FrcnnTrainDataLayer<Dtype>::X1 + j] * im_scale);
      }
    }
    DLOG(INFO) << "copy rois";

    // copy label to top_label
    for (int i = 0; i < fg_per_this_image; ++i) {
      int offset = 1 * (batch_size * ims_per_batch + rois_per_image * im_id + i);
      float label = rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::LABEL];
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      top_label[offset] = static_cast<Dtype>(label);
    }
    DLOG(INFO) << "copy label";

    // get bbox targets and weights and copy to top
    for (int i = 0; i < fg_per_this_image; ++i) {
      int label = static_cast<int>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::LABEL]);
      CHECK_GT(label, 0.0);
      CHECK_LT(label, num_class);
      int stride = 4 * num_class;
      int offset = stride * (batch_size * ims_per_batch + rois_per_image * im_id + i) + 4 * label;
      for (int j = 0; j < 4; ++j) {
        top_bbox_targets[offset+j] = static_cast<Dtype>(rois[fg_inds[i]][FrcnnTrainDataLayer<Dtype>::DX + j]);
        top_bbox_weights[offset+j] = static_cast<Dtype>(1);
      }
    }
    DLOG(INFO) << "copy bbox";
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(FrcnnTrainDataLayer);
REGISTER_LAYER_CLASS(FrcnnTrainData);

}  // namespace caffe
