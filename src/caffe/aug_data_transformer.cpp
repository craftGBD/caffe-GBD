#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "caffe/aug_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::vector;
using namespace cv;

namespace caffe {

template<typename Dtype>
AugDataTransformer<Dtype>::AugDataTransformer(const AugTransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {}

template<typename Dtype>
void AugDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                          const vector<Dtype>& cv_points,
                                          Blob<Dtype>* transformed_blob,
                                          Blob<Dtype>* transformed_blob2) {
  
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const int img_point_num = cv_points.size();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  vector<Dtype> points(cv_points);
  const int point_num = transformed_blob2->count(0);

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);
  CHECK_EQ(img_point_num, point_num);
  CHECK_EQ((img_point_num % 2), 0);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const double aug_prob = param_.aug_prob();
  
  int crop_w, crop_h;

  if(param_.crop_h() > 0 && param_.crop_w() > 0)
       crop_h = param_.crop_h(), crop_w = param_.crop_w();
  else if(param_.crop_size() > 0)
       crop_h = crop_w = param_.crop_size();
  else
       crop_h = height, crop_w = width;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  const bool aug_flag =  (Rand(unsigned(~0)) <= aug_prob * unsigned(~0));
 
  const bool trans = aug_flag && param_.trans();
  double trans_dx = param_.trans_dx();
  double trans_dy  = param_.trans_dy();
  if(trans){
    if(param_.trans_x_rng().type() != MRNGParameter_MRNGType_NONE
       || param_.trans_y_rng().type() != MRNGParameter_MRNGType_NONE){
      m_rand_gen(param_.trans_x_rng(), trans_dx);
      m_rand_gen(param_.trans_y_rng(), trans_dy);
    }
    else if(param_.trans_rng().type() != MRNGParameter_MRNGType_NONE){
      m_rand_gen(param_.trans_rng(), trans_dx);
      m_rand_gen(param_.trans_rng(), trans_dy);
    }
  }
  else
    trans_dx = trans_dy = 0;

  const bool rotate = aug_flag && param_.rotate();
  double rotate_deg = param_.rotate_deg();
  if(rotate){
    m_rand_gen(param_.rotate_rng(), rotate_deg);
    CHECK_GE(rotate_deg, -180.0);
    CHECK_LE(rotate_deg, 180.0);
  }
  else
    rotate_deg = 0;

  const bool zoom = aug_flag && param_.zoom();
  int zoom_type = param_.zoom_type();
  double zoom_x_scale = param_.zoom_x_scale();
  double zoom_y_scale = param_.zoom_y_scale();
  if(zoom){
    if(zoom_type == 0){
      double zoom_scale = param_.zoom_scale();
      m_rand_gen(param_.zoom_rng(), zoom_scale);
      zoom_x_scale = zoom_y_scale = zoom_scale;
    }
    else{
      m_rand_gen(param_.zoom_x_rng(), zoom_x_scale);
      m_rand_gen(param_.zoom_y_rng(), zoom_y_scale);
    }
    CHECK_GT(zoom_x_scale,  0);
    CHECK_GT(zoom_y_scale,  0);
  }
  else
    zoom_x_scale = zoom_y_scale =1.0;
  
  const bool mirror = aug_flag && param_.mirror() && Rand(2);
  
  /*
  Mat before = cv_img.clone();
  for(int p = 0; p < 10; p += 2)
    for(int i = -15; i < 16; i++)
       for(int j = -15; j < 16; j++)
         before.at<uchar>((int)points[p + 1] + j, (int)points[p] + i) = 255 ;
   std::cout << "======" << std::endl;
   imwrite("before.png", before);
   //*/

  Mat cv_tmp_img;
  CvPoint2D32f center;
  center.x = img_width / 2.0 - 0.5;
  center.y = img_height / 2.0 - 0.5;
  if(zoom || rotate || trans){
    cv_tmp_img = Mat::zeros(cv_img.rows, cv_img.cols, cv_img.type());
    Mat transform_matrix =  getRotationMatrix2D(center, rotate_deg, 1);
    transform_matrix.at<double>(0, 2) += trans_dx ;
    transform_matrix.at<double>(1, 2) += trans_dy ;
    for(int i = 0; i < 3; i++){
      transform_matrix.at<double>(0, i) *= zoom_x_scale;
      transform_matrix.at<double>(1, i) *= zoom_y_scale;
    }
    transform_matrix.at<double>(0, 2) += (1 - zoom_x_scale) * center.x;
    transform_matrix.at<double>(1, 2) += (1 - zoom_y_scale) * center.y;

    if (mirror) {
      for(int i = 0; i < 3; i++)
        transform_matrix.at<double>(0, i) = -transform_matrix.at<double>(0, i);
      transform_matrix.at<double>(0, 2) += 2 * center.x;
      for(int i = 0; i < param_.corr_list_size(); i += 2){
        int p, q;
        Dtype t;
        p = param_.corr_list(i) - 1;
        q = param_.corr_list(i + 1) - 1;
        CHECK_GE(points.size() - 2, p * 2);
        CHECK_GE(points.size() - 2, q * 2);
        for(int j = 0; j < 2; j++){
          t = points[p * 2 + j];
          points[p * 2 + j] = points[q * 2 + j];
          points[q * 2 + j] = t;
        }
      }
    }

    CHECK_EQ(transform_matrix.rows, 2);
    CHECK_EQ(transform_matrix.cols, 3);
    warpAffine(cv_img, cv_tmp_img, transform_matrix, cv_tmp_img.size());
    for(int i = 0; i < img_point_num; i += 2){
      double tx = points[i];
      double ty = points[i + 1] ;
      points[i] = transform_matrix.at<double>(0, 0) * tx + transform_matrix.at<double>(0, 1) * ty ;
      points[i + 1] = transform_matrix.at<double>(1, 0) * tx + transform_matrix.at<double>(1, 1) * ty;
      points[i] += transform_matrix.at<double>(0, 2);
      points[i + 1] += transform_matrix.at<double>(1, 2);
    }
  }
  else
    cv_tmp_img = cv_img.clone();
  
  /*
  for(int p = 0; p < 10; p += 2)
    for(int i = -15; i < 16; i++)
      for(int j = -15; j < 16; j++)
       cv_tmp_img.at<uchar>((int)points[p + 1] + j, (int)points[p] + i) = 255;
  std::cout << "======" << std::endl;
  imwrite("after.png", cv_tmp_img);
  getchar();
  //*/

  CHECK_EQ(img_height, cv_tmp_img.rows);
  CHECK_EQ(img_width, cv_tmp_img.cols);
  
  int h_off = 0; 
  int w_off = 0;
  cv::Mat cv_cropped_img =  Mat::zeros(cv_tmp_img.rows, cv_tmp_img.cols, cv_tmp_img.type());
  if (crop_h != img_height || crop_w != img_width) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    h_off = (img_height - crop_h) / 2 - 0.5;
    w_off = (img_width - crop_w) / 2 - 0.5;
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_tmp_img(roi);
    for(int i = 0; i < img_point_num; i += 2){
      points[i] -= w_off;
      points[i + 1] -= h_off;
    }
  } 
  else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);
  
  // random occlusion
  const bool occlusion = aug_flag && param_.occlusion();
  if(occlusion)
  {
    double occlusion_width = param_.occlusion_width();
    double occlusion_height  = param_.occlusion_height();
    double occlusion_center_x = param_.occlusion_center_x();
    double occlusion_center_y  = param_.occlusion_center_y();
    Scalar occlusion_color(param_.occlusion_color(), param_.occlusion_color(), param_.occlusion_color(), param_.occlusion_color());
    
    if(param_.occlusion_width_rng().type() != MRNGParameter_MRNGType_NONE || 
       param_.occlusion_height_rng().type() != MRNGParameter_MRNGType_NONE)
    {
      m_rand_gen(param_.occlusion_width_rng(), occlusion_width);
      m_rand_gen(param_.occlusion_height_rng(), occlusion_height);
    }
    else if(param_.occlusion_size_rng().type() != MRNGParameter_MRNGType_NONE)
    {
      m_rand_gen(param_.occlusion_size_rng(), occlusion_width);
      m_rand_gen(param_.occlusion_size_rng(), occlusion_height);
    }
    
    if(param_.occlusion_center_x_rng().type() != MRNGParameter_MRNGType_NONE ||
       param_.occlusion_center_y_rng().type() != MRNGParameter_MRNGType_NONE)
    {
      m_rand_gen(param_.occlusion_center_x_rng(), occlusion_center_x);
      m_rand_gen(param_.occlusion_center_y_rng(), occlusion_center_y);
    }
    else if(param_.occlusion_center_rng().type() != MRNGParameter_MRNGType_NONE)
    {
      m_rand_gen(param_.occlusion_center_rng(), occlusion_center_x);
      m_rand_gen(param_.occlusion_center_rng(), occlusion_center_y);
    }
    
    if(param_.occlusion_color_rng().type() != MRNGParameter_MRNGType_NONE)
    {
      for(size_t i=0;i<img_channels;i++)
      {
        m_rand_gen(param_.occlusion_color_rng(), occlusion_color[i]);
      }
    }
    
    //check
    for(size_t i=0;i<img_channels;i++)
    {
      CHECK(occlusion_color[i]>=0 && occlusion_color[i]<=255);
    }
    
    Rect occlusion_rect(occlusion_center_x-occlusion_width/2, occlusion_center_y-occlusion_height/2, occlusion_width, occlusion_height);
    Rect img_rect(0, 0, cv_cropped_img.cols, cv_cropped_img.rows);
    Mat occlusionROI=cv_cropped_img(occlusion_rect&img_rect);
    occlusionROI.setTo(occlusion_color);
  }

  /*
  Mat final = cv_cropped_img.clone();
  for(int i = 0; i < final.rows; i++)
      for(int j = 0; j < final.cols; j++){
        final.at<uchar>(i, j) = (uchar)(final.at<uchar>(i, j));
      }
  std::cout << "======" << std::endl;
  imwrite("final.png", final);
  getchar();
  //*/
  
  vector<Mat> split_img(img_channels);
  split(cv_cropped_img, &(split_img[0]));
    
  // normalization by mean and std
  Mat mean;
  Mat std;
  const bool normalize = param_.normalize();
  if(normalize) 
  {
    meanStdDev(cv_cropped_img, mean, std);
    for(size_t i=0;i<img_channels;i++) 
    {
      if(std.at<double>(i, 0)<1E-6)
      {
        std.at<double>(i, 0)=1;
      }
    }
  }
  else 
  {
    for(size_t i=0;i<img_channels;i++) 
    {
      mean.at<double>(i, 0)=0;
      std.at<double>(i, 0)=1;
    }
  }
  for(size_t i=0;i<img_channels;i++)
  {
    split_img[i].convertTo(split_img[i], sizeof(Dtype)==4?CV_32F:CV_64F, 1.0/std.at<double>(i, 0), -1*mean.at<double>(i, 0)/std.at<double>(i, 0));
  }
  
  // copy image and label to output blobs
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_blob2->mutable_cpu_data();
  for(size_t i=0;i<img_channels;i++)
  {
    caffe_copy(height*width, (Dtype*)split_img[i].data, transformed_data+i*height*width);
  }
  for(size_t i = 0; i < points.size(); i++)
  {
    transformed_label[i] = points[i];
  }
}

template <typename Dtype>
void AugDataTransformer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
unsigned AugDataTransformer<Dtype>::Rand(unsigned n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((unsigned)(*rng)() % n);
}

template <typename Dtype>
double AugDataTransformer<Dtype>::m_rand_gen(const MRNGParameter& para, double &value){
  if(para.type() == MRNGParameter_MRNGType_NONE)
    return value;
  else if(para.type() == MRNGParameter_MRNGType_UNIFORM){
    const double min = para.min();
    const double max = para.max();
    CHECK_LE(min, max);
    value = min + (max - min) * (double)Rand((unsigned)(~0)) / (unsigned)(~0);
    return value;
  }
  else if(para.type() == MRNGParameter_MRNGType_NORMAL){
    const double mean = para.mean();
    const double std = para.std();
    const double min = para.min();
    const double max = para.max();
    CHECK_GE(std, 0);
    CHECK_LE(min, mean);
    CHECK_LE(mean, max);
    
    double V1, V2, S;
    double X;
    do{ 
      do {
         double U1 = (double)Rand((unsigned)(~0)) / (unsigned)(~0);
         double U2 = (double)Rand((unsigned)(~0)) / (unsigned)(~0);
       
         V1 = 2 * U1 - 1;
         V2 = 2 * U2 - 1;
         S = V1 * V1 + V2 * V2;
      }while(S >= 1 || S == 0);
           
      X = V1 * sqrt(-2 * log(S) / S);
      value = X * std + mean;
    }while(value < min || value > max);
    return value;
  }
  CHECK(0);
  return 0;
}

INSTANTIATE_CLASS(AugDataTransformer);

}  // namespace caffe
