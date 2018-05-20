/*
 * GroupNorm Layer
 *
 * Created on: March 24, 2018
 * Author: hujie
 */

#ifndef CAFFE_GROUP_NORM_LAYER_HPP_
#define CAFFE_GROUP_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class GroupNormLayer : public Layer<Dtype> {
 public:
  explicit GroupNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GroupNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    return DiagonalAffineMap<Dtype>::identity(2);
  }

#ifdef USE_MMPL
  virtual inline void set_param_sync() {
    this->param_sync_.clear();
    this->param_sync_.resize(2, Caffe::DIFF);
  }
#endif

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int group_;
  Dtype eps_;
  Blob<Dtype> save_mean_;
  Blob<Dtype> save_inv_std_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_GROUP_NORM_LAYER_HPP_
