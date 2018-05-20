/*
 * GroupNorm Layer
 *
 * Created on: March 24, 2018
 * Author: hujie
 */

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/group_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void GroupNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  GroupNormParameter param = this->layer_param_.group_norm_param();
  group_ = param.group();
  CHECK_EQ(bottom[0]->shape(1) % group_, 0);
  eps_ = param.eps();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(1, bottom[0]->shape(1), 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, bottom[0]->shape(1), 1, 1));
    shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(param.scale_filler()));
    scale_filler->Fill(this->blobs_[0].get());
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]); 
  save_mean_.Reshape(vector<int>(1, bottom[0]->num() * group_));
  save_inv_std_.ReshapeLike(save_mean_);
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(GroupNormLayer);
REGISTER_LAYER_CLASS(GroupNorm);
}  // namespace caffe
