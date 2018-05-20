/*
 * GroupNorm Layer
 *
 * Created on: March 24, 2018
 * Author: hujie
 */

#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "thrust/device_vector.h"
#include "caffe/layers/group_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GroupNormForward(const int g, const int c_per_g, 
    const int n_per_g, const int spatial_dim, const Dtype eps, 
    const Dtype* bottom_data, const Dtype* scale_data, 
    const Dtype* bias_data, Dtype* save_mean, Dtype* save_inv_std, 
    Dtype* top_data) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS]; 
  __shared__ Dtype scale_buffer[CAFFE_CUDA_NUM_THREADS]; 
  __shared__ Dtype bias_buffer[CAFFE_CUDA_NUM_THREADS]; 
  unsigned int tid = threadIdx.x;
  // static mean
  buffer[tid] = 0;
  for (unsigned int i = tid; i < n_per_g; i += blockDim.x) {
    buffer[tid] += bottom_data[blockIdx.x * n_per_g + i];
  }
  __syncthreads();
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) buffer[tid] += buffer[tid + i];
    __syncthreads();
  }
  Dtype mean = buffer[0] / n_per_g; 
  __syncthreads();
  // static var
  buffer[tid] = 0;
  for (unsigned int i = tid; i < n_per_g; i += blockDim.x) {
    Dtype sub = bottom_data[blockIdx.x * n_per_g + i] - mean; 
    buffer[tid] += sub * sub ;
  }
  __syncthreads();
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) buffer[tid] += buffer[tid + i];
    __syncthreads();
  }
  Dtype inv_std = Dtype(1) / sqrt(buffer[0] / n_per_g + eps); 
  if (tid == 0) {
    save_mean[blockIdx.x] = mean;
    save_inv_std[blockIdx.x] = inv_std;
  }
  // load scale data and bias data
  if (tid < c_per_g) {
    unsigned int c_idx = blockIdx.x % g * c_per_g + tid;
    scale_buffer[tid] = scale_data[c_idx];
    bias_buffer[tid] = bias_data[c_idx];
  }
  __syncthreads();
 
  // x scale + bias
  for (int i = tid; i < n_per_g; i += blockDim.x) {
    unsigned int idx = blockIdx.x * n_per_g + i;
    unsigned int c_idx = i / spatial_dim;
    top_data[idx] = (bottom_data[idx] - mean) * inv_std * 
        scale_buffer[c_idx] + bias_buffer[c_idx]; 
  }
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int c = bottom[0]->shape(1);
  const int c_per_g = c / group_;
  const int spatial_dim = bottom[0]->count(2);
  const int n_per_g = spatial_dim * c_per_g;
  const int n_g = bottom[0]->shape(0) * group_;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();  
  const Dtype* bias_data = this->blobs_[1]->gpu_data();
  Dtype* save_mean_data = save_mean_.mutable_gpu_data(); 
  Dtype* save_inv_std_data = save_inv_std_.mutable_gpu_data(); 
  Dtype* top_data = top[0]->mutable_gpu_data();
  CHECK_GE(CAFFE_CUDA_NUM_THREADS, c_per_g);

  GroupNormForward<Dtype><<<n_g, CAFFE_CUDA_NUM_THREADS>>>(
      group_, c_per_g, n_per_g, spatial_dim, eps_, bottom_data, 
      scale_data, bias_data, save_mean_data, save_inv_std_data, top_data);
}

template <typename Dtype>
__global__ void GroupNormWeightsBackward(const int c, const int g, const int c_per_g, 
    const int n_per_g, const int spatial_dim,  const int count, 
    const Dtype* save_mean, 
    const Dtype* save_inv_std, const Dtype* top_diff, const Dtype* bottom_data, 
    Dtype* scale_diff, Dtype* bias_diff) {
  __shared__ Dtype buffer_scale_diff[CAFFE_CUDA_NUM_THREADS]; 
  __shared__ Dtype buffer_bias_diff[CAFFE_CUDA_NUM_THREADS]; 
  unsigned int tid = threadIdx.x;
  buffer_scale_diff[tid] = 0;
  buffer_bias_diff[tid] = 0;

  for (unsigned int i = tid; i < count; i += blockDim.x) {
    unsigned int group_id = (i / spatial_dim * c + blockIdx.x) / c_per_g;
    Dtype mean = save_mean[group_id];
    Dtype inv_std = save_inv_std[group_id];
    unsigned int location = i / spatial_dim * spatial_dim * c +
        blockIdx.x * spatial_dim + (i % spatial_dim);
    Dtype norm_data = (bottom_data[location] - mean) * inv_std;
    buffer_scale_diff[tid] += top_diff[location] * norm_data;
    buffer_bias_diff[tid] += top_diff[location];
  }
  __syncthreads();
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) { 
      buffer_scale_diff[tid] += buffer_scale_diff[tid + i];
      buffer_bias_diff[tid] += buffer_bias_diff[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    scale_diff[blockIdx.x] += buffer_scale_diff[0];
    bias_diff[blockIdx.x] += buffer_bias_diff[0];
  }
}

template <typename Dtype>
__global__ void GroupNormDataBackward(const int g, const int c_per_g, 
    const int n_per_g, const int spatial_dim, const Dtype* bottom_data,
    const Dtype* scale_data,  const Dtype* save_mean, const Dtype* save_inv_std, 
    const Dtype* top_diff, Dtype* bottom_diff) {
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS]; 
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS]; 
  __shared__ Dtype scale_buffer[CAFFE_CUDA_NUM_THREADS]; 
  unsigned int tid = threadIdx.x;
  buffer1[tid] = 0;
  buffer2[tid] = 0;
 
  // load scale data
  if (tid < c_per_g) {
    unsigned int c_idx = blockIdx.x % g * c_per_g + tid;
    scale_buffer[tid] = scale_data[c_idx];
  }
  __syncthreads();
  for (unsigned int i = tid; i < n_per_g; i += blockDim.x) {
    unsigned int idx = blockIdx.x * n_per_g + i;
    Dtype cur_top_diff = top_diff[idx];
    Dtype mean = save_mean[blockIdx.x];
    Dtype inv_std = save_inv_std[blockIdx.x];
    Dtype norm_data = (bottom_data[idx] - mean) * inv_std;
    buffer1[tid] += cur_top_diff * scale_buffer[i / spatial_dim];
    buffer2[tid] += cur_top_diff * scale_buffer[i / spatial_dim] * norm_data; 
  }
  __syncthreads();
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) { 
      buffer1[tid] += buffer1[tid + i];
      buffer2[tid] += buffer2[tid + i];
    }
    __syncthreads();
  }
  for (unsigned int i = tid; i < n_per_g; i += blockDim.x) {
    unsigned int idx = blockIdx.x * n_per_g + i;
    Dtype mean = save_mean[blockIdx.x];
    Dtype inv_std = save_inv_std[blockIdx.x];
    Dtype norm_data = (bottom_data[idx] - mean) * inv_std;
    bottom_diff[idx] = (top_diff[idx] * scale_buffer[i / spatial_dim] 
        - (buffer1[0] + norm_data * buffer2[0]) / n_per_g) * inv_std;
    //if (blockIdx.x == 1) {
    //  printf(">>> %f %f %f %f %f %d", top_diff[idx] * scale_buffer[i / spatial_dim], buffer1[blockIdx.x], norm_data, buffer2[blockIdx.x], inv_std, n_per_g);
    //}
  }
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0] || this->param_propagate_down_[1] || 
      propagate_down[0]) {
    const int c = bottom[0]->shape(1);
    const int c_per_g = c / group_;
    const int spatial_dim = bottom[0]->count(2);
    const int n_per_g = spatial_dim * c_per_g;
    const int n_g = bottom[0]->shape(0) * group_;
    const int count = bottom[0]->count() / c;

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* scale_data = this->blobs_[0]->gpu_data();  
    const Dtype* save_mean_data = save_mean_.gpu_data(); 
    const Dtype* save_inv_std_data = save_inv_std_.gpu_data(); 
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();    
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();    
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
      GroupNormWeightsBackward<Dtype><<<c, CAFFE_CUDA_NUM_THREADS>>>(
          c, group_, c_per_g, n_per_g, spatial_dim, count, 
          save_mean_data, save_inv_std_data, top_diff, 
          bottom_data, scale_diff, bias_diff);
    }
    if (propagate_down[0]) {
      GroupNormDataBackward<Dtype><<<n_g, CAFFE_CUDA_NUM_THREADS>>>(
          group_, c_per_g, n_per_g, spatial_dim, bottom_data,
          scale_data, save_mean_data, save_inv_std_data, 
          top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GroupNormLayer);

}  // namespace caffe
