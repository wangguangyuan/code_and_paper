#include <algorithm>
#include <vector>

#include "caffe/layers/bilinear_layer.hpp"

namespace caffe {

#define THREAD_BLOCK_SIZE 512

template <typename Dtype>
__global__ void BilinearForward(const int n, const int num, const int c1, const int c2,
    const int h, const int w, const int spatial_dim, const Dtype* x, const Dtype* y, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int num_idx = index / (c1*c2*spatial_dim);
    int c_idx = index / spatial_dim % (c1*c2);
    int c1_idx = c_idx / c2;
    int c2_idx = c_idx % c2;
    int h_idx = index % spatial_dim / w; 
    int w_idx = index % w;
    out[index] = x[(num_idx * c1 + c1_idx) * spatial_dim + h_idx * w + w_idx] 
        * y[(num_idx * c2 + c2_idx) * spatial_dim + h_idx * w + w_idx];
  }
}

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int c1 = bottom[0]->channels();
  const int c2 = bottom[1]->channels();
  const int h = top[0]->height();
  const int w = top[0]->width();
  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* y = bottom[1]->gpu_data();
  Dtype* out = top[0]->mutable_gpu_data();
  
  BilinearForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, num, c1, c2, h, w, h*w, x, y, out);
}

template <typename Dtype>
__global__ void BilinearBackward(const int c1, const int c2,
    const int h, const int w, const int spatial_dim,
    const Dtype* in, const Dtype* top_diff, Dtype* bottom_diff) {
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  
  int num_idx = blockIdx.x / (c1*spatial_dim);
  int c_idx = blockIdx.x / spatial_dim % c1;
  int h_idx = blockIdx.x % spatial_dim / w;
  int w_idx = blockIdx.x % w;

  for (int i = threadIdx.x; i < c2; i += blockDim.x) {
    int cout_idx = c_idx * c2 + i;
    buffer[i % THREAD_BLOCK_SIZE] += 
        top_diff[(num_idx * c1 * c2 + cout_idx) * spatial_dim + h_idx * w + w_idx] 
        * in[(num_idx * c2 + i) * spatial_dim + h_idx * w + w_idx];
  }
  
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
    __syncthreads();
  }
 
  bottom_diff[blockIdx.x] = buffer[0]; 
}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int c1 = bottom[0]->channels();
  const int c2 = bottom[1]->channels();
  const int h = top[0]->height();
  const int w = top[0]->width();
  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* y = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    BilinearBackward<Dtype><<<bottom[0]->count(), THREAD_BLOCK_SIZE>>>(
        c1, c2, h, w, h*w, y, top_diff, bottom_diff); 
  }
  if (propagate_down[1]) {
    Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
    BilinearBackward<Dtype><<<bottom[1]->count(), THREAD_BLOCK_SIZE>>>(
       c2, c1, h, w, h*w, x, top_diff, bottom_diff); 
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearLayer);

}  // namespace caffe
