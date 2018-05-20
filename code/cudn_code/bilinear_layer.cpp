#include <algorithm>
#include <vector>

#include "caffe/layers/bilinear_layer.hpp"

namespace caffe {

template <typename Dtype>
void BilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count()/bottom[0]->channels(), 
      bottom[1]->count()/bottom[1]->channels()) 
      << "The shape of bottom[0] must equal to bottom[1] except for channel dimension.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] *= bottom[1]->channels();
  top[0]->Reshape(top_shape); 
}

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(BilinearLayer);
#endif

INSTANTIATE_CLASS(BilinearLayer);
REGISTER_LAYER_CLASS(Bilinear);

}  // namespace caffe
