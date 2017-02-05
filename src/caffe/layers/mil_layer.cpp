#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void MILLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MILParameter pool_param = this->layer_param_.mil_param();
  CHECK(pool_param.pool()
          == MILParameter_PoolMethod_AVE
          || this->layer_param_.mil_param().pool()
          == MILParameter_PoolMethod_MAX)
          << "MIL implemented only for average and max pooling.";
  batch_size_ = pool_param.batch_size();
}

template <typename Dtype>
void MILLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->num_axes()) << "Input must have 2 axes, "
      << "corresponding to (batch_size, class_size)";
  // assume after last fully connected
  instance_ = bottom[0]->shape()[0];
  CHECK_EQ(instance_ % batch_size_, 0) << "Instance count must be integer" 
          <<  "multiple of pool batch size";
  
  class_ = bottom[0]->shape()[1];
  vector<int> shape(2);
  shape[0] = instance_ / batch_size_;
  shape[1] = class_;
  //top[0] -> pooled probabilities
  top[0]->Reshape(shape);
  //max_idx_ -> maximum indices
  max_idx_.Reshape(shape);
  //top[1] -> pooled labels 
  vector<int> label_shape_(1);
  label_shape_[0] = instance_ / batch_size_;
  top[1]->Reshape(label_shape_);
}

template <typename Dtype>
void MILLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  
  const int top_count = top[0]->count();
  const int top_count_label = top[1]->count();
  int* mask = max_idx_.mutable_cpu_data(); 
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  int box_num = instance_ / batch_size_;
  int mindex = 0, start_offset = 0, topindex = 0, bottomlabelindex = 0;
  switch (this->layer_param_.mil_param().pool()) {
  case MILParameter_PoolMethod_MAX:
      
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    caffe_set(top_count_label, Dtype(0), top_label);
    for (int bni = 0; bni < box_num; bni++){
        start_offset = bni * class_ * batch_size_;
        for (int c = 0; c < class_; c++){
            for (int bi = 0; bi < batch_size_; bi++){
                mindex = start_offset + bi * class_ + c;
                topindex = bni * class_ + c;
                bottomlabelindex = bni * batch_size_ + bi;
                if(bottom_data[mindex] > top_data[topindex]){
                    top_data[topindex] = bottom_data[mindex]; // set data
                    top_label[bni] = bottom_label[bottomlabelindex];
                    mask[topindex] = mindex;
                }
                if (bottom_label[bottomlabelindex] != top_label[bni]){ // check if label consistency in batch
                    LOG(FATAL) << "Label in a batch cannot be different";
                }
            }
        }
    }
    break;
  case MILParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const int* mask = max_idx_.mutable_cpu_data();  // suppress warnings about uninitialized variables
  const int top_count = top[0]->count();
  switch (this->layer_param_.mil_param().pool()) {
  case MILParameter_PoolMethod_MAX:
    for(int c = 0; c < top_count ; c++){
        bottom_diff[mask[c]] = top_diff[c];
    }
    break;
  case MILParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case MILParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(MILLayer);
#endif

INSTANTIATE_CLASS(MILLayer);
REGISTER_LAYER_CLASS(MIL);

}  // namespace caffe
