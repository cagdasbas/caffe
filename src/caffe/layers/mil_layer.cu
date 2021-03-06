#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxMILForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_label, 
        const int box_num, const int batch_size, Dtype* top_data, Dtype* top_label, int* mask) {
    int mindex = 0, start_offset = 0, topindex = 0, bottom_label_index = 0;
    CUDA_KERNEL_LOOP(index, nthreads) {
        for (int bni = 0; bni < box_num ; bni++){
            start_offset = bni * nthreads * batch_size;
            for (int bi = 0; bi < batch_size ; bi++){
                //LOG(INFO) << "bottom_data[" << i << "*" << class_ << "+" << c << "]=" <<  i*class_ + c;
                mindex = start_offset + bi * nthreads + index;
                topindex = bni * nthreads + index;
                bottom_label_index = bni * batch_size + bi;
                if(bottom_data[mindex] > top_data[topindex]){
                    top_data[topindex] = bottom_data[mindex]; // set data
                    top_label[bni] = bottom_label[bottom_label_index];
                    mask[topindex] = mindex;
                }
                /*if (bottom_label[bottom_label_index] != top_label[b]){ // check if label consistency in batch
                    LOG(FATAL) << "Label in a batch cannot be different";
                }*/
            }
        }
    }
}


template <typename Dtype>
void MILLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* top_label = top[1]->mutable_gpu_data();
  
  const int top_count = top[0]->count();
  caffe_gpu_set(top_count, Dtype(-FLT_MAX), top_data);

  int* mask = max_idx_.mutable_gpu_data(); 
  int box_num = instance_ / batch_size_;
  // We'll output the mask to top[1] if it's of size >1.
  switch (this->layer_param_.mil_param().pool()) {
  case MILParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxMILForward<Dtype><<<CAFFE_GET_BLOCKS(class_), CAFFE_CUDA_NUM_THREADS>>>(
        class_, bottom_data, bottom_label, box_num, batch_size_, top_data, top_label, mask);
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
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads,
        const int* const mask, const Dtype* const top_diff,
        Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      bottom_diff[mask[index]] = top_diff[index];
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* mask = max_idx_.gpu_data();
  const int top_count = top[0]->count();
  switch (this->layer_param_.mil_param().pool()) {
  case MILParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, mask, top_diff, bottom_diff);
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
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(MILLayer);


}  // namespace caffe
