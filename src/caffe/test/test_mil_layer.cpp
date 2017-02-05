#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mil_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MILLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MILLayerTest()
      : blob_bottom_data(new Blob<Dtype>()),
        blob_bottom_label(new Blob<Dtype>()),
        blob_top_data(new Blob<Dtype>()),
        blob_top_label(new Blob<Dtype>()){ }
 
   virtual void SetUp(){
    Caffe::set_random_seed(1701);
    vector<int> shape_bottom(2);
    shape_bottom[0] = 8;
    shape_bottom[1] = 3;
    
    blob_bottom_data->Reshape(shape_bottom);
    shape_bottom.resize(1);
    shape_bottom[0] = 8;
    blob_bottom_label->Reshape(shape_bottom);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data);
    //filler.Fill(this->blob_bottom_label);
    blob_bottom_label->mutable_cpu_data()[0] = 3;
    blob_bottom_label->mutable_cpu_data()[1] = 3;
    blob_bottom_label->mutable_cpu_data()[2] = 2;
    blob_bottom_label->mutable_cpu_data()[3] = 2;
    blob_bottom_label->mutable_cpu_data()[4] = 1;
    blob_bottom_label->mutable_cpu_data()[5] = 1;
    blob_bottom_label->mutable_cpu_data()[6] = 4;
    blob_bottom_label->mutable_cpu_data()[7] = 4;
    blob_bottom_vec_.push_back(blob_bottom_data);
    blob_bottom_vec_.push_back(blob_bottom_label);
    blob_top_vec_.push_back(blob_top_data);
    blob_top_vec_.push_back(blob_top_label);
  }
  virtual ~MILLayerTest() {
    delete blob_bottom_data;
    delete blob_bottom_label;
    delete blob_top_data;
    delete blob_top_label;
  }
  Blob<Dtype>* const blob_bottom_data;
  Blob<Dtype>* const blob_bottom_label;
  Blob<Dtype>* const blob_top_data;
  Blob<Dtype>* const blob_top_label;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    MILParameter* mil_param = layer_param.mutable_mil_param();
    mil_param->set_pool(MILParameter_PoolMethod_MAX);
    mil_param->set_batch_size(2);
    vector<int> shape_bottom(2);
    shape_bottom[0] = 8;
    shape_bottom[1] = 3;
    blob_bottom_data->Reshape(shape_bottom);
    shape_bottom.resize(1);
    shape_bottom[0] = 8;
    blob_bottom_label->Reshape(shape_bottom);
    // Input: 8x3 channels of:
    //     [7 6 4]
    //     [2 3 7]
    //     [9 9 3]
    //     [8 8 7]
    //     [2 3 4]
    //     [2 7 4]
    //     [3 2 3]
    //     [5 7 8]
    // Created with matlab randi() function
    
    // initialize bottom data (batch of prediction)
    blob_bottom_data->mutable_cpu_data()[0] = 7;
    blob_bottom_data->mutable_cpu_data()[1] = 6;
    blob_bottom_data->mutable_cpu_data()[2] = 4;
    blob_bottom_data->mutable_cpu_data()[3] = 2;
    blob_bottom_data->mutable_cpu_data()[4] = 3;
    blob_bottom_data->mutable_cpu_data()[5] = 7;
    blob_bottom_data->mutable_cpu_data()[6] = 9;
    blob_bottom_data->mutable_cpu_data()[7] = 9;
    blob_bottom_data->mutable_cpu_data()[8] = 3;
    blob_bottom_data->mutable_cpu_data()[9] = 8;
    blob_bottom_data->mutable_cpu_data()[10] = 8;
    blob_bottom_data->mutable_cpu_data()[11] = 7;
    blob_bottom_data->mutable_cpu_data()[12] = 2;
    blob_bottom_data->mutable_cpu_data()[13] = 3;
    blob_bottom_data->mutable_cpu_data()[14] = 4;
    blob_bottom_data->mutable_cpu_data()[15] = 2;
    blob_bottom_data->mutable_cpu_data()[16] = 7;
    blob_bottom_data->mutable_cpu_data()[17] = 4;
    blob_bottom_data->mutable_cpu_data()[18] = 3;
    blob_bottom_data->mutable_cpu_data()[19] = 2;
    blob_bottom_data->mutable_cpu_data()[20] = 3;
    blob_bottom_data->mutable_cpu_data()[21] = 5;
    blob_bottom_data->mutable_cpu_data()[22] = 7;
    blob_bottom_data->mutable_cpu_data()[23] = 8;
    
    // initizalize bottom label (label for each batch element)
    blob_bottom_label->mutable_cpu_data()[0] = 3;
    blob_bottom_label->mutable_cpu_data()[1] = 3;
    blob_bottom_label->mutable_cpu_data()[2] = 2;
    blob_bottom_label->mutable_cpu_data()[3] = 2;
    blob_bottom_label->mutable_cpu_data()[4] = 1;
    blob_bottom_label->mutable_cpu_data()[5] = 1;
    blob_bottom_label->mutable_cpu_data()[6] = 4;
    blob_bottom_label->mutable_cpu_data()[7] = 4;
    
    MILLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data->num(), 4);
    EXPECT_EQ(this->blob_top_data->channels(), 3);
    EXPECT_EQ(this->blob_top_label->num(), 4);
    EXPECT_EQ(this->blob_top_label->channels(), 1);
    
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected data output: 4x3 channels of:
    //     [7 6 7]
    //     [9 9 7]
    //     [2 7 4]
    //     [5 7 8]
    EXPECT_EQ(blob_top_data->cpu_data()[0], 7);
    EXPECT_EQ(blob_top_data->cpu_data()[1], 6);
    EXPECT_EQ(blob_top_data->cpu_data()[2], 7);
    EXPECT_EQ(blob_top_data->cpu_data()[3], 9);
    EXPECT_EQ(blob_top_data->cpu_data()[4], 9);
    EXPECT_EQ(blob_top_data->cpu_data()[5], 7);
    EXPECT_EQ(blob_top_data->cpu_data()[6], 2);
    EXPECT_EQ(blob_top_data->cpu_data()[7], 7);
    EXPECT_EQ(blob_top_data->cpu_data()[8], 4);
    EXPECT_EQ(blob_top_data->cpu_data()[9], 5);
    EXPECT_EQ(blob_top_data->cpu_data()[10], 7);
    EXPECT_EQ(blob_top_data->cpu_data()[11], 8);
    
    // Expected label output: 4x1 channels of:
    //     [3]
    //     [1]
    //     [2]
    //     [4]
    EXPECT_EQ(blob_top_label->cpu_data()[0], 3);
    EXPECT_EQ(blob_top_label->cpu_data()[1], 2);
    EXPECT_EQ(blob_top_label->cpu_data()[2], 1);
    EXPECT_EQ(blob_top_label->cpu_data()[3], 4);
  }
};

TYPED_TEST_CASE(MILLayerTest, TestDtypesAndDevices);

TYPED_TEST(MILLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MILParameter* mil_param = layer_param.mutable_mil_param();
  mil_param->set_pool(MILParameter_PoolMethod_MAX);
  mil_param->set_batch_size(2);
  MILLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data->num(), 4);
  EXPECT_EQ(this->blob_top_data->channels(), 3);
  EXPECT_EQ(this->blob_top_label->num(), 4);
  EXPECT_EQ(this->blob_top_label->channels(), 1);
}


/*TYPED_TEST(MILLayerTest, PrintBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MILParameter* mil_param = layer_param.mutable_mil_param();
  mil_param->set_pool(MILParameter_PoolMethod_MAX);
  MILLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_data->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_data->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_data->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_data->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_data->count(); ++i) {
    this->blob_top_data->mutable_cpu_diff()[i] = i;
  }
  vector<bool> propagate_down;
  propagate_down[0] = true;
  propagate_down[1] = false;
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_data->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_data->cpu_diff()[i] << endl;
  }
}*/

TYPED_TEST(MILLayerTest, TestForwardMax) {
  this->TestForwardSquare();
}

TYPED_TEST(MILLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MILParameter* mil_param = layer_param.mutable_mil_param();
  mil_param->set_pool(MILParameter_PoolMethod_MAX);
  mil_param->set_batch_size(2);
  MILLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_, 0);
}

}  // namespace caffe
