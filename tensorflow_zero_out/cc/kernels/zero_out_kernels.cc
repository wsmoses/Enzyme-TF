/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  
  string filename;
  string function;
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));
  }

  void Compute(OpKernelContext* context) override {


    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    // Set all but the first element of the output tensor to 0.
    if (input_tensor.dtype() == DT_FLOAT) {
        float* d = (float*)output_tensor->data();
        for(int i=0; i<input_tensor.shape().num_elements(); i++) {
            d[i] = 2;
        }
    } else {
        double* d = (double*)output_tensor->data();
        for(int i=0; i<input_tensor.shape().num_elements(); i++) {
            d[i] = 3;
        }
    }


  }
};

REGISTER_KERNEL_BUILDER(Name("Enzyme").Device(DEVICE_CPU), ZeroOutOp);

class EnzymeG : public OpKernel {
 public:
    string filename;
    string function;
  explicit EnzymeG(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();
    const Tensor& outp = context->input(1);
     

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    int32_t* d = (int32_t*)output_tensor->data();
    for(int i=0; i<N; i++) {
        d[i] = 2;
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("EnzymeG").Device(DEVICE_CPU), EnzymeG);
