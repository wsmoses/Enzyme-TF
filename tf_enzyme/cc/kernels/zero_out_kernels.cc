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

#include <dlfcn.h>

class ZeroOutOp : public OpKernel {
 public:

  string filename;
  string function;
  void (*f)(void*, size_t, void*);
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));


    char buffer [L_tmpnam];
    tmpnam (buffer);
    int res;
    char data[1024];
    sprintf(data, "clang++ %s -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -O3 -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);

    printf("making buffer 2\n");

    char buffer2 [L_tmpnam];
    printf("making tm buffer 2\n");
    tmpnam (buffer2);
    printf("made buffer 2\n");

    sprintf(data, "clang++ -fPIC -shared %s.ll -o %s.so", buffer, buffer2);
    printf("running library - %s\n", data);
    res = system(data);
    printf("ran library - %s\n", data);
    assert(res == 0);

    char buffer3[L_tmpnam];
    sprintf(buffer3, "%s.so", buffer2);

    printf("running dlopen\n");
    void* lib = dlopen(buffer3, RTLD_LAZY);
    assert(lib);
    printf("running dlsym\n");
    void* sym = dlsym(lib, function.c_str());
    assert(sym);
    f = (void(*)(void*, size_t, void*))sym;
    //remove()
  }

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    printf("running compute\n");
    // Set all but the first element of the output tensor to 0.
    if (input_tensor.dtype() == DT_FLOAT) {
        f(input_tensor.data(), input_tensor.shape().num_elements(), output_tensor->data());
        /*
        float* d = (float*)output_tensor->data();
        for(int i=0; i<input_tensor.shape().num_elements(); i++) {
            d[i] = 2;
        }*/
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
    void (*diffef)(void*, void*, size_t, void*);
  explicit EnzymeG(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));

    int res;

    char buffer [L_tmpnam];
    tmpnam (buffer);
    char data[1024];
    sprintf(data, "clang++ %s -DTF_ENZYME=1 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -O3 -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);

    sprintf(data, "~/git/Enzyme/build/bin/opt %s.ll -load=%s -S -enzyme -O3 -o %s.ll", buffer, "/home/wmoses/git/Enzyme/enzyme/build-dbg/Enzyme/LLVMEnzyme-7.so", buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);


    printf("making buffer 2\n");

    char buffer2 [L_tmpnam];
    printf("making tm buffer 2\n");
    tmpnam (buffer2);
    printf("made buffer 2\n");

    sprintf(data, "clang++ -fPIC -shared %s.ll -o %s.so", buffer, buffer2);
    printf("running library - %s\n", data);
    res = system(data);
    printf("ran library - %s\n", data);
    assert(res == 0);

    char buffer3[L_tmpnam];
    sprintf(buffer3, "%s.so", buffer2);

    printf("running dlopen\n");
    void* lib = dlopen(buffer3, RTLD_LAZY);
    assert(lib);
    std::string tofind = "diffe" + function;
    printf("running dlsym %s\n", tofind.c_str());
    void* sym = dlsym(lib, tofind.c_str());
    assert(sym);
    diffef = (void(*)(void*, void*, size_t, void*))sym;
    //remove()
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& outp = context->input(1);


    // Create an output tensor
    Tensor* dinp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &dinp));

    // Set all but the first element of the output tensor to 0.
    printf("running dcompute\n");
    // Set all but the first element of the output tensor to 0.
    if (input_tensor.dtype() == DT_FLOAT) {
        diffef(input_tensor.data(), dinp->data(), input_tensor.shape().num_elements(), outp.data());
        /*
        float* d = (float*)output_tensor->data();
        for(int i=0; i<input_tensor.shape().num_elements(); i++) {
            d[i] = 2;
        }*/
    } else {
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("EnzymeG").Device(DEVICE_CPU), EnzymeG);
