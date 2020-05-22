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
void* compile(std::string filename, std::string function) {
    char buffer [L_tmpnam];
    tmpnam (buffer);
    int res;
    char data[1024];
    sprintf(data, "clang++ %s -O3 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
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
    return sym;
}

void* diffecompile(std::string filename, std::string function) {
    int res;

    char buffer [L_tmpnam];
    tmpnam (buffer);
    char data[1024];
    sprintf(data, "clang++ -O3 %s -DTF_ENZYME=1 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
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
    return sym;
}

#include <ffi.h>
class ZeroOutOp : public OpKernel {
 public:

  string filename;
  string function;
  void* f;
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));
    f = compile(filename, function);
    //remove()
  }

  void Compute(OpKernelContext* context) override {

    // TODO generic output tensor

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, context->input(context->num_inputs()-1).shape(),
                                                     &output_tensor));
    //printf("running compute\n");

    // Describe the function arguments. Note that ffi_type_pointer is used
    // for any C pointer (the pointee type does not matter in the ABI).
    std::vector<ffi_type*> args;
    for(unsigned i=0; i<context->num_inputs(); i++) {
        args.push_back(&ffi_type_pointer);
        args.push_back(&ffi_type_uint64);
    }
    args.push_back(&ffi_type_pointer);

    // Describe the interface of add_data to libffi.
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size(), &ffi_type_void, args.data());
    if (status != FFI_OK) {
        fprintf(stderr, "ffi_prep_cif failed: %d\n", status);
        exit(1);
    }

    std::vector<void*>  datas(context->num_inputs());
    std::vector<size_t> sizes(context->num_inputs());
    std::vector<void*> avalues;

    for(unsigned i=0; i<context->num_inputs(); i++) {
        const Tensor& input_tensor = context->input(i);
        datas[i] = input_tensor.data();
        sizes[i] = input_tensor.shape().num_elements();
        avalues.push_back(&datas[i]);
        avalues.push_back(&sizes[i]);
        // TODO: if (input_tensor.dtype() == DT_FLOAT)
    }
    void* outd = output_tensor->data();
    avalues.push_back(&outd);

    ffi_call(&cif, FFI_FN(f), NULL, avalues.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Enzyme").Device(DEVICE_CPU), ZeroOutOp);

class EnzymeG : public OpKernel {
 public:
    string filename;
    string function;
    void* diffef;
  explicit EnzymeG(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename));
    OP_REQUIRES_OK(context, context->GetAttr("function", &function));
    diffef = diffecompile(filename, function);
  }

  void Compute(OpKernelContext* context) override {
    // Describe the function arguments. Note that ffi_type_pointer is used
    // for any C pointer (the pointee type does not matter in the ABI).
    std::vector<ffi_type*> args;
    for(unsigned i=0; i<context->num_inputs()-1; i++) {
        args.push_back(&ffi_type_pointer);
        args.push_back(&ffi_type_pointer);
        args.push_back(&ffi_type_uint64);
    }
    args.push_back(&ffi_type_pointer);

    // Describe the interface of add_data to libffi.
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size(), &ffi_type_void, args.data());
    if (status != FFI_OK) {
        fprintf(stderr, "ffi_prep_cif failed: %d\n", status);
        exit(1);
    }

    std::vector<void*>  datas(2*(context->num_inputs()-1));
    std::vector<size_t> sizes(context->num_inputs()-1);
    std::vector<void*> avalues;

    for(unsigned i=0; i<context->num_inputs()-1; i++) {
        const Tensor& input_tensor = context->input(i);
        datas[2*i] = input_tensor.data();

        Tensor* dinp = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(i, input_tensor.shape(), &dinp));
        datas[2*i+1] = dinp->data();
        //printf("data is aligned: %d:\n", dinp->IsAligned());

        sizes[i] = input_tensor.shape().num_elements();

        avalues.push_back(&datas[2*i]);
        avalues.push_back(&datas[2*i+1]);
        avalues.push_back(&sizes[i]);
        // TODO: if (input_tensor.dtype() == DT_FLOAT)
    }

    const Tensor& doutput_tensor = context->input(context->num_inputs()-1);
    void* outd = doutput_tensor.data();
    avalues.push_back(&outd);

    //printf("avalues.size()=%d args.size()=%d\n", avalues.size(), args.size());

    //printf("running dcompute\n");
    ffi_call(&cif, FFI_FN(diffef), NULL, avalues.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("EnzymeG").Device(DEVICE_CPU), EnzymeG);
