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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Enzyme")
    .Input("inp: T")
    .Output("out: T")
    .Attr("T: {float, double}")
    .Attr("filename: string")
    .Attr("function: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("EnzymeG")
    .Input("inp: T")
    .Input("doutp: T")
    .Output("dinp: T")
    .Attr("T: {float, double}")
    .Attr("filename: string")
    .Attr("function: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

Status EnzymeGrad(const AttrSlice& attrs, FunctionDef* g) {
    *g = FunctionDefHelper::Define(
            //Name
            "EnzymeGrad",
            //Args
            {"inp: T", "doutp: T"},
            //Return
            {"dinp: T"},
            // Attr
            {"filename:string ", "function:string ", "T: {float, double}"},
            //Nodes
            {
                {{"doutp"}, "EnzymeG", {"inp", "doutp"}, {{"T", "$T"}, {"filename", "$filename"}, {"function", "$function"}}}
            });
}

REGISTER_OP_GRADIENT("Enzyme", EnzymeGrad);
