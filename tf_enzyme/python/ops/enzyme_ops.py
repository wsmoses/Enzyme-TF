# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use zero_out ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

enzyme_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_enzyme_ops.so'))

def enzyme(*args, filename, function):
    return enzyme_ops.enzyme(args, filename=filename, function=function)

@ops.RegisterGradient("Enzyme")
def _enzyme_grad(op, grad):
    return enzyme_ops.enzyme_g(list(op.inputs) + [grad], filename=op.attrs[op.attrs.index("filename")+1], function=op.attrs[op.attrs.index("function")+1], M=len(op.inputs))
