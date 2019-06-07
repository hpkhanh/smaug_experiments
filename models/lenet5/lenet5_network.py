#!/usr/bin/env python
#
# Python code for creating the LeNet5 network.
#

import sys
sys.path.append('../../../nnet_lib/src/python/')
import numpy as np
from graph import *
from tensor import *
from ops import *
from types_pb2 import *

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_lenet5_model():
  with Graph(name="lenet5_smv", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((1, 28, 28, 1)))
    conv0_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((32, 3, 3, 1)))
    conv1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((32, 3, 3, 32)))
    fc0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (128, 4608)))
    fc1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (10, 128)))

    act = input_data("input", input_tensor)
    act = convolution("conv0", act, conv0_tensor, stride=[1, 1],
                      padding="valid", activation=ReLU)
    act = convolution("conv1", act, conv1_tensor, stride=[1, 1],
                      padding="valid", activation=ReLU)
    act = max_pool("pool", act, pool_size=[2, 2], stride=[2, 2])
    act = mat_mul("fc0", act, fc0_tensor, activation=ReLU)
    act = mat_mul("fc1", act, fc1_tensor)
    return graph

if __name__ != "main":
  graph = create_lenet5_model()
  graph.print_summary()
  graph.write_graph()
