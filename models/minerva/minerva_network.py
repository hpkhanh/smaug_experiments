#!/usr/bin/env python
#
# Python code for creating the Minerva network.
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

def create_minerva_model():
  with Graph(name="minerva_smv", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((1, 28, 28, 1)))
    fc0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (256, 784)))
    fc1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (256, 256)))
    fc2_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (256, 256)))
    fc3_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (10, 256)))

    act = input_data("input", input_tensor)
    act = mat_mul("fc0", act, fc0_tensor, activation=ReLU)
    act = mat_mul("fc1", act, fc1_tensor, activation=ReLU)
    act = mat_mul("fc2", act, fc2_tensor, activation=ReLU)
    act = mat_mul("fc3", act, fc3_tensor)
    return graph

if __name__ != "main":
  graph = create_minerva_model()
  graph.print_summary()
  graph.write_graph()
