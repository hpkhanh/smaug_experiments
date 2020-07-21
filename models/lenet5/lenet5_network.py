#!/usr/bin/env python

"""Create the LeNet-5 network."""

import numpy as np

from smaug.core import types_pb2
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import data_op
from smaug.python.ops import nn_ops

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_lenet5_model():
  with Graph(name="lenet5_ref", backend="Reference") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=types_pb2.NHWC, tensor_data=generate_random_data(
            (1, 28, 28, 1)))
    conv0_tensor = Tensor(
        data_layout=types_pb2.NHWC, tensor_data=generate_random_data(
            (32, 3, 3, 1)))
    conv1_tensor = Tensor(
        data_layout=types_pb2.NHWC, tensor_data=generate_random_data(
            (32, 3, 3, 32)))
    fc0_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((128, 4608)))
    fc1_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((10, 128)))

    act = data_op.input_data(input_tensor, name="input")
    act = nn_ops.convolution(
        act, conv0_tensor, stride=[1, 1], padding="valid",
        activation=types_pb2.ReLU, name="conv0")
    act = nn_ops.convolution(
        act, conv1_tensor, stride=[1, 1], padding="valid",
        activation=types_pb2.ReLU, name="conv1")
    act = nn_ops.max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool")
    act = nn_ops.mat_mul(act, fc0_tensor, activation=types_pb2.ReLU, name="fc0")
    act = nn_ops.mat_mul(act, fc1_tensor, name="fc1")
    return graph

if __name__ != "main":
  graph = create_lenet5_model()
  graph.print_summary()
  graph.write_graph()
