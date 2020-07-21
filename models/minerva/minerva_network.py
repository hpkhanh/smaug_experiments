#!/usr/bin/env python

"""Create the Minerva network."""

import numpy as np

from smaug.core import types_pb2
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import data_op
from smaug.python.ops import nn_ops

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_minerva_model():
  with Graph(name="minerva_ref", backend="Reference") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=types_pb2.NHWC, tensor_data=generate_random_data(
            (1, 28, 28, 1)))
    fc0_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((256, 784)))
    fc1_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((256, 256)))
    fc2_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((256, 256)))
    fc3_tensor = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((10, 256)))

    act = data_op.input_data(input_tensor, "input")
    act = nn_ops.mat_mul(act, fc0_tensor, activation=types_pb2.ReLU, name="fc0")
    act = nn_ops.mat_mul(act, fc1_tensor, activation=types_pb2.ReLU, name="fc1")
    act = nn_ops.mat_mul(act, fc2_tensor, activation=types_pb2.ReLU, name="fc2")
    act = nn_ops.mat_mul(act, fc3_tensor, name="fc3")
    return graph

if __name__ != "main":
  graph = create_minerva_model()
  graph.print_summary()
  graph.write_graph()
