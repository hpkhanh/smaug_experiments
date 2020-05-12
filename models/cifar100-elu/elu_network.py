#!/usr/bin/env python
#
# Examples for creating the CIFAR100-ELU network.
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
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_elu_model():
  with Graph(name="elu_ref", backend="Reference", mem_policy=AllDma) as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((1, 32, 32, 3)))
    conv0_stack0_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((192, 5, 5, 3)))
    conv0_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((192, 1, 1, 192)))
    conv1_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((240, 3, 3, 192)))
    conv0_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((240, 1, 1, 240)))
    conv1_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((260, 2, 2, 240)))
    conv0_stack3_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((260, 1, 1, 260)))
    conv1_stack3_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((280, 2, 2, 260)))
    conv0_stack4_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((280, 1, 1, 280)))
    conv1_stack4_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((300, 2, 2, 280)))
    conv0_stack5_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((300, 1, 1, 300)))
    conv0_stack6_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((100, 1, 1, 300)))

    act = input_data(input_tensor, name="input")
    act = convolution(
        act, conv0_stack0_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack0")
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool_stack0")
    act = convolution(
        act, conv0_stack1_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack1")
    act = convolution(
        act, conv1_stack1_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv1_stack1")
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool_stack1")
    act = convolution(
        act, conv0_stack2_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack2")
    act = convolution(
        act, conv1_stack2_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv1_stack2")
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool_stack2")
    act = convolution(
        act, conv0_stack3_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack3")
    act = convolution(
        act, conv1_stack3_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv1_stack3")
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool_stack3")
    act = convolution(
        act, conv0_stack4_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack4")
    act = convolution(
        act, conv1_stack4_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack4")
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2], name="pool_stack4")
    act = convolution(
        act, conv0_stack5_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack5")
    act = convolution(
        act, conv0_stack6_tensor, stride=[1, 1], padding="same", activation=ELU,
        name="conv0_stack6")
    return graph

if __name__ != "main":
  graph = create_elu_model()
  graph.print_summary()
  graph.write_graph()
