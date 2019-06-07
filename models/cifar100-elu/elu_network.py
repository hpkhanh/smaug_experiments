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
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_elu_model():
  with Graph(name="elu_smv", backend="SMV") as graph:
    graph.disable_layout_transform()
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

    act = input_data("input", input_tensor)
    act = convolution("conv0_stack0", act, conv0_stack0_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack0", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv0_stack1", act, conv0_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack1", act, conv1_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack1", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv0_stack2", act, conv0_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack2", act, conv1_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack2", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv0_stack3", act, conv0_stack3_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack3", act, conv1_stack3_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack3", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv0_stack4", act, conv0_stack4_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack4", act, conv1_stack4_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack4", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv0_stack5", act, conv0_stack5_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv0_stack6", act, conv0_stack6_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    return graph

if __name__ != "main":
  graph = create_elu_model()
  graph.print_summary()
  graph.write_graph()
