#!/usr/bin/env python
#
# Examples for creating the CIFAR100-Large-ELU network.
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
  return (r.rand(*shape) * 0.003).astype(np.float16)

def create_elu_model():
  with Graph(name="large_elu_smv", backend="SMV", mem_policy=AllDma) as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((1, 32, 32, 3)))
    conv0_stack0_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((384, 3, 3, 3)))
    conv0_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((384, 1, 1, 384)))
    conv1_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((384, 2, 2, 384)))
    conv2_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((640, 2, 2, 384)))
    conv3_stack1_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((640, 2, 2, 640)))
    conv0_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((640, 1, 1, 640)))
    conv1_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((768, 2, 2, 640)))
    conv2_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((768, 2, 2, 768)))
    conv3_stack2_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((768, 2, 2, 768)))
    conv0_stack3_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((768, 1, 1, 768)))
    conv1_stack3_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((896, 2, 2, 768)))
    conv2_stack3_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((896, 2, 2, 896)))
    conv0_stack4_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((896, 3, 3, 896)))
    conv1_stack4_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((1024, 2, 2, 896)))
    conv2_stack4_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((1024, 2, 2, 1024)))
    conv0_stack5_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((1024, 1, 1, 1024)))
    conv1_stack5_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((1152, 2, 2, 1024)))
    conv0_stack6_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((1152, 1, 1, 1152)))
    conv0_stack7_tensor = Tensor(
        data_layout=NHWC, tensor_data=generate_random_data((100, 1, 1, 1152)))

    act = input_data("input", input_tensor)
    # Stack 0
    act = convolution("conv0_stack0", act, conv0_stack0_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack0", act, pool_size=[2, 2], stride=[2, 2])
    # Stack 1
    act = convolution("conv0_stack1", act, conv0_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack1", act, conv1_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv2_stack1", act, conv2_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv3_stack1", act, conv3_stack1_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack1", act, pool_size=[2, 2], stride=[2, 2])
    # Stack 2
    act = convolution("conv0_stack2", act, conv0_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack2", act, conv1_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv2_stack2", act, conv2_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv3_stack2", act, conv3_stack2_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack2", act, pool_size=[2, 2], stride=[2, 2])
    # Stack 3
    act = convolution("conv0_stack3", act, conv0_stack3_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack3", act, conv1_stack3_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv2_stack3", act, conv2_stack3_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack3", act, pool_size=[2, 2], stride=[2, 2])
    # Stack 4
    act = convolution("conv0_stack4", act, conv0_stack4_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack4", act, conv1_stack4_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv2_stack4", act, conv2_stack4_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = max_pool("pool_stack4", act, pool_size=[2, 2], stride=[2, 2])
    # Stack 5
    act = convolution("conv0_stack5", act, conv0_stack5_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    act = convolution("conv1_stack5", act, conv1_stack5_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    # Stack 6
    act = convolution("conv0_stack6", act, conv0_stack6_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    # Stack 7
    act = convolution("conv0_stack7", act, conv0_stack7_tensor, stride=[1, 1],
                      padding="same", activation=ELU)
    return graph

if __name__ != "main":
  graph = create_elu_model()
  graph.print_summary()
  graph.write_graph()
