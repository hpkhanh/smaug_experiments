#!/usr/bin/env python
#
# Examples for creating the CIFAR10-VGG network.
#

import sys
sys.path.append('../../nnet_lib/src/python/')
import numpy as np
from graph import *
from tensor import *
from ops import *
from types_pb2 import *

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_vgg_model():
  with Graph(name="vgg_smv", backend="SMV") as graph:
    input_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((1, 32, 32, 3)))
    conv0_tensor = Tensor(
        data_layout=NHWC,
        tensor_data=generate_random_data((64, 3, 3, 3)))
    conv1_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (128, 3, 3, 64)))
    conv2_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (128, 3, 3, 128)))
    conv3_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (128, 3, 3, 128)))
    conv4_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (256, 3, 3, 128)))
    conv5_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (256, 3, 3, 256)))
    conv6_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (256, 3, 3, 256)))
    conv7_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (512, 3, 3, 256)))
    conv8_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (512, 3, 3, 512)))
    conv9_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (512, 3, 3, 512)))
    fc0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (512, 2048)))
    fc1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data((10,
                                                                          512)))

    act = input_data("input", input_tensor)
    act = convolution("conv0", act, conv0_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv1", act, conv1_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = max_pool("pool0", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv2", act, conv2_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv3", act, conv3_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = max_pool("pool1", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv4", act, conv4_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv5", act, conv5_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv6", act, conv6_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = max_pool("pool2", act, pool_size=[2, 2], stride=[2, 2])
    act = convolution("conv7", act, conv7_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv8", act, conv8_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = convolution("conv9", act, conv9_tensor, stride=[1, 1], padding="same",
                      activation=ReLU)
    act = max_pool("pool3", act, pool_size=[2, 2], stride=[2, 2])
    act = mat_mul("fc0", act, fc0_tensor, activation=ReLU)
    act = mat_mul("fc1", act, fc1_tensor)
    return graph

if __name__ != "main":
  graph = create_vgg_model()
  graph.print_summary()
  graph.write_graph()
