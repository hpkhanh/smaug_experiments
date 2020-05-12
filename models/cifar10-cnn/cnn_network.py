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
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_cnn_model():
  with Graph(name="cnn_ref", backend="Reference", mem_policy=AllDma) as graph:
    input_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (1, 32, 32, 3)))
    conv0_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (32, 3, 3, 3)))
    bn0_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn0_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn0_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn0_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    conv1_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (32, 3, 3, 32)))
    bn1_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn1_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn1_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    bn1_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 32)))
    conv2_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (64, 3, 3, 32)))
    conv3_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (64, 3, 3, 64)))
    bn2_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn2_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn2_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn2_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    fc0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (512, 4096)))
    fc1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data((10,
                                                                          512)))

    act = input_data(input_tensor)
    act = convolution(
        act, conv0_tensor, stride=[1, 1], padding="same", activation=ReLU)
    act = batch_norm(
        act, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor)
    act = convolution(
        act, conv1_tensor, stride=[1, 1], padding="same", activation=ReLU)
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = batch_norm(
        act, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor)
    act = convolution(
        act, conv2_tensor, stride=[1, 1], padding="same", activation=ReLU)
    act = convolution(
        act, conv3_tensor, stride=[1, 1], padding="same", activation=ReLU)
    act = max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = batch_norm(
        act, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor)
    act = mat_mul(act, fc0_tensor, activation=ReLU)
    act = mat_mul(act, fc1_tensor)
    return graph

if __name__ != "main":
  graph = create_cnn_model()
  graph.print_summary()
  graph.write_graph()
