#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np

from smaug.core.types_pb2 import *
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import data_op
from smaug.python.ops import math_ops
from smaug.python.ops import nn_ops
from smaug.python.ops import activation_ops

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float32)

def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names

  Returns:
    Output tensor for the block.
  """
  filters0, filters1, filters2 = filters
  conv_name_base = 'res' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

  # Tensors
  input_tensor_chans = input_tensor.dims(
      3) if input_tensor.shape.layout == NHWC else input_tensor.dims(1)
  conv0_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters0, 1, 1, input_tensor_chans)))
  bn0_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters0)))
  bn0_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters0)))
  bn0_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters0)))
  bn0_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters0)))
  conv1_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters1, kernel_size, kernel_size, filters0)))
  bn1_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters1)))
  bn1_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters1)))
  bn1_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters1)))
  bn1_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters1)))
  conv2_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, filters1)))
  bn2_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters2)))
  bn2_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters2)))
  bn2_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters2)))
  bn2_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
      (1, filters2)))

  x = nn_ops.convolution(
      input_tensor, conv0_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2a')
  x = nn_ops.batch_norm(
      x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
      activation=ReLU, name=bn_name_base + '_2a')
  x = nn_ops.convolution(
      x, conv1_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2b')
  x = nn_ops.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation=ReLU, name=bn_name_base + '_2b')
  x = nn_ops.convolution(
      x, conv2_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2c')
  x = nn_ops.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name=bn_name_base + '_2c')
  x = math_ops.add(x, input_tensor, name=add_name)
  x = activation_ops.relu(x, name=relu_name)
  return x

def conv_block(
    input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.

  Returns:
    Output tensor for the block.
  """
  filters0, filters1, filters2 = filters
  conv_name_base = 'res' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

  # Tensors
  input_tensor_chans = input_tensor.dims(
      3) if input_tensor.shape.layout == NHWC else input_tensor.dims(1)
  conv0_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters0, 1, 1, input_tensor_chans)))
  bn0_mean_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters0)))
  bn0_var_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters0)))
  bn0_gamma_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters0)))
  bn0_beta_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters0)))
  conv1_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters1, kernel_size, kernel_size, filters0)))
  bn1_mean_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters1)))
  bn1_var_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters1)))
  bn1_gamma_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters1)))
  bn1_beta_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters1)))
  conv2_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, filters1)))
  bn2_mean_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn2_var_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn2_gamma_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn2_beta_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  conv3_tensor = Tensor(
      data_layout=NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, input_tensor_chans)))
  bn3_mean_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn3_var_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn3_gamma_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))
  bn3_beta_tensor = Tensor(
      data_layout=NC, tensor_data=generate_random_data((1, filters2)))

  x = nn_ops.convolution(
      input_tensor, conv0_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2a')
  x = nn_ops.batch_norm(
      x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
      activation=ReLU)
  x = nn_ops.convolution(
      x, conv1_tensor, stride=strides, padding="same",
      name=conv_name_base + '_2b')
  x = nn_ops.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation=ReLU, name=bn_name_base + '_2b')
  x = nn_ops.convolution(
      x, conv2_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2c')
  x = nn_ops.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name=bn_name_base + '_2c')
  shortcut = nn_ops.convolution(
      input_tensor, conv3_tensor, stride=strides, padding="same",
      name=conv_name_base + '_1')
  shortcut = nn_ops.batch_norm(
      shortcut, bn3_mean_tensor, bn3_var_tensor, bn3_gamma_tensor,
      bn3_beta_tensor, name=bn_name_base + '_1')
  x = math_ops.add(x, shortcut, name=add_name)
  x = activation_ops.relu(x, name=relu_name)
  return x

def create_resnet50():
  with Graph(name="resnet_ref", backend="Reference") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (1, 225, 225, 3)))
    conv0_tensor = Tensor(data_layout=NHWC, tensor_data=generate_random_data(
        (64, 7, 7, 3)))
    bn0_mean_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn0_var_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn0_gamma_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    bn0_beta_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (1, 64)))
    fc_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (10, 7*7*2048)))

    x = data_op.input_data(input_tensor, name="input")
    x = nn_ops.convolution(
        x, conv0_tensor, stride=[2, 2], padding="same", name="conv0")
    x = nn_ops.batch_norm(
        x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
        activation=ReLU, name="bn0")
    x = nn_ops.max_pool(x, pool_size=[3, 3], stride=[2, 2], name="pool")

    # Four resnet blocks.
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x= nn_ops.mat_mul(x, fc_tensor, name="fc")
    return graph

if __name__ != "main":
  resnet50 = create_resnet50()
  resnet50.print_summary()
  resnet50.write_graph()
