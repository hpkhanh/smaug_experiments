#!/usr/bin/env python

"""Example for creating the CIFAR10-CNN network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def generate_unified_data(shape, value):
  return (np.ones(shape,np.float16)*value)

def create_cnn_model():
  with sg.Graph(name="cnn_smv", backend="SMV",
                mem_policy=sg.AllDma) as graph:
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_unified_data((1, 32, 32, 3), 0.05))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_unified_data((32, 3, 3, 3), 0.05))
    bn0_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn0_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn0_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn0_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    conv1_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_unified_data((32, 3, 3, 32), 0.05))
    bn1_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn1_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn1_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    bn1_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 32), 0.05))
    conv2_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_unified_data((64, 3, 3, 32), 0.05))
    conv3_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_unified_data((64, 3, 3, 64), 0.05))
    bn2_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 64), 0.05))
    bn2_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 64), 0.05))
    bn2_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 64), 0.05))
    bn2_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((1, 64), 0.05))
    fc0_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((512, 4096), 0.05))
    fc1_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_unified_data((10, 512), 0.05))

    act = sg.input_data(input_tensor)
    act = sg.nn.convolution(
        act, conv0_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.batch_norm(
        act, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor)
    act = sg.nn.convolution(
        act, conv1_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.batch_norm(
        act, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor)
    act = sg.nn.convolution(
        act, conv2_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv3_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.batch_norm(
        act, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor)
    act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc1_tensor)
    return graph

if __name__ != "main":
  graph = create_cnn_model()
  graph.print_summary()
  graph.print_nodes("layers.cfg")
  graph.write_graph()
