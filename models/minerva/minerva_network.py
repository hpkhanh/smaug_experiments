#!/usr/bin/env python

"""Create the Minerva network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_minerva_model():
  with sg.Graph(name="minerva_smv", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    # input_tensor = sg.Tensor(
    #     data_layout=sg.NHWC, tensor_data=generate_random_data((1, 28, 28, 1)))
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=np.ones((1, 28, 28, 1), np.float16)*0.005)
    fc0_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.ones((256, 784), np.float16)*0.005)
    fc1_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.ones((256, 256), np.float16)*0.005)
    fc2_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.ones((256, 256), np.float16)*0.005)
    fc3_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.ones((10, 256), np.float16)*0.005)

    act = sg.input_data(input_tensor)
    act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc1_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc2_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc3_tensor)
    return graph

if __name__ != "main":
  graph = create_minerva_model()
  graph.print_summary()
  graph.write_graph()
