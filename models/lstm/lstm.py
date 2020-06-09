#!/usr/bin/env python
#
# Python code for creating the Minerva network.
#

import sys
import numpy as np

from smaug.python.graph import *
from smaug.python.tensor import *
from smaug.python.ops import *
from smaug.python.recurrent import *
from smaug.core.types_pb2 import *

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_lstm_model():
  with Graph(name="lstm_smv", backend="SMV") as graph:
    input_tensor = Tensor(
        data_layout=NTC, tensor_data=generate_random_data((1, 4, 32)))
    # Tensors and kernels are initialized as NC layout.
    # Layer 1 of LSTM.
    wf0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wi0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wc0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wo0_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    # Layer 2 of LSTM.
    wf1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wi1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wc1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))
    wo1_tensor = Tensor(data_layout=NC, tensor_data=generate_random_data(
        (32, 64)))

    # Inputs specified in shape (batch, timestep, size)
    inputs = input_data(input_tensor, name="input")
    lstm_layer0 = LSTM([wf0_tensor, wi0_tensor, wc0_tensor, wo0_tensor],
                       name="lstm0")
    lstm_layer1 = LSTM([wf1_tensor, wi1_tensor, wc1_tensor, wo1_tensor],
                       name="lstm1")
    outputs, state = lstm_layer0(inputs)
    outputs, state = lstm_layer1(outputs)
    return graph

if __name__ != "main":
  graph = create_lstm_model()
  graph.print_summary()
  graph.write_graph()
