#!/usr/bin/env python

import numpy as np

from smaug.core import types_pb2
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import data_op
from smaug.python.ops import recurrent

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_lstm_model():
  with Graph(name="bidi_lstm_ref", backend="Reference") as graph:
    input_tensor = Tensor(
        data_layout=types_pb2.NTC, tensor_data=generate_random_data((1, 4, 32)))
    # Tensors and kernels are initialized as NC layout.
    # Weights of forward LSTM.
    w_f = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((128, 32)))
    u_f = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((128, 32)))
    # Weights of backward LSTM.
    w_b = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((128, 32)))
    u_b = Tensor(
        data_layout=types_pb2.NC, tensor_data=generate_random_data((128, 32)))

    # Inputs specified in shape (batch, timestep, size)
    inputs = data_op.input_data(input_tensor, name="input")
    bidi_lstm = recurrent.BidirectionalLSTM([w_f, u_f], [w_b, u_b],
                                            name="bidi_lstm")
    outputs, state_fwd, state_bwd = bidi_lstm(inputs)
    return graph

if __name__ != "main":
  graph = create_lstm_model()
  graph.print_summary()
  graph.write_graph()
