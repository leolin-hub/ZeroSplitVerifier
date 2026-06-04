#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export trained ReLU-LSTM models to ONNX format for GenBaB verification.

Wraps ReLULSTMClassifier in a FlatWrapper that accepts a flat (1, 784) input
and reshapes it to (1, ts, 784//ts) internally, so VNN-LIB specs can declare
exactly 784 input variables matching raw MNIST pixels.

Output: /home/sausage/GenBaB/benchmarks/mnist_relu_lstm/onnx/h{hs}_t{ts}.onnx
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                              # lstm/
sys.path.insert(0, os.path.join(_HERE, '..'))          # POPQORN/

import argparse
import torch
import torch.nn as nn

from train_mnist_relu_lstm import ReLULSTMClassifier


HIDDEN_SIZES = [4, 8, 16, 32]
TIMESTEPS    = [1, 2, 4, 7]

MODEL_ROOT = '../models/mnist_relu_lstm'
ONNX_OUT   = '/home/sausage/GenBaB/benchmarks/mnist_relu_lstm/onnx'


class FlatWrapper(nn.Module):
    """Accepts flat (batch, 784) input and runs a ReLU-LSTM explicitly.

    Gate weights are pre-sliced into buffers so the ONNX graph has no Slice
    nodes on the weight/gate dimension.  This gives auto_LiRPA a clean linear
    graph that propagates tighter bounds without needing onnxsim.

    h and c are initialised via x_seq.new_zeros() so their shape is derived
    from the input tensor: onnxsim will NOT constant-fold them, preserving the
    dynamic batch dimension required by GenBaB's BaB domain management.
    """

    def __init__(self, inner: ReLULSTMClassifier, ts: int):
        super().__init__()
        hid    = inner.rnn.hidden_size
        W_ih   = inner.rnn.weight_ih_l0.data          # (4h, in)
        W_hh   = inner.rnn.weight_hh_l0.data          # (4h, h)
        b      = (inner.rnn.bias_ih_l0.data +
                  inner.rnn.bias_hh_l0.data)           # (4h,)

        # Pre-slice: each gate gets its own constant buffer → no Slice in ONNX
        for name, mat in (('Wix', W_ih[0*hid:1*hid]),
                          ('Wfx', W_ih[1*hid:2*hid]),
                          ('Wgx', W_ih[2*hid:3*hid]),
                          ('Wox', W_ih[3*hid:4*hid]),
                          ('Wia', W_hh[0*hid:1*hid]),
                          ('Wfa', W_hh[1*hid:2*hid]),
                          ('Wga', W_hh[2*hid:3*hid]),
                          ('Woa', W_hh[3*hid:4*hid])):
            self.register_buffer(name, mat.clone())
        for name, vec in (('bi', b[0*hid:1*hid]),
                          ('bf', b[1*hid:2*hid]),
                          ('bg', b[2*hid:3*hid]),
                          ('bo', b[3*hid:4*hid])):
            self.register_buffer(name, vec.clone())

        self.register_buffer('fc_w', inner.fc.weight.data.clone())
        self.register_buffer('fc_b', inner.fc.bias.data.clone())
        # h0/c0 as [1, hid] constant buffers — auto_LiRPA treats them as
        # non-perturbed constants, so A-matrix propagation works correctly.
        self.register_buffer('h0', torch.zeros(1, hid))
        self.register_buffer('c0', torch.zeros(1, hid))
        self.ts  = ts
        self.hid = hid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 784)
        # Slice on axis=1 of the flat input avoids Gather(axis=1),
        # which auto_LiRPA's BoundGather does not propagate A-matrices for.
        # BoundSlice with constant start/end IS supported.
        batch = x.shape[0]
        step  = 784 // self.ts

        h = self.h0.expand(batch, self.hid)
        c = self.c0.expand(batch, self.hid)

        for t in range(self.ts):
            xt = x[:, t * step:(t + 1) * step]                  # (batch, in)
            yi = xt @ self.Wix.t() + self.bi + h @ self.Wia.t()
            yf = xt @ self.Wfx.t() + self.bf + h @ self.Wfa.t()
            yg = xt @ self.Wgx.t() + self.bg + h @ self.Wga.t()
            yo = xt @ self.Wox.t() + self.bo + h @ self.Woa.t()

            c = torch.sigmoid(yf) * c + torch.sigmoid(yi) * torch.relu(yg)
            h = torch.sigmoid(yo) * c

        return h @ self.fc_w.t() + self.fc_b


def export_one(hidden_size: int, timestep: int, model_root: str, onnx_out: str) -> bool:
    input_size  = 784 // timestep
    output_size = 10

    model_path = os.path.join(
        model_root,
        f'relu_lstm_{timestep}_{hidden_size}',
        'relu_lstm'
    )
    if not os.path.exists(model_path):
        print(f'  [SKIP] model not found: {model_path}')
        return False

    inner = ReLULSTMClassifier(input_size, hidden_size, output_size, dropout=0.0)
    state = torch.load(model_path, map_location='cpu')
    inner.load_state_dict(state)
    inner.eval()

    model = FlatWrapper(inner, timestep)
    model.eval()

    dummy = torch.zeros(1, 784)
    out_name = f'h{hidden_size}_t{timestep}.onnx'
    out_path = os.path.join(onnx_out, out_name)

    # Gate weights are pre-sliced in FlatWrapper, so no Slice nodes in the graph.
    # onnxsim is not needed; export directly.
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print(f'  Exported → {out_path}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Export only this hidden size (default: all)')
    parser.add_argument('--time-step', type=int, default=None,
                        help='Export only this timestep (default: all)')
    parser.add_argument('--model-root', default=MODEL_ROOT)
    parser.add_argument('--onnx-out',   default=ONNX_OUT)
    args = parser.parse_args()

    hs_list = [args.hidden_size] if args.hidden_size else HIDDEN_SIZES
    ts_list = [args.time_step]   if args.time_step   else TIMESTEPS

    os.makedirs(args.onnx_out, exist_ok=True)
    ok = failed = skipped = 0

    for hs in hs_list:
        for ts in ts_list:
            print(f'Exporting h{hs}_t{ts} ...')
            result = export_one(hs, ts, args.model_root, args.onnx_out)
            if result is True:
                ok += 1
            elif result is False:
                skipped += 1

    print(f'\nDone: {ok} exported, {skipped} skipped.')


if __name__ == '__main__':
    main()
