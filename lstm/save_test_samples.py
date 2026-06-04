#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save exactly the same 50 test samples that lstm_zerosplit_verifier.py uses,
so both POPQORN and GenBaB verify the same data points.

The verifier calls sample_mnist_data(N, seq_len, device, rnn=model, shuffle=True)
which filters for correctly-predicted samples using a fixed seed (seed=2025 in
vanilla_rnn/utils/sample_data.py). We replicate this call per (ts, hs) and
save the results.

Output: lstm/test_samples/samples_h{hs}_t{ts}.pt
  {'X':      FloatTensor (N, ts, 784//ts)   # normalized [-1,1]
   'labels': LongTensor  (N,)               # true labels
   'X_flat': FloatTensor (N, 784)           # flat, for VNN-LIB generation
  }
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))

import argparse
import torch
import numpy as np
import random

from train_mnist_relu_lstm import ReLULSTMClassifier
from vanilla_rnn.utils.sample_data import sample_mnist_data

HIDDEN_SIZES = [4, 8, 16, 32]
TIMESTEPS    = [1, 2, 4, 7]
N            = 50
MODEL_ROOT   = '../models/mnist_relu_lstm'
DATA_DIR     = './data'
OUT_DIR      = 'lstm/test_samples'


def reset_seed(seed=2025):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_for(hidden_size, timestep, model_root, data_dir, n, device):
    input_size  = 784 // timestep
    output_size = 10

    model_path = os.path.join(
        model_root,
        f'relu_lstm_{timestep}_{hidden_size}',
        'relu_lstm'
    )
    model = ReLULSTMClassifier(input_size, hidden_size, output_size, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval().to(device)

    reset_seed()
    X, labels, _ = sample_mnist_data(
        N=n, seq_len=timestep, device=device,
        data_dir=data_dir, train=False, shuffle=True, rnn=model
    )
    return X.cpu(), labels.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--time-step',   type=int, default=None)
    parser.add_argument('--N', type=int, default=N)
    parser.add_argument('--model-root',  default=MODEL_ROOT)
    parser.add_argument('--data-dir',    default=DATA_DIR)
    parser.add_argument('--out-dir',     default=OUT_DIR)
    args = parser.parse_args()

    hs_list = [args.hidden_size] if args.hidden_size else HIDDEN_SIZES
    ts_list = [args.time_step]   if args.time_step   else TIMESTEPS
    device  = torch.device('cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    for hs in hs_list:
        for ts in ts_list:
            print(f'Sampling h{hs}_t{ts} (N={args.N}) ...', end=' ', flush=True)
            X, labels = sample_for(hs, ts, args.model_root, args.data_dir, args.N, device)
            X_flat = X.view(X.shape[0], -1)   # (N, 784)

            out_path = os.path.join(args.out_dir, f'samples_h{hs}_t{ts}.pt')
            torch.save({'X': X, 'labels': labels, 'X_flat': X_flat}, out_path)
            print(f'saved → {out_path}  labels={labels[:5].tolist()}')

    print('Done.')


if __name__ == '__main__':
    main()
