#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-(hidden_size, timestep) YAML config files for GenBaB verification.

Epsilon is NOT embedded in the config — it's passed via --epsilon CLI override
at runtime (one run per epsilon value).

Output: /home/sausage/GenBaB/benchmarks/mnist_relu_lstm/configs/config_h{hs}_t{ts}.yaml
"""

import os
import argparse

HIDDEN_SIZES = [4, 8, 16, 32]
TIMESTEPS    = [1, 2, 4, 7]
ONNX_ROOT    = '/home/sausage/GenBaB/benchmarks/mnist_relu_lstm/onnx'
CONFIGS_OUT  = '/home/sausage/GenBaB/benchmarks/mnist_relu_lstm/configs'

CONFIG_TEMPLATE = """\
general:
  device: cpu

model:
  onnx_path: {onnx_path}

data:
  dataset: 'Customized("custom_relu_lstm_data", "load_relu_lstm_samples", hidden_size={hs}, timestep={ts})'
  std: 1.0
  mean: 0.0

specification:
  type: lp
  norm: 2
  epsilon: 0.01   # overridden per-run via --epsilon CLI argument

attack:
  pgd_order: skip   # no PGD attack for L2 norm

solver:
  batch_size: 64

bab:
  timeout: 3600
  max_iterations: 5
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--time-step',   type=int, default=None)
    parser.add_argument('--onnx-root',   default=ONNX_ROOT)
    parser.add_argument('--configs-out', default=CONFIGS_OUT)
    args = parser.parse_args()

    hs_list = [args.hidden_size] if args.hidden_size else HIDDEN_SIZES
    ts_list = [args.time_step]   if args.time_step   else TIMESTEPS

    os.makedirs(args.configs_out, exist_ok=True)

    for hs in hs_list:
        for ts in ts_list:
            onnx_path = os.path.join(args.onnx_root, f'h{hs}_t{ts}.onnx')
            content   = CONFIG_TEMPLATE.format(hs=hs, ts=ts, onnx_path=onnx_path)
            out_path  = os.path.join(args.configs_out, f'config_h{hs}_t{ts}.yaml')
            with open(out_path, 'w') as f:
                f.write(content)
            print(f'  Written → {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
