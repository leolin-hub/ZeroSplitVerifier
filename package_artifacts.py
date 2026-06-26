#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package reproducibility artifacts for a GitHub Release.

Exports seed-fixed (seed=2025) test-sample .pt snapshots and bundles the
paper-grid trained models into two zips:

  dist/popqorn_models.zip        → top-level models/...   (extract beside the repo → ../models/)
  dist/popqorn_test_samples.zip  → lstm/test_samples/*.pt + test_samples/<family>/*.pt
                                   (extract at the repo root)

Each .pt is drawn the same way a fresh verifier invocation draws it: reset_seed(2025)
then one shuffled, correctness-filtered sample (rnn=model). Only the paper grid is included.
"""

import os
import sys
import shutil
import zipfile
import argparse

import torch
import numpy as np
import random

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'vanilla_rnn'))
sys.path.insert(0, os.path.join(_ROOT, 'lstm'))

from utils.sample_data import sample_mnist_data
from utils.sample_seq_mnist import sample_seq_mnist_data
from utils.sample_cifar10 import sample_cifar10_data

SEED = 2025

# Paper experiment grid. acts=[None] marks a relu-lstm family (no activation suffix).
FAMILIES = [
    dict(name='mnist_classifier', dataset='mnist',
         ts=[1, 2, 4, 7], hs=[4, 8, 16, 32], acts=['relu', 'tanh'],
         input_size=lambda ts: 784 // ts),
    dict(name='cifar10_classifier', dataset='cifar10', use_rgb=True,
         ts=[8, 12, 24, 32], hs=[16, 32, 64, 128], acts=['relu', 'tanh'],
         input_size=lambda ts: 3072 // ts),
    dict(name='mnist_seq_classifier', dataset='mnist-seq',
         ts=[30, 35, 40, 45], hs=[16, 32, 64, 128], acts=['relu', 'tanh'],
         input_size=lambda ts: 3),
    dict(name='mnist_relu_lstm', dataset='relu-lstm',
         ts=[1, 2, 4, 7], hs=[4, 8, 16, 32], acts=[None],
         input_size=lambda ts: 784 // ts),
]

DATA_MNIST = os.path.join(_ROOT, 'data', 'mnist')
DATA_CIFAR = os.path.join(_ROOT, 'data')
DATA_SEQ   = os.path.join(_ROOT, 'data', 'mnist_seq', 'sequences') + os.sep


def reset_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def model_subdir(fam, ts, hs, act):
    if fam['name'] == 'mnist_classifier':
        return f'rnn_{ts}_{hs}_{act}', 'rnn'
    if fam['name'] == 'cifar10_classifier':
        return f'rnn_{ts}_{hs}_{act}', 'rnn'
    if fam['name'] == 'mnist_seq_classifier':
        return f'rnn_seq_{ts}_{hs}_{act}', 'rnn'
    if fam['name'] == 'mnist_relu_lstm':
        return f'relu_lstm_{ts}_{hs}', 'relu_lstm'
    raise ValueError(fam['name'])


def load_model(fam, ts, hs, act, ckpt_path, device):
    if fam['dataset'] == 'relu-lstm':
        from train_mnist_relu_lstm import ReLULSTMClassifier
        m = ReLULSTMClassifier(fam['input_size'](ts), hs, 10, dropout=0.0)
    else:
        from bound_vanilla_rnn import RNN
        m = RNN(fam['input_size'](ts), hs, 10, ts, act)
    m.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return m.eval().to(device)


def draw_samples(fam, model, ts, n, device):
    """reset_seed(2025) then one shuffled correctness-filtered draw — matches a fresh verifier run."""
    reset_seed()
    if fam['dataset'] == 'mnist' or fam['dataset'] == 'relu-lstm':
        x, y, t = sample_mnist_data(n, ts, device, data_dir=DATA_MNIST,
                                    train=False, shuffle=True, rnn=model)
    elif fam['dataset'] == 'cifar10':
        x, y, t = sample_cifar10_data(n, ts, device, data_dir=DATA_CIFAR, train=False,
                                      use_rgb=fam['use_rgb'], shuffle=True, rnn=model)
    elif fam['dataset'] == 'mnist-seq':
        x, y, t = sample_seq_mnist_data(n, ts, device, data_dir=DATA_SEQ,
                                        train=False, shuffle=True, rnn=model)
    else:
        raise ValueError(fam['dataset'])
    return x.cpu(), y.cpu(), t.cpu()


def export_samples(models_root, n, device):
    """Write .pt snapshots for every in-grid (ts,hs,act). Returns (lstm_dir, rnn_dir)."""
    lstm_dir = os.path.join(_ROOT, 'lstm', 'test_samples')
    rnn_dir  = os.path.join(_ROOT, 'test_samples')
    os.makedirs(lstm_dir, exist_ok=True)

    for fam in FAMILIES:
        for ts in fam['ts']:
            for hs in fam['hs']:
                for act in fam['acts']:
                    sub, ckpt = model_subdir(fam, ts, hs, act)
                    ckpt_path = os.path.join(models_root, fam['name'], sub, ckpt)
                    if not os.path.exists(ckpt_path):
                        print(f'  [skip] missing {ckpt_path}')
                        continue
                    model = load_model(fam, ts, hs, act, ckpt_path, device)
                    x, y, t = draw_samples(fam, model, ts, n, device)

                    if fam['dataset'] == 'relu-lstm':
                        out = os.path.join(lstm_dir, f'samples_h{hs}_t{ts}.pt')
                        torch.save({'X': x, 'labels': y, 'X_flat': x.view(x.shape[0], -1)}, out)
                    else:
                        fam_dir = os.path.join(rnn_dir, fam['name'])
                        os.makedirs(fam_dir, exist_ok=True)
                        out = os.path.join(fam_dir, f'samples_{act}_h{hs}_t{ts}.pt')
                        torch.save({'X': x, 'labels': y, 'target_label': t}, out)
                    print(f'  saved {os.path.relpath(out, _ROOT)}  labels={y[:5].tolist()}')
    return lstm_dir, rnn_dir


def collect_models(models_root, out_dir):
    """Copy in-grid model dirs into out_dir/models/<family>/<sub>/. Returns count copied."""
    dst_root = os.path.join(out_dir, 'models')
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    count = 0
    for fam in FAMILIES:
        for ts in fam['ts']:
            for hs in fam['hs']:
                for act in fam['acts']:
                    sub, _ = model_subdir(fam, ts, hs, act)
                    src = os.path.join(models_root, fam['name'], sub)
                    if not os.path.isdir(src):
                        print(f'  [skip] missing {src}')
                        continue
                    shutil.copytree(src, os.path.join(dst_root, fam['name'], sub))
                    count += 1
    return dst_root, count


def _zip_tree(zf, root, arc_prefix):
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            arc = os.path.join(arc_prefix, os.path.relpath(full, root))
            zf.write(full, arc)


def main():
    ap = argparse.ArgumentParser(description='Package reproducibility artifacts')
    ap.add_argument('--models-root', default=os.path.join(_ROOT, '..', 'models'))
    ap.add_argument('--out', default=os.path.join(_ROOT, 'dist'))
    ap.add_argument('--N', type=int, default=50)
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--zip-only', action='store_true')
    args = ap.parse_args()

    models_root = os.path.abspath(args.models_root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cpu')

    lstm_dir = os.path.join(_ROOT, 'lstm', 'test_samples')
    rnn_dir  = os.path.join(_ROOT, 'test_samples')

    if not args.zip_only:
        print('Exporting test samples ...')
        lstm_dir, rnn_dir = export_samples(models_root, args.N, device)

    if args.export_only:
        print('Done (export only).')
        return

    print('Collecting models ...')
    dst_models, n_models = collect_models(models_root, out_dir)
    print(f'  copied {n_models} model dirs')

    models_zip = os.path.join(out_dir, 'popqorn_models.zip')
    print(f'Writing {os.path.relpath(models_zip, _ROOT)} ...')
    with zipfile.ZipFile(models_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        _zip_tree(zf, dst_models, 'models')

    samples_zip = os.path.join(out_dir, 'popqorn_test_samples.zip')
    print(f'Writing {os.path.relpath(samples_zip, _ROOT)} ...')
    with zipfile.ZipFile(samples_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(lstm_dir):
            _zip_tree(zf, lstm_dir, os.path.join('lstm', 'test_samples'))
        if os.path.isdir(rnn_dir):
            _zip_tree(zf, rnn_dir, 'test_samples')

    print(f'Done. {n_models} models + samples → {os.path.relpath(out_dir, _ROOT)}/')


if __name__ == '__main__':
    main()
