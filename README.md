# Robustness Verification of RNN with Abstraction Refinement

## Abstract

Certified local robustness verification for recurrent neural networks (RNNs) is challenging because approximation errors introduced by nonlinear relaxations can propagate through recurrent connections and accumulate over time. As a result, scalable linear bound propagation methods often become overly conservative and fail to certify inputs that are in fact robust, especially when many pre-activation intervals straddle zero. We propose an abstraction-refinement framework for RNN verification that partitions such intervals to remove the dominant relaxation error: on each refined branch, ReLU becomes exact and smooth activations such as tanh and Sigmoid admit substantially tighter linear envelopes. To control the combinatorial cost of splitting in long sequences, we introduce a SHAP-guided timestep selection strategy that ranks hidden states by their contribution to the verification objective and refines only the most critical timesteps in temporal order. Experiments on CIFAR-10 and MNIST stroke benchmarks demonstrate consistent improvements in verification success and robustness-margin tightness over abstraction-only baselines, while exposing clear runtime trade-offs between ReLU and tanh models.

---

## Overview

This repository implements an **abstraction-refinement (ZeroSplit)** verification framework for recurrent neural networks, inspired by the CROWN-style linear bound propagation approach of [POPQORN (ICML 2019)](https://arxiv.org/abs/1905.07387). Two network families are supported:

| Network | Activation | Splitting target | Optimal branching |
|---------|-----------|-----------------|-------------------|
| Vanilla RNN | ReLU / tanh | Hidden state pre-activation (single scalar) | Fixed at 0 |
| LSTM | Sigmoid / tanh / ReLU-g | Gate pre-activation (one of i/f/g/o) | Pre-computed p* via LUT |

The key idea in both cases is **ZeroSplit**: identify neurons whose pre-activation interval crosses zero (worst case for linear relaxation), split the interval into a negative branch and a positive branch, certify each branch independently with tighter bounds, and return `verified` only when both branches succeed.

---

## Repository Structure

```
POPQORN/
├── vanilla_rnn/                   # Vanilla RNN verification
│   ├── bound_vanilla_rnn.py       # Core RNN bound propagation
│   ├── rnn_zerosplit_verifier.py  # ZeroSplit verifier (EVR loop)
│   ├── zerosplit_verifier.py      # ZeroSplit verifier (fixed-eps mode)
│   ├── zsv.py                     # Shared ZeroSplit helpers
│   ├── locate_timestep_shap.py    # SHAP-guided (timestep, neuron) ranking
│   ├── get_bound_for_general_activation_function.py  # CROWN linear bounds
│   ├── train_rnn_cifar10.py       # CIFAR-10 RNN trainer
│   ├── train_rnn_mnist_classifier.py
│   ├── train_rnn_mnist_seq.py
│   └── utils/                     # Data loaders (MNIST, seq-MNIST, CIFAR-10)
│
├── lstm/                          # LSTM verification
│   ├── lstm.py                    # My_lstm: core bound propagation
│   ├── lstm_relu.py               # My_relu_lstm: ReLU-g gate variant
│   ├── lstm_zerosplit_verifier.py # LSTMZeroSplitVerifier + ReLULSTMZeroSplitVerifier
│   ├── branching_point_optimizer.py  # Offline LUT builder for optimal p*
│   ├── locate_neuron_lstm.py      # SHAP-guided (t, gate, neuron) ranking
│   ├── bound_tanhx_sigmoidy.py    # 2D bounding planes for tanh(x)·σ(y)
│   ├── bound_x_sigmoidy.py        # 2D bounding planes for x·σ(y)
│   ├── train_cifar10_lstm.py
│   ├── train_mnist_lstm.py
│   └── train_mnist_relu_lstm.py
│
├── lookup_tables/                 # Pre-built branching point LUTs (pkl)
├── data/                          # Datasets
└── models/                        # Saved model checkpoints
```

---

## Method

### 1. Baseline: Linear Bound Propagation

For each timestep, pre-activation bounds are computed by back-substituting through the recurrent weight matrices and replacing each nonlinear activation with a linear upper/lower envelope (CROWN-style):

- **ReLU**: tangent / secant bounding
- **tanh / Sigmoid**: chord / tangent bounding depending on the sign of the pre-activation interval

Bounds propagate forward in time; at the output layer, certified robustness holds when the lower bound on the true-class logit exceeds all other classes' upper bounds.

### 2. ZeroSplit Refinement

When the baseline fails to certify a sample, ZeroSplit is invoked:

1. A target neuron `(t, n)` is selected via SHAP ranking (highest SHAP value in temporal order).
2. The pre-activation interval `[l, u]` at `(t, n)` is split into two branches:
   - **neg branch**: clamp `u ← min(u, p*)` → recompute all subsequent bounds
   - **pos branch**: clamp `l ← max(l, p*)` → recompute all subsequent bounds
3. Recurse on each branch up to `max_splits` depth.
4. Return `verified` iff both branches verify.
5. Restore the original interval and recompute on return (so the parent's state is correct).

For **vanilla RNN**, `p* = 0` always (split at zero).

For **LSTM**, the optimal split point `p*` is looked up from a pre-built table (see below).

### 3. SHAP-Guided Timestep Selection

Rather than exhaustive search, SHAP values identify which `(timestep, neuron)` pairs contribute most to the output margin. The ranking is computed once per sample and reused throughout the DFS tree. Timesteps are visited in **temporal order** (non-decreasing) to ensure each split's downstream recomputation is minimal.

### 4. Optimal Branching Point (LSTM only)

For LSTM gates with smooth activations (tanh, Sigmoid), splitting at zero is not optimal. The branching point optimizer pre-computes:

```
p*(l, u) = argmin_p  ∫[l→p] gap₁(x)dx  +  ∫[p→u] gap₂(x)dx
```

where `gap_i` is the area between the upper and lower linear envelopes on each sub-interval. This is computed offline for all `(l, u)` pairs on a grid and stored as a lookup table (pkl file).

Build LUTs once before running LSTM verification:

```bash
python lstm/branching_point_optimizer.py \
    --func all --output_dir ./lookup_tables
```

---

## Installation

```bash
conda activate torch-env
# Key dependencies
pip install torch torchvision shap loguru tqdm
```

Replace `python` with the full path to your environment's interpreter as needed.

---

## Training

### Vanilla RNN

```bash
# CIFAR-10 (ReLU, hidden=64, T=8)
python vanilla_rnn/train_rnn_cifar10.py \
    --hidden-size 64 --time-step 8 --activation relu

# MNIST stroke sequence (tanh, hidden=32, T=7)
python vanilla_rnn/train_rnn_mnist_classifier.py \
    --hidden-size 32 --time-step 7 --activation tanh
```

### LSTM

```bash
# CIFAR-10 LSTM (hidden=64, T=8)
python lstm/train_cifar10_lstm.py \
    --hidden-size 64 --time-step 8

# MNIST ReLU-LSTM (hidden=64, T=4)
python lstm/train_mnist_relu_lstm.py \
    --hidden-size 64 --time-step 4
```

---

## Verification

### Vanilla RNN — EVR (Epsilon-Value-Range) Mode

```bash
python vanilla_rnn/rnn_zerosplit_verifier.py \
    --hidden-size 64 --time-step 8 --activation relu \
    --dataset cifar10 \
    --work-dir <path/to/model/dir> \
    --N 50 --p 2 \
    --eps-min 0.005 --eps-max 0.015 \
    --max-splits 5 \
    --save-dir ./evr_results
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--activation` | `relu` or `tanh` |
| `--dataset` | `mnist`, `mnist-seq`, or `cifar10` |
| `--work-dir` | Directory containing the saved model checkpoint |
| `--eps-min/max` | Perturbation radius scan range |
| `--max-splits` | Maximum ZeroSplit recursion depth |
| `--n-workers` | Parallel workers (default: `cpu_count`) |
| `--save-dir` | Output directory for JSON results |

### LSTM — EVR Mode

```bash
python lstm/lstm_zerosplit_verifier.py \
    --hidden-size 64 --time-step 8 \
    --dataset cifar10 \
    --work-dir <path/to/model/dir> \
    --N 50 --p 2 \
    --eps-min 0.005 --eps-max 0.02 \
    --max-splits 5 \
    --lut-dir ./lookup_tables \
    --save-dir ./lstm/evr_results

# ReLU-LSTM variant
python lstm/lstm_zerosplit_verifier.py \
    --relu --hidden-size 64 --time-step 4 --dataset mnist \
    --max-splits 5 --lut-dir ./lookup_tables
```

Additional LSTM arguments:

| Argument | Description |
|----------|-------------|
| `--relu` | Use ReLU-g gate variant |
| `--lut-dir` | Directory with pre-built branching point LUTs |
| `--gate-filter` | Restrict splits to specific gates, e.g. `g` or `g,i` |

### Toy RNN (Sanity Check)

```bash
python vanilla_rnn/rnn_zerosplit_verifier.py --toy-rnn --max-splits 2
```

---

## Result Flags

| Flag | Meaning |
|------|---------|
| `pq_all_pass` | Baseline certified at all eps; ZeroSplit never ran |
| `zs_better` | Baseline failed at some eps, ZeroSplit succeeded — refinement helped |
| `both_fail` | Both baseline and ZeroSplit failed up to eps_max |

---

## Citation

```bibtex
@inproceedings{ko2019popqorn,
  title     = {POPQORN: Quantifying Robustness of Recurrent Neural Networks},
  author    = {Ko, Ching-Yun and Lyu, Zhiyuan and Weng, Lily and Daniel, Luca and Wong, Ngai and Lin, Dahua},
  booktitle = {ICML},
  year      = {2019}
}
```
