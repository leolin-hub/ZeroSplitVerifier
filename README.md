# Robustness Verification of RNN with Abstraction Refinement

## Abstract

Certified local robustness verification for recurrent neural networks (RNNs) is challenging because approximation errors introduced by nonlinear relaxations can propagate through recurrent connections and accumulate over time. As a result, scalable linear bound propagation methods often become overly conservative and fail to certify inputs that are in fact robust, especially when many pre-activation intervals straddle zero. We propose an abstraction-refinement framework for RNN verification that partitions such intervals to remove the dominant relaxation error: on each refined branch, ReLU becomes exact and smooth activations such as tanh and Sigmoid admit substantially tighter linear envelopes. To control the combinatorial cost of splitting in long sequences, we introduce a SHAP-guided timestep selection strategy that ranks hidden states by their contribution to the verification objective and refines only the most critical timesteps in temporal order. Experiments on CIFAR-10 and MNIST stroke benchmarks demonstrate consistent improvements in verification success and robustness-margin tightness over abstraction-only baselines, while exposing clear runtime trade-offs between ReLU and tanh models.

---

## Overview

This repository implements an **abstraction-refinement (ZeroSplit)** verification framework for recurrent neural networks. Two network families are supported:

| Network | Activation | Splitting target | Optimal branching |
|---------|-----------|-----------------|-------------------|
| Vanilla RNN | ReLU / tanh | Hidden state pre-activation (single scalar) | Fixed at 0 |
| LSTM | Sigmoid / tanh | Gate pre-activation (one of i/f/g/o) | Pre-computed p* via LUT |

The key idea in both cases is **ZeroSplit**: identify neurons whose pre-activation interval crosses zero (worst case for linear relaxation), split the interval into a negative branch and a positive branch, certify each branch independently with tighter bounds, and return `verified` only when both branches succeed.

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

---

## Environment & Conventions

- Python environment: conda `torch-env`. Two usage patterns:
  - Activate first: `conda activate torch-env` (recommended for tmux multi-window parallelism)
  - Direct path: set `PYTHON_BIN` environment variable (see below)
- **Run all commands from the repo root** — `auto_test_*.py` scripts invoke verifiers via relative paths, so the working directory must be the repo root.

> `auto_test_*.py` supports two environment variable overrides:
> - `PYTHON_BIN`: path to the Python interpreter, defaults to `sys.executable` (current environment)
> - `MODEL_ROOT`: model root directory, defaults to `../models`
>
> Example: `MODEL_ROOT=/home/user/models PYTHON_BIN=/path/to/python python vanilla_rnn/auto_test_zs_mnist.py`

---

## Reproducibility & Experiment Artifacts

The trained models and the exact test samples used in the paper are **not tracked in git**:
models live in a `models/` directory that is a *sibling* of this repo (referenced as `../models/`
from `vanilla_rnn/` and `lstm/`), and `lstm/test_samples/` is git-ignored. They are distributed
as **GitHub Release** assets so experiments can be reproduced without retraining.

### 1. Download & placement

Download both assets from the [`v1.0-artifacts` release](https://github.com/leolin-hub/ZeroSplitVerifier/releases/tag/v1.0-artifacts):

| Asset | SHA-256 |
|-------|---------|
| [`models.zip`](https://github.com/leolin-hub/ZeroSplitVerifier/releases/download/v1.0-artifacts/models.zip) | `f9212f5a6d7273f2c73eec3f8394414ac1a163ab971345ee031bb8a6446409f6` |
| [`test_samples.zip`](https://github.com/leolin-hub/ZeroSplitVerifier/releases/download/v1.0-artifacts/test_samples.zip) | `8a6c1b3d298216c124909c4dc410dc1afa632d26910c8e12ad4c41e03c3b8863` |

Extract so that `models/` ends up as a *sibling* of the repo (`<repo>` = your clone's folder name):

```
<parent>/
├── <repo>/                       # this repo  ← extract test_samples.zip here
│   ├── lstm/test_samples/        #   relu-lstm .pt (also used by the GenBaB comparison)
│   └── test_samples/             #   vanilla-RNN .pt snapshots
└── models/                       # ← extract models.zip here (becomes ../models/)
```

```bash
# from the repo's parent directory
sha256sum -c <<'SUMS'
f9212f5a6d7273f2c73eec3f8394414ac1a163ab971345ee031bb8a6446409f6  models.zip
8a6c1b3d298216c124909c4dc410dc1afa632d26910c8e12ad4c41e03c3b8863  test_samples.zip
SUMS
unzip models.zip                  # → ./models/...
unzip test_samples.zip -d <repo>   # → <repo>/lstm/test_samples + <repo>/test_samples
```

### 2. Models in the Release (paper grid)

| Family | Dir / checkpoint | Grid |
|--------|------------------|------|
| Vanilla RNN — MNIST | `models/mnist_classifier/rnn_{ts}_{hs}_{act}/rnn` | ts {1,2,4,7} × hs {4,8,16,32} × act {relu,tanh} |
| Vanilla RNN — CIFAR-10 | `models/cifar10_classifier/rnn_{ts}_{hs}_{act}/rnn` | ts {8,12,24,32} × hs {16,32,64,128} × act {relu,tanh} |
| Vanilla RNN — MNIST stroke | `models/mnist_seq_classifier/rnn_seq_{ts}_{hs}_{act}/rnn` | ts {30,35,40,45} × hs {16,32,64,128} × act {relu,tanh} |
| ReLU-LSTM — MNIST | `models/mnist_relu_lstm/relu_lstm_{ts}_{hs}/relu_lstm` | ts {1,2,4,7} × hs {4,8,16,32} |

112 checkpoints total (32 + 32 + 32 + 16). The `mnist_relu_lstm` set is also required for the
GenBaB comparison in the Appendix. CIFAR-10 models are trained on **RGB** input
(`3072/ts`), so verification must pass `--use-rgb` (see Verification below).

### 3. Random seed & sampling parameters

All sampling is deterministic via a fixed **`seed = 2025`** set at import in every sampler —
[`vanilla_rnn/utils/sample_data.py`](vanilla_rnn/utils/sample_data.py#L10-L13),
[`sample_seq_mnist.py`](vanilla_rnn/utils/sample_seq_mnist.py#L11-L14),
[`sample_cifar10.py`](vanilla_rnn/utils/sample_cifar10.py#L6-L9),
[`sample_cifar10_lstm.py`](vanilla_rnn/utils/sample_cifar10_lstm.py#L6-L9).
Reproducibility relies on this seed **plus** `shuffle=True` producing a deterministic DataLoader
order; a fresh verifier process draws exactly `N` (default 50) samples. Passing `rnn=model`
makes the sampler report correctly-predicted counts, and the target class is
`target_label = (y + randint(1..9)) % 10`.

| Dataset / verifier | Sampler |
|--------------------|---------|
| RNN MNIST | `sample_mnist_data` |
| RNN MNIST stroke | `sample_seq_mnist_data` |
| RNN CIFAR-10 | `sample_cifar10_data` |
| ReLU-LSTM MNIST | `sample_mnist_data` |

### 4. Regenerating the artifacts

The Release bundle is produced by [`package_artifacts.py`](package_artifacts.py) (export `.pt`
snapshots → collect the paper-grid model dirs → write both zips under `dist/`):

```bash
python package_artifacts.py                 # export samples + collect models + zip → dist/
python package_artifacts.py --export-only    # only regenerate the .pt snapshots
```

Each `.pt` holds `{'X', 'labels', 'target_label'}` (relu-lstm also stores `X_flat` for VNN-LIB);
re-running produces byte-identical files (seed 2025). The relu-lstm snapshots match
[`lstm/save_test_samples.py`](lstm/save_test_samples.py) so POPQORN and GenBaB score identical points.

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

# MNIST LSTM (hidden=64, T=4)
python lstm/train_mnist_lstm.py \
    --hidden-size 64 --time-step 4
```

---

## Verification

### Vanilla RNN — EVR (Exact Verifiable Robustness) Mode

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

> `--work-dir` must point at a model directory extracted from `models.zip`
> (e.g. `../models/cifar10_classifier/rnn_8_64_relu/`). For CIFAR-10 also pass `--use-rgb`,
> since those models are trained on RGB input (`3072/ts`); see [Reproducibility](#reproducibility--experiment-artifacts).

### LSTM — EVR (Exact Verifiable Robustness) Mode

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
```

Additional LSTM arguments:

| Argument | Description |
|----------|-------------|
| `--lut-dir` | Directory with pre-built branching point LUTs |
| `--relu` | Use ReLU-LSTM (relu(yg), identity output) — required for GenBaB comparison |
| `--pq-only` | Run POPQORN bound only, skip ZeroSplit |
| `--gate-filter` | Restrict splits to specific gates, e.g. `g` or `g,i` |

### Toy RNN (Sanity Check)

```bash
python vanilla_rnn/rnn_zerosplit_verifier.py --toy-rnn --max-splits 2
```

---

## Batch Scanning (Auto Test)

Each experiment line consists of a core verifier plus an outer `auto_test` driver that sweeps over parameter grids. Every `auto_test_*.py` supports:
- `--timestep N`: run only this timestep (designed for tmux multi-window parallelism)
- `--resume`: skip already-completed runs from the latest `auto_test_results_*/progress.json`

### Vanilla RNN

| Dataset | Script | Timesteps | Hidden sizes |
|---------|--------|-----------|--------------|
| MNIST | `vanilla_rnn/auto_test_zs_mnist.py` | 1, 2, 4, 7 | 4, 8, 16, 32 |
| MNIST sequential | `vanilla_rnn/auto_test_seq.py` | 30, 35, 40, 45 | 16, 32, 64, 128 |
| CIFAR-10 | `vanilla_rnn/auto_test_cifar10.py` | 8, 12, 24, 32 | 16, 32, 64, 128 |

### LSTM (MNIST)

```bash
python lstm/auto_test_mnist_lstm.py --timestep {1,2,4,7}
```

hidden `[4, 8, 16, 32]` × timesteps `[1, 2, 4, 7]`, eps range 0.01–0.3

### tmux Four-Window Parallelism

```bash
conda activate torch-env
tmux new-session -d -s rnn -c /path/to/POPQORN     # window 0
tmux new-window  -t rnn                             # windows 1-3
tmux send-keys -t rnn:0 "python vanilla_rnn/auto_test_zs_mnist.py --timestep 1" C-m
tmux send-keys -t rnn:1 "python vanilla_rnn/auto_test_zs_mnist.py --timestep 2" C-m
tmux send-keys -t rnn:2 "python vanilla_rnn/auto_test_zs_mnist.py --timestep 4" C-m
tmux send-keys -t rnn:3 "python vanilla_rnn/auto_test_zs_mnist.py --timestep 7" C-m
tmux attach -t rnn
```

> ⚠️ Each window runs `--n-workers 4` internally; 4 windows × 4 workers = 16 processes. Adjust based on available CPU cores.

---

## Result Parsing

Each verifier run writes a JSON file under `--save-dir` containing `experiment_info`, `evr_summary`, `timing_stats`, and `sample_records`:
- RNN: `evr_results/session_rnn_{act}_hidden{hs}_ts{ts}_p{p}/evr_rnn_*.json`
- LSTM: `lstm/evr_results/session_lstm_hidden{hs}_ts{ts}_p{p}/evr_lstm_*.json`

**Export to Excel:**

```bash
python vanilla_rnn/parse_evr.py          # RNN (paths configured in __main__)
python lstm/parse_evr_lstm.py --input-dir ./lstm/evr_results --output evr_lstm_summary.xlsx
```

---

## Result Flags

| Flag | Meaning |
|------|---------|
| `pq_all_pass` | Baseline certified at all eps; ZeroSplit never ran |
| `zs_better` | Baseline failed at some eps, ZeroSplit succeeded — refinement helped |
| `both_fail` | Both baseline and ZeroSplit failed up to eps_max |

---

## Appendix: GenBaB (α,β-CROWN) Comparison

Comparison target: LSTM on MNIST, hidden `[4, 8, 16, 32]` × timestep `[1, 2, 4, 7]`, N=50, L2 norm, CPU only.
GenBaB lives in a separate repo (not tracked here); bridge scripts in `lstm/`:

| Script | Purpose |
|--------|---------|
| `lstm/save_test_samples.py` | Sample N test points with seed=2025 (shared across both methods) |
| `lstm/export_relu_lstm_onnx.py` | Export ONNX (FlatWrapper pre-splits gate weights — onnxsim not required) |
| `lstm/create_genbab_configs.py` | Generate GenBaB yaml configs |
| `lstm/auto_test_genbab_mnist.py` | Run α,β-CROWN sweep and collect wall-clock timing |
| `lstm/parse_genbab_results.py` | Parse results into `lstm/genbab_results.xlsx` |

**Execution order:**

```bash
python lstm/save_test_samples.py
python lstm/export_relu_lstm_onnx.py
python lstm/create_genbab_configs.py
python lstm/auto_test_genbab_mnist.py
python lstm/parse_genbab_results.py
```

### Minimal changes made to the GenBaB repo

Two files in `alpha-beta-CROWN/complete_verifier/` were modified:

| File | Change |
|------|--------|
| `abcrown.py` | Added two `print` lines around the BaB call — `[TIMING] crown: X.Xs` before and `[TIMING] bab: X.Xs` after — so `auto_test_genbab_mnist.py` can extract per-sample CROWN and BaB wall-clock times via regex. |
| `custom/custom_relu_lstm_data.py` | Custom data loader (reads `lstm/test_samples/samples_h{hs}_t{ts}.pt` produced by `save_test_samples.py`). Fixed `data_min`/`data_max` shape from `(1, 784)` to `(784,)` to resolve a BaB forward-pass shape mismatch (`mat1 (1,1) × mat2 (784,4)`). |

Benchmark artefacts generated by our scripts but placed inside the GenBaB repo:
- `benchmarks/mnist_relu_lstm/onnx/h{4,8,16,32}_t{1,2,4,7}.onnx` (16 files, produced by `export_relu_lstm_onnx.py`)
- `benchmarks/mnist_relu_lstm/configs/config_h*_t*.yaml` (16 files, produced by `create_genbab_configs.py`)

### Output columns

GenBaB certifies in two stages: CROWN (fast, loose) → BaB (slow, tight). Output columns:
- `genbab_crown_only`: samples certified by CROWN alone
- `genbab_bab_rescued`: samples where BaB succeeded after CROWN failed
- `genbab_n_unknown`: timeout / inconclusive
- `genbab_cert_pct`: overall certification rate (%)

> ⚠️ `avg_crown_ms` and `avg_bab_ms` are averaged over the CROWN-failed subset; `avg_total_ms` is averaged over all N samples — the three columns use different denominators and cannot be summed.
