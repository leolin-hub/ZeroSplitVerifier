# Robustness Verification of RNN with Abstraction Refinement

## Abstract

Certified local robustness verification for recurrent neural networks (RNNs) is challenging because approximation errors introduced by nonlinear relaxations can propagate through recurrent connections and accumulate over time. As a result, scalable linear bound propagation methods often become overly conservative and fail to certify inputs that are in fact robust, especially when many pre-activation intervals straddle zero. We propose an abstraction-refinement framework for RNN verification that partitions such intervals to remove the dominant relaxation error: on each refined branch, ReLU becomes exact and smooth activations such as tanh and Sigmoid admit substantially tighter linear envelopes. To control the combinatorial cost of splitting in long sequences, we introduce a SHAP-guided timestep selection strategy that ranks hidden states by their contribution to the verification objective and refines only the most critical timesteps in temporal order. Experiments on CIFAR-10 and MNIST stroke benchmarks demonstrate consistent improvements in verification success and robustness-margin tightness over abstraction-only baselines, while exposing clear runtime trade-offs between ReLU and tanh models.

---

## Overview

This repository implements an **abstraction-refinement (ZeroSplit)** verification framework for recurrent neural networks. Two network families are supported:

| Network | Activation | Splitting target | Optimal branching |
|---------|-----------|-----------------|-------------------|
| Vanilla RNN | ReLU / tanh | Hidden state pre-activation (single scalar) | Fixed at 0 |
| LSTM *(Ongoing Work)* | Sigmoid / tanh | Gate pre-activation (one of i/f/g/o) | Pre-computed p* via LUT |

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

### 4. Optimal Branching Point (LSTM only) *(Ongoing Work)*

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

## 環境與執行慣例

- Python 環境：conda `torch-env`。兩種用法擇一：
  - 先啟動：`conda activate torch-env`（tmux 多 window 並行建議採此）
  - 直接指定：`$PYTHON_BIN` 環境變數（見下）
- **所有指令從 repo 根目錄執行**（auto_test 腳本以相對路徑呼叫 verifier，cwd 須是 repo 根）

> `auto_test_*.py` 支援兩個環境變數覆寫：
> - `PYTHON_BIN`：指定 Python 直譯器路徑，預設 `sys.executable`（當前環境）
> - `MODEL_ROOT`：模型根目錄，預設 `../models`
>
> 範例：`MODEL_ROOT=/home/user/models PYTHON_BIN=/path/to/python python vanilla_rnn/auto_test_zs_mnist.py`

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

## 批次掃描（Auto Test）

每條實驗線都是「核心 verifier + 外層 auto_test 掃描器」。每個 `auto_test_*.py` 支援：
- `--timestep N`：只跑這個 timestep（tmux 多 window 並行設計）
- `--resume`：從最新 `auto_test_results_*/progress.json` 跳過已完成項

### Vanilla RNN

| 資料集 | 腳本 | timesteps | hidden |
|--------|------|-----------|--------|
| MNIST | `vanilla_rnn/auto_test_zs_mnist.py` | 1,2,4,7 | 4,8,16,32 |
| MNIST sequential | `vanilla_rnn/auto_test_seq.py` | 50 | 16,32,64,128 |
| CIFAR-10 | `vanilla_rnn/auto_test_cifar10.py` | 8 | 16,32,64,128 |

### LSTM（MNIST）

```bash
python lstm/auto_test_mnist_lstm.py --timestep {1,2,4,7}
```

hidden `[4,8,16,32]` × timesteps `[1,2,4,7]`，eps 0.01–0.3

### tmux 四 window 並行

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

> 4 window × `--n-workers 4` = 16 個 process，請依 CPU 核心數調整。

---

## 結果解析

每個 verifier 跑完在 `--save-dir` 下寫 JSON（含 `experiment_info`、`evr_summary`、`timing_stats`、`sample_records`）：
- RNN：`evr_results/session_rnn_{act}_hidden{hs}_ts{ts}_p{p}/evr_rnn_*.json`
- LSTM：`lstm/evr_results/session_lstm_hidden{hs}_ts{ts}_p{p}/evr_lstm_*.json`

**解析成 Excel**：

```bash
python vanilla_rnn/parse_evr.py          # RNN（路徑寫在 __main__）
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

## 附錄：GenBaB（α,β-CROWN）比較

比較對象：LSTM on MNIST，hidden `[4,8,16,32]` × timestep `[1,2,4,7]`，N=50，L2 norm，CPU only。
GenBaB 安裝在獨立 repo（非本 repo 範圍）；橋接腳本在 `lstm/`：

| 腳本 | 用途 |
|------|------|
| `lstm/save_test_samples.py` | 以 seed=2025 抽樣，存共用樣本（確保兩邊測同一批）|
| `lstm/export_relu_lstm_onnx.py` | 匯出 ONNX（FlatWrapper 預切 gate 權重，不需 onnxsim）|
| `lstm/create_genbab_configs.py` | 產生 GenBaB yaml config |
| `lstm/auto_test_genbab_mnist.py` | 呼叫 α,β-CROWN 掃描，收集 wall-clock 計時 |
| `lstm/parse_genbab_results.py` | 解析結果成 `lstm/genbab_results.xlsx` |

**執行順序**：

```bash
python lstm/save_test_samples.py
python lstm/export_relu_lstm_onnx.py
python lstm/create_genbab_configs.py
python lstm/auto_test_genbab_mnist.py
python lstm/parse_genbab_results.py
```

GenBaB 認證分兩階段：CROWN（快但鬆）→ BaB（慢但緊）。結果欄位：
- `genbab_crown_only`：CROWN 單獨成功的樣本數
- `genbab_bab_rescued`：BaB 救回的樣本數（CROWN 失敗後 BaB 成功）
- `genbab_n_unknown`：timeout / 未知
- `genbab_cert_pct`：總認證率%

> 計時：`avg_crown_ms`、`avg_bab_ms` 分母是「CROWN 失敗子集」；`avg_total_ms` 分母是全體 N，三者不能直接相加。

