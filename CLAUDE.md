# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

POPQORN is a certified robustness verification framework for Recurrent Neural Networks (RNNs, LSTMs, GRUs), published at ICML 2019 (paper: `1905.07387v1.pdf`). It computes provable lower bounds on the minimum adversarial perturbation needed to change an RNN classifier's output.

Current focus: SHAP-based timestep selection (`locate_timestep_shap.py`). PGD-based selection has been removed.

## Code Style

- 精簡為主，無 redundant code
- 無多餘 print
- 只寫必要邏輯

## Python Environment

Use the conda env executable directly:
```
C:/Users/zxczx/anaconda3/envs/torch-env/python.exe
```
Not `python` or `python3`.

## Dependencies

Key packages: PyTorch, NumPy, SHAP, Loguru, Torchvision. No requirements.txt exists — install manually via pip/conda.

## File Locations

- Models: `../models/` (relative to `vanilla_rnn/`)
- Training scripts: `vanilla_rnn/train_rnn_*.py`
- Verification: `vanilla_rnn/zerosplit_verifier.py`

## Common Commands

### Training
```bash
# Vanilla RNN on MNIST
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/train_rnn_mnist_classifier.py --hidden-size 32 --time-step 7 --activation tanh

# Vanilla RNN on CIFAR-10
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/train_rnn_cifar10.py --hidden-size 64 --time-step 8

# LSTM
C:/Users/zxczx/anaconda3/envs/torch-env/python.exelstm/train_model.py --hidden-size 32 --time-step 7 --activation tanh
```

### Verification
```bash
# ZeroSplit verifier (primary verification tool)
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/zerosplit_verifier.py \
  --hidden-size 64 --time-step 8 --activation relu \
  --work-dir <model_dir> --N 50 --p 2 --eps 0.5 \
  --max-splits 5 --mode shap --merge-results

# Basic bound checking
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/check_bounds.py
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/test_verifiers.py
```

### Automated Tests
```bash
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/auto_test_cifar10.py
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/auto_test_zs_mnist.py
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/auto_test_seq.py
```

### Result Parsing
```bash
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/parse_zsv.py   # ZeroSplit results
C:/Users/zxczx/anaconda3/envs/torch-env/python.exevanilla_rnn/parse_evr.py   # Epsilon-value-range results
```

## Architecture

### Two Sub-Packages

**`vanilla_rnn/`** — Verification for standard RNNs:
- `bound_vanilla_rnn.py`: Core `RNN` class with bound-propagation methods (`computePreactivationBounds`, `compute2sideBound`, `getConvenientGeneralActivationBound`)
- `get_bound_for_general_activation_function.py`: Linear bounding of activation functions (tanh, relu, sigmoid)
- `zerosplit_verifier.py` + `zsv.py`: ZeroSplit refinement algorithm (splits the verification problem at critical zero-crossing timesteps to tighten bounds)
- `locate_timestep_shap.py`: Uses SHAP values to rank (timestep, neuron) pairs for splitting priority
- `utils/`: Data loaders for MNIST, sequential MNIST, CIFAR-10, stock data

**`lstm/`** — Verification for LSTMs:
- `lstm.py`: LSTM with 2D bounding planes for cross-nonlinearities (sigma(v)·tanh(z) interactions between gates)
- `bound_tanhx_sigmoidy.py` / `bound_x_sigmoidy.py`: Specialized 2D bounding for LSTM gate products
- `get_bound_for_general_activation_function_lstm.py`: **Dead code** — filename has `_lstm` suffix so the import `from get_bound_for_general_activation_function import ...` never resolves to it; all LSTM code actually uses `vanilla_rnn/get_bound_for_general_activation_function.py`
- `BoundTanhSigmoidy/`: Gradient-descent optimization for finding tight bounding planes
- `locate_neuron_lstm.py`: SHAP-based (t, gate, neuron) importance ranking for ZeroSplit target selection
- `lstm_zerosplit_verifier.py`: `LSTMZeroSplitVerifier(My_lstm)` — wraps `My_lstm` with recursive ZeroSplit; takes `model.rnn` (nn.LSTM) + `WF/bF` (output layer)
- `train_cifar10_lstm.py`: Trains `LSTMClassifier(nn.LSTM + nn.Linear)`; saved state_dict compatible with `My_lstm.__init__`

### Core Verification Approach

1. **Bound Propagation**: Computes linear upper/lower bounds on each layer's pre-activation values, working backwards from the output to the input perturbation set.
2. **Activation Bounding**: Replaces nonlinear activations with linear functions (tangent lines or secant lines) that upper/lower bound the true activation.
3. **LSTM Cross-Nonlinearity**: Unlike vanilla RNNs (univariate bounds), LSTM gates require 2D bounding planes because the cell state involves products of two different nonlinear functions.
4. **ZeroSplit Refinement**: Identifies timesteps where the pre-activation interval crosses zero (worst case for linear bounds), splits the problem into sub-cases, and merges certified bounds.

### LSTM Bound Computation Flow (lstm.py My_lstm)

Per timestep k (called in `compute_all_bounds`):
```
get_y(k)         → yi/yf/yg/yo pre-activation bounds via Hölder: W·x₀ ± eps·‖W‖_q
get_hfc(k)       → fits 2D plane for  c_{k-1} · σ(yf_k)        → alpha/beta/gamma_fc
get_hig(k)       → fits 2D plane for  tanh(yg_k) · σ(yi_k)     → alpha/beta/gamma_ig
get_c(k)         → propagates planes to get c_k bounds
get_hoc(k)       → fits 2D plane for  tanh(c_k) · σ(yo_k)      → alpha/beta/gamma_oc
get_Wa_b(I,0,k)  → backward-fold all planes → a_k bounds (save_a=True)
```
Final: `get_Wa_b(W_out, b_out, seq_len)` → logit bounds

**`get_Wa_b(W, b, m)`**: computes `bound(W·a_m + b)` by backward-propagating stored α/β/γ coefficients through all timesteps, adding `eps·‖W_eff‖_q` per timestep via Hölder. With `W=I, b=0` yields hidden state bounds; with `W=W_out` yields logit bounds.

**`separate(W, α_l, α_u)`**: selects `α_u` where `W≥0`, `α_l` where `W<0` (interval arithmetic for maximizing/minimizing `W·α·x`).

### LSTM 2D Bounding Planes

`tanh(x)·σ(y)` and `x·σ(y)` are 2D nonlinear surfaces. Replaced by linear plane `α·x + β·y + γ`:
- **2D plane** (default): 500-step Adam on 3 corner z-values, maximize/minimize volume subject to `qualification_loss ≤ 0`
- **1D line** (`use_1D_line`): fix x at boundary, only fit 1D sigmoid bound analytically — faster, looser
- **Constant** (`use_constant`): min/max of 4 corners — fastest, loosest

**Orthant classification** (3×3=9 cases) in `bound_tanhx_sigmoidy.py`:
```
         y≥0            y≤0            y crosses 0
x≥0      1st            4th            one_four
x≤0      2nd            3rd            two_three
x cross  one_two        three_four     four (all 4 orthants)
```
Assertion `add_idx==1` ensures every neuron falls in exactly one case.

### ZeroSplit: Fixes & Logic

#### Bug 1 — shape mismatch in `tanh_sigmoid.py` (已修)
`sigmoid_lower` / `tanh_lower` 的 `zero` branch（`alpha==0` 或 `beta==0`）：
```python
# 修前（shape 不符 → RuntimeError）
loss[zero] = torch.relu(k*y_minus + b) + torch.relu(k*y_plus + b)
# 修後
loss[zero] = torch.relu(k[zero]*y_minus[zero] + b[zero]) + torch.relu(k[zero]*y_plus[zero] + b[zero])
```
`loss[zero]` shape 為 `[zero.sum()]`，RHS 必須同樣 index。修完後 ZeroSplit 可直接 clamp 到 `±0`，不再需要 `_SPLIT_EPS`。

#### Bug 2 — cross-zero neuron 為空時錯誤回傳失敗 (已修)
`select_next_split` 找不到可切 neuron（`t is None`）時，原本直接 return False。
但若 POPQORN 本身已驗證成功，應回傳成功。**Fix**：`t is None` 時 fallback 到 `_is_verified(current bounds)`。

#### Bug 3 — depth=0 已驗證仍不切分 (已修，保留至少切一次語意)
`_is_verified and split_history` 在 depth=0（`split_history=set()`）恆為 False，導致 POPQORN 已成功時仍繼續。
**設計意圖**：depth=0 強制至少切一次（若有可切 neuron），確認 ZeroSplit 結果與 POPQORN 一致。`t is None` fallback 處理無 neuron 可切的情形。

#### `_evr_recursive` 遞迴邏輯
- **Split clamp**：neg → `gate_u[t-1][:,n].clamp_(max=0)`；pos → `gate_l[t-1][:,n].clamp_(min=0)`
- **Restore**：每層在自己的 split target 上 `gate_l/u[t-1] = orig_l/u`，再 `_recompute_from(t)`；`_recompute_from` 跳過 `get_y(t)`（caller 已設定），只從 `get_hfc/hig/c/hoc/Wa_b(t)` 開始，t+1 以後完整重算
- **DFS 剪枝**：`if not neg_ok: restore; return False`（跳過整個 pos 子樹）。neg 子樹若任一路徑失敗則整體為 False，無需探索 pos
- **ranked list 固定**：`compute_ranking` 在 worker 只算一次，遞迴全程共用同一份（按 timestep 升序），保證切分 timestep 單調不遞減

### Model Compatibility

`My_lstm.__init__(net)` reads directly from `nn.LSTM` attributes:
```
net.weight_ih_l0  (4h×input)  → Wix/Wfx/Wgx/Wox  (gate order: i,f,g,o)
net.weight_hh_l0  (4h×hidden) → Wia/Wfa/Wga/Woa
net.bias_ih_l0 + net.bias_hh_l0 → bi/bf/bg/bo
WF, bF passed separately (output layer)
```
Constraints: single-layer, no bidirectional, output is linear `W·a_T + b`.

### Test Results

Timestamped output directories `auto_test_results_YYYYMMDD_HHMMSS/` contain `progress.json` and per-session JSON/TXT result files. `shap_timing_*.xlsx` files contain performance benchmarks by architecture.
