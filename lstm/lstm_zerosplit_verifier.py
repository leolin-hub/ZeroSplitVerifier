#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM ZeroSplit Verifier
Applies ZeroSplit refinement to LSTM certification, analogous to
vanilla_rnn/zerosplit_verifier.py but adapted for LSTM's 4-gate structure.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import torch
import torch.nn as nn
from loguru import logger
from lstm import My_lstm
from locate_neuron_lstm import LSTMNeuronLocator


GATE_NAMES = ['i', 'f', 'g', 'o']


class LSTMZeroSplitVerifier(My_lstm):

    def __init__(self, net, device, WF=None, bF=None, seq_len=None,
                 a0=None, c0=None, max_splits=1, debug=False):
        My_lstm.__init__(self, net, device, WF, bF, seq_len, a0, c0)
        self.max_splits = max_splits
        self.debug = debug
        self.split_count = 0

    # ------------------------------------------------------------------
    # Step 1: bound computation
    # 直接呼叫繼承自 My_lstm 的方法，按序計算所有 timestep 的 bounds。
    # 等同於 lstm.py get_k_th_layer_bound() 的逐層呼叫（無 print）。
    # ------------------------------------------------------------------

    def compute_all_bounds(self, eps, p, x, eps_idx=None):
        """
        計算所有 timestep 的 bounds。
        填入 yi_l/u, yf_l/u, yg_l/u, yo_l/u, c_l/u, a_l/u
        以及所有 alpha/beta/gamma 係數。

        對應 lstm.py get_last_layer_bound() 前段逐層呼叫 get_k_th_layer_bound()。
        """
        if eps_idx is None:
            eps_idx = torch.ones(self.seq_len, device=x.device)

        self.init_h()
        self.init_yc()

        W_eye = torch.eye(self.hidden_size, device=self.Wia.device)
        for k in range(1, self.seq_len + 1):
            self.get_y(m=k, eps=eps, x=x, p=p, eps_idx=eps_idx) # bound y_k
            self.get_hfc(k) # bound c_k-1 * sigmoid(yf_k)
            self.get_hig(k) # bound tanh(yg_k) * sigmoid(yi_k)
            self.get_c(v=k, eps=eps, x=x, p=p, eps_idx=eps_idx) # bound c_k
            self.get_hoc(k) # bound tanh(c_k) * sigmoid(yo_k)
            self.get_Wa_b(W_eye, 0, k, x, eps, p, save_a=True, eps_idx=eps_idx) # bound a_k

    # ------------------------------------------------------------------
    # Step 2: cross-zero detection
    # ------------------------------------------------------------------

    def detect_cross_zero(self, m):
        """
        偵測 timestep m 中，哪些 gate 的 pre-activation bounds 跨越 0。
        必須在 get_y(m) 之後呼叫。

        Args:
            m: timestep index (1-indexed)

        Returns:
            dict[str, BoolTensor(batch, hidden)]:
                key 為 gate name ('i','f','g','o')，
                value 為該 gate cross-zero 的 neuron mask。
        """
        assert self.yi_l[m-1] is not None, \
            f"Timestep {m} bounds not computed. Call get_y(m) first."

        return {
            'i': (self.yi_l[m-1] < 0) & (self.yi_u[m-1] > 0),
            'f': (self.yf_l[m-1] < 0) & (self.yf_u[m-1] > 0),
            'g': (self.yg_l[m-1] < 0) & (self.yg_u[m-1] > 0),
            'o': (self.yo_l[m-1] < 0) & (self.yo_u[m-1] > 0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recompute_from(self, m, eps, p, x, eps_idx):
        """
        從 timestep m 重新計算 bounds，假設 timestep 1..m-1 已正確。
        gate bounds (yi/yf/yg/yo) 在 m 處由呼叫者預先設定後再呼叫此方法。
        """
        W_eye = torch.eye(self.hidden_size, device=self.Wia.device)

        # timestep m：gate bounds 已由呼叫者設定，直接從 bounding planes 開始
        self.get_hfc(m)
        self.get_hig(m)
        self.get_c(v=m, eps=eps, x=x, p=p, eps_idx=eps_idx)
        self.get_hoc(m)
        self.get_Wa_b(W_eye, 0, m, x, eps, p, save_a=True, eps_idx=eps_idx)

        # timestep m+1..seq_len：完整重算
        for k in range(m + 1, self.seq_len + 1):
            self.get_y(m=k, eps=eps, x=x, p=p, eps_idx=eps_idx)
            self.get_hfc(k)
            self.get_hig(k)
            self.get_c(v=k, eps=eps, x=x, p=p, eps_idx=eps_idx)
            self.get_hoc(k)
            self.get_Wa_b(W_eye, 0, k, x, eps, p, save_a=True, eps_idx=eps_idx)

    def _is_verified(self, minimum, maximum, true_label):
        """
        對每個樣本 i（真實標籤 L），檢查是否：
            min_logit[i, L] > max_logit[i, c]  for all c ≠ L
        """
        for i in range(minimum.shape[0]):
            L = int(true_label[i].item())
            l_L = minimum[i, L]
            mask = torch.arange(maximum.shape[1], device=maximum.device) != L
            if not (l_L > maximum[i][mask]).all():
                return False
        return True

    def _get_output_bounds(self, eps, p, x, eps_idx):
        """呼叫 get_Wa_b 取得最終輸出 (minimum, maximum)。"""
        return self.get_Wa_b(
            self.W, self.b, self.seq_len, x, eps, p, save_a=False, eps_idx=eps_idx
        )

    # ------------------------------------------------------------------
    # Step 3: zero-split on a single (timestep, gate, neuron)
    # ------------------------------------------------------------------

    def split_at(self, m, gate, neuron_idx, eps, p, x, eps_idx=None):
        """
        在 timestep m 的 <gate> 的第 neuron_idx 個 neuron 的 pre-activation 上
        做 zero-split，計算 pos / neg 兩個子問題的最終輸出 bounds。

        neg: gate_preact[neuron_idx] 的 upper bound 夾至 0（pre-act ≤ 0）
        pos: gate_preact[neuron_idx] 的 lower bound 夾至 0（pre-act ≥ 0）

        Returns:
            (pos_bounds, neg_bounds): each = (minimum, maximum) output Tensors
        """
        if eps_idx is None:
            eps_idx = torch.ones(self.seq_len, device=x.device)

        gate_l = getattr(self, f'y{gate}_l')
        gate_u = getattr(self, f'y{gate}_u')
        orig_l = gate_l[m-1].clone()
        orig_u = gate_u[m-1].clone()

        # neg sub-problem: upper bound clamped to 0
        gate_l[m-1] = orig_l.clone()
        gate_u[m-1] = orig_u.clone()
        gate_u[m-1][:, neuron_idx].clamp_(max=0.0)
        self._recompute_from(m, eps, p, x, eps_idx)
        neg_min, neg_max = self._get_output_bounds(eps, p, x, eps_idx)

        # pos sub-problem: lower bound clamped to 0
        gate_l[m-1] = orig_l.clone()
        gate_u[m-1] = orig_u.clone()
        gate_l[m-1][:, neuron_idx].clamp_(min=0.0)
        self._recompute_from(m, eps, p, x, eps_idx)
        pos_min, pos_max = self._get_output_bounds(eps, p, x, eps_idx)

        # Restore original bounds
        gate_l[m-1] = orig_l
        gate_u[m-1] = orig_u
        self._recompute_from(m, eps, p, x, eps_idx)

        return (pos_min, pos_max), (neg_min, neg_max)

    # ------------------------------------------------------------------
    # Step 4: EVR verification main loop
    # ------------------------------------------------------------------

    def verify_evr(self, X, true_label, p, eps_range, precision=0.001,
                   max_splits=None, eps_idx=None, background_size=20, top_k=3):
        """
        對每個樣本逐一執行 eps_range 內的驗證流程：
          1. POPQORN：compute_all_bounds → pq_verified
          2. SHAP ranking：LSTMNeuronLocator.compute_ranking
          3. ZeroSplit：_evr_recursive → zs_verified

        若 pq_verified ≠ zs_verified 則對該樣本提早結束；所有 eps 相同則標記 'equal'。

        Args:
            X:              (N, seq_len, input_size)
            true_label:     (N,) ground-truth labels
            p:              lp-norm
            eps_range:      (eps_min, eps_max)
            precision:      eps step size
            max_splits:     override self.max_splits if provided
            eps_idx:        optional (seq_len,) mask for perturbed timesteps
            background_size: SHAP background sample count

        Returns:
            list of (flag, eps, result) per sample:
                flag:   'zs_better' | 'pq_better' | 'equal'
                eps:    the eps at which the decision was made
                result: dict with keys 'eps', 'pq_verified', 'zs_verified'
        """
        if max_splits is None:
            max_splits = self.max_splits
        if eps_idx is None:
            eps_idx = torch.ones(self.seq_len, device=X.device)

        low, high = eps_range
        eps_values = []
        current = low
        while current <= high + precision / 2:
            eps_values.append(round(current, 8))
            current += precision

        all_results = []

        for i in range(X.shape[0]):
            x_i = X[i:i+1]
            y_i = true_label[i:i+1]
            logger.info(f'Sample {i} (true={int(y_i.item())}):')

            last_result = None
            sample_flag = 'equal'
            sample_eps  = eps_values[-1] if eps_values else low

            for eps in eps_values:
                # Phase 1: POPQORN
                self.compute_all_bounds(eps, p, x_i, eps_idx)
                pq_min, pq_max = self._get_output_bounds(eps, p, x_i, eps_idx)
                pq_verified = self._is_verified(pq_min, pq_max, y_i)
                logger.info(f"Sample {i} eps={eps:.4f}, POPQORN verified: {pq_verified}")

                # Phase 2: SHAP ranking
                locator = LSTMNeuronLocator(self, background_size=background_size,
                                            eps=eps, p=p, top_k=top_k)
                ranked = locator.compute_ranking(x_i, y_i)

                # Phase 3: ZeroSplit
                zs_verified = self._evr_recursive(
                    eps, p, x_i, y_i, eps_idx,
                    depth=0, split_history=set(),
                    max_splits=max_splits, locator=locator, ranked=ranked
                )
                logger.info(f"Sample {i} eps={eps:.4f}, ZeroSplit verified: {zs_verified}")

                last_result = {
                    'eps': eps,
                    'pq_verified': pq_verified,
                    'zs_verified': zs_verified,
                }
                logger.info(f"Sample {i} eps={eps:.4f}, PQ={pq_verified}, ZS={zs_verified}")

                if not pq_verified and zs_verified:
                    logger.info(f"Sample {i} ZS better (PQ=F, ZS=T at eps={eps:.4f})")
                    sample_flag = 'zs_better'
                    sample_eps  = eps
                    break
                if pq_verified and not zs_verified:
                    logger.info(f"Sample {i} PQ better (PQ=T, ZS=F at eps={eps:.4f})")
                    sample_flag = 'pq_better'
                    sample_eps  = eps
                    break
            else:
                logger.info(f"Sample {i} Completed all eps values without disagreement, equal")

            all_results.append((sample_flag, sample_eps, last_result))

        return all_results

    def _evr_recursive(self, eps, p, x, true_label, eps_idx,
                       depth, split_history, max_splits, locator, ranked):
        """
        遞迴 EVR 驗證。
        使用預先計算好的 ranked list 選出 split target（避免重複計算 SHAP）。
        若目前 bounds 已驗證則回傳 True；否則選出目標，分割後遞迴驗證兩子問題。
        """
        minimum, maximum = self._get_output_bounds(eps, p, x, eps_idx)

        if self._is_verified(minimum, maximum, true_label) and split_history:
            logger.info(f"  Verified at depth {depth}.")
            return True

        if depth >= max_splits:
            logger.info(f"  Max splits ({max_splits}) reached, unverified.")
            return False

        t, gate, n = locator.select_next_split(ranked, split_history=split_history)
        if t is None:
            logger.info("  No valid split target found, unverified.")
            return False

        logger.info(f"  Splitting: t={t}, gate={gate}, n={n}, depth={depth}")
        new_history = split_history | {(t, gate, n)}

        gate_l = getattr(self, f'y{gate}_l')
        gate_u = getattr(self, f'y{gate}_u')
        orig_l = gate_l[t-1].clone()
        orig_u = gate_u[t-1].clone()

        _SPLIT_EPS = 1e-6

        # neg sub-problem
        gate_l[t-1] = orig_l.clone()
        gate_u[t-1] = orig_u.clone()
        gate_u[t-1][:, n].clamp_(max=-_SPLIT_EPS)
        self._recompute_from(t, eps, p, x, eps_idx)
        neg_ok = self._evr_recursive(eps, p, x, true_label, eps_idx,
                                     depth + 1, new_history,
                                     max_splits, locator, ranked)

        # Early exit: neg failed at max depth → whole branch is False
        if not neg_ok and depth + 1 >= max_splits:
            gate_l[t-1] = orig_l
            gate_u[t-1] = orig_u
            self._recompute_from(t, eps, p, x, eps_idx)
            logger.info(f"  Neg branch failed at max depth, returning False.")
            return False

        # pos sub-problem
        gate_l[t-1] = orig_l.clone()
        gate_u[t-1] = orig_u.clone()
        gate_l[t-1][:, n].clamp_(min=_SPLIT_EPS)
        self._recompute_from(t, eps, p, x, eps_idx)
        pos_ok = self._evr_recursive(eps, p, x, true_label, eps_idx,
                                     depth + 1, new_history,
                                     max_splits, locator, ranked)

        # Restore and propagate
        gate_l[t-1] = orig_l
        gate_u[t-1] = orig_u
        self._recompute_from(t, eps, p, x, eps_idx)

        return neg_ok and pos_ok


# ------------------------------------------------------------------
# Simple LSTM classifier (MNIST)
# ------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    單層 LSTM + 線性輸出，供 MNIST 測試使用。
    rnn 屬性為 nn.LSTM，可直接傳入 LSTMZeroSplitVerifier / My_lstm。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h, _) = self.rnn(x)
        return self.fc(h.squeeze(0))


# ------------------------------------------------------------------
# __main__: sample data → load weights → compute output bounds
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='LSTM ZeroSplit Verifier — EVR verification'
    )
    parser.add_argument('--hidden-size', default=8, type=int, metavar='HS',
                        help='hidden layer size (default: 32)')
    parser.add_argument('--time-step', default=4, type=int, metavar='TS',
                        help='sequence length / time steps (default: 8)')
    parser.add_argument('--use-rgb', default=True, type=bool,
                        help='use RGB (3072) or grayscale (1024) (default: True)')
    parser.add_argument('--work-dir',
                        default='../models/cifar10_lstm/',
                        type=str, metavar='WD',
                        help='directory containing the pretrained model')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU if available')
    parser.add_argument('--N', default=5, type=int,
                        help='number of test samples (default: 5)')
    parser.add_argument('--p', default=2, type=int,
                        help='lp-norm; p > 100 is treated as inf (default: 2)')
    parser.add_argument('--eps', default=0.005, type=float,
                        help='perturbation radius (default: 0.005)')
    parser.add_argument('--eps-min', default=0.005, type=float,
                        help='eps range lower bound for verify_evr (default: 0.001)')
    parser.add_argument('--eps-max', default=0.007, type=float,
                        help='eps range upper bound for verify_evr (default: 0.01)')
    parser.add_argument('--max-splits', default=3, type=int,
                        help='max ZeroSplit iterations (default: 3)')
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='CIFAR-10 data directory (default: ./data)')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    )
    p = args.p if args.p <= 100 else float('inf')
    time_step   = args.time_step
    hidden_size = args.hidden_size
    total_dim   = 3072 if args.use_rgb else 1024
    input_size  = total_dim // time_step
    output_size = 10

    model = LSTMClassifier(input_size, hidden_size, output_size)
    model_path = os.path.join(args.work_dir, f'lstm_{time_step}_{hidden_size}', 'lstm')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    logger.info(f'Loaded model from {model_path}')

    from vanilla_rnn.utils.sample_cifar10_lstm import sample_cifar10_data

    X, y, _ = sample_cifar10_data(
        N=args.N, time_step=time_step, device=device,
        data_dir=args.data_dir, train=False, use_rgb=args.use_rgb,
        shuffle=True, rnn=model
    )
    logger.info(f'Sampled {X.shape[0]} CIFAR-10 samples (shape {list(X.shape)})')
    logger.info(f'True labels: {y.tolist()}')

    verifier = LSTMZeroSplitVerifier(
        model.rnn, device,
        WF=model.fc.weight, bF=model.fc.bias,
        seq_len=time_step, max_splits=args.max_splits
    )

    eps_idx = torch.ones(time_step, device=device)

    logger.info(f'\nRunning verify_evr (eps_range=({args.eps_min}, {args.eps_max}), p={p}, max_splits={args.max_splits}) ...')
    results = verifier.verify_evr(
        X, y, p,
        eps_range=(args.eps_min, args.eps_max),
        max_splits=args.max_splits,
        eps_idx=eps_idx
    )

    logger.info('\nPer-sample results:')
    for i, (flag, eps_result, result) in enumerate(results):
        logger.info(
            f'  Sample {i}: {flag}, eps={eps_result:.3f}, '
            f'pq={result["pq_verified"]}, zs={result["zs_verified"]}'
        )


if __name__ == '__main__':
    main()
