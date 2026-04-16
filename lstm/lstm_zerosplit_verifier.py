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
import multiprocessing as mp
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from lstm import My_lstm
from vanilla_rnn.utils.sample_cifar10_lstm import sample_cifar10_data
from vanilla_rnn.utils.sample_data import sample_mnist_data


GATE_NAMES = ['i', 'f', 'g', 'o']


class Timer:
    """輕量 context manager，累積函數呼叫時間到 timing_stats dict。"""
    def __init__(self, stats: dict, key: str):
        self.stats = stats
        self.key = key

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t
        e = self.stats.setdefault(self.key, {'total_sec': 0.0, 'count': 0})
        e['total_sec'] += elapsed
        e['count'] += 1


def _run_evr_sample(v, sample_id, x_i, p,
                    eps_values, max_splits, eps_idx, background_size, top_k,
                    gate_filter=None):
    """
    EVR 驗證主體。由 worker 在建立 verifier v 後呼叫。
    對 LSTMZeroSplitVerifier 和 ReLULSTMZeroSplitVerifier 共用，避免程式碼重複。
    """
    import torch as _torch
    from locate_neuron_lstm import LSTMNeuronLocator

    with _torch.no_grad():
        top1_i = v.get_final_output(x_i).argmax(dim=1)

    v.sample_id = sample_id
    last_result = None
    sample_flag = 'pq_all_pass'
    sample_eps  = eps_values[-1] if eps_values else 0.0

    n_eps = len(eps_values)
    for eps_step, eps in enumerate(eps_values, 1):
        logger.info(f"[S{sample_id}] eps={eps:.6f} ({eps_step}/{n_eps})")
        v.compute_all_bounds(eps, p, x_i, eps_idx)
        pq_min, pq_max = v._get_output_bounds(eps, p, x_i, eps_idx)
        pq_verified = v._is_verified(pq_min, pq_max, top1_i)

        logger.info(f"[S{sample_id}] eps={eps:.6f} PQ={pq_verified}")

        if pq_verified:
            continue  # ZS not needed; scan next eps

        # pq failed: run SHAP + ZeroSplit once then stop
        locator = LSTMNeuronLocator(v, background_size=background_size,
                                    eps=eps, p=p, top_k=top_k,
                                    gate_filter=gate_filter)
        ranked = locator.compute_ranking(x_i, top1_i)

        v._pq_min = pq_min
        v._pq_max = pq_max
        v._leaf_margins = []
        _t0 = time.perf_counter()
        zs_verified = v._evr_recursive(
            eps, p, x_i, top1_i, eps_idx,
            depth=0, split_history=set(),
            max_splits=max_splits, locator=locator, ranked=ranked,
        )
        _e = v.timing_stats.setdefault('zs_total', {'total_sec': 0.0, 'count': 0})
        _e['total_sec'] += time.perf_counter() - _t0
        _e['count'] += 1

        L_val = int(top1_i[0].item())
        mask_val = _torch.arange(pq_max.shape[1]) != L_val
        pq_margin_val = (pq_min[0, L_val] - pq_max[0][mask_val].max()).item()
        zs_margin_val = min(v._leaf_margins) if v._leaf_margins else None
        gain_val = (zs_margin_val - pq_margin_val) if zs_margin_val is not None else None

        last_result = {
            'eps': eps,
            'pq_verified': pq_verified,
            'zs_verified': zs_verified,
            'pq_margin': round(pq_margin_val, 6),
            'zs_margin': round(zs_margin_val, 6) if zs_margin_val is not None else None,
            'gain': round(gain_val, 6) if gain_val is not None else None,
        }

        logger.info(
            f"[S{sample_id}] eps={eps:.6f} "
            f"PQ={pq_verified} ZS={zs_verified} "
            f"pq_margin={pq_margin_val:.6f} zs_margin={zs_margin_val}"
        )

        sample_eps  = eps
        if zs_verified:
            sample_flag = 'zs_better'
            break  # ZS rescued; stop scanning
        else:
            sample_flag = 'both_fail'
            # continue to next eps

    return sample_id, sample_flag, sample_eps, last_result, v.timing_stats


class LSTMZeroSplitVerifier(My_lstm):

    def __init__(self, net, device, WF=None, bF=None, seq_len=None,
                 a0=None, c0=None, max_splits=1, debug=False, lut_dir=None):
        My_lstm.__init__(self, net, device, WF, bF, seq_len, a0, c0)
        self.max_splits = max_splits
        self.debug = debug
        self.split_count = 0
        self.timing_stats = {}
        self._lut_dir = lut_dir
        self._luts = {}
        if lut_dir is not None:
            self._load_luts(lut_dir)

    # ------------------------------------------------------------------
    # LUT helpers: optimal branching point lookup
    # ------------------------------------------------------------------

    def _load_luts(self, lut_dir: str) -> None:
        """載入 pre-computed branching point lookup tables（pkl）。"""
        from branching_point_optimizer import load_lookup_table
        for act in ('relu', 'sigmoid', 'tanh'):
            path = os.path.join(lut_dir, f'{act}_lookup.pkl')
            if os.path.exists(path):
                self._luts[act] = load_lookup_table(path)
                logger.info(f'Loaded {act} LUT from {path}')

    def _gate_activation(self, gate: str) -> str:
        """gate name → activation function name（供 LUT 查表）。"""
        return 'tanh' if gate == 'g' else 'sigmoid'

    def _lookup_split_point(self, gate: str, l: float, u: float) -> float:
        """
        查 LUT 取最優分割點 p*。
        無 LUT 或 p* 不在 (l, u) 內時 fallback 到中點。
        """
        mid = (l + u) / 2.0
        if not self._luts:
            return mid
        lut = self._luts.get(self._gate_activation(gate))
        if lut is None:
            return mid
        from branching_point_optimizer import query_lookup_table_1d
        p_opt = query_lookup_table_1d(lut, l, u)
        return p_opt if (l < p_opt < u) else mid

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
            with Timer(self.timing_stats, 'get_y'):
                self.get_y(m=k, eps=eps, x=x, p=p, eps_idx=eps_idx)
            with Timer(self.timing_stats, 'get_hfc'):
                self.get_hfc(k)
            with Timer(self.timing_stats, 'get_hig'):
                self.get_hig(k)
            with Timer(self.timing_stats, 'get_c'):
                self.get_c(v=k, eps=eps, x=x, p=p, eps_idx=eps_idx)
            with Timer(self.timing_stats, 'get_hoc'):
                self.get_hoc(k)
            with Timer(self.timing_stats, 'get_Wa_b_hidden'):
                self.get_Wa_b(W_eye, 0, k, x, eps, p, save_a=True, eps_idx=eps_idx)

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

    def _apply_split(self, gate_l, gate_u, t, n, orig_l, orig_u, side,
                     eps, p, x, eps_idx, split_point=0.0):
        """
        將 gate 還原到 orig，套用單側 clamp，再從 timestep t 重算所有 bounds。
        side='neg' → gate_u[:,n] clamp ≤ split_point
        side='pos' → gate_l[:,n] clamp ≥ split_point
        split_point 預設 0.0（ZeroSplit 原語意）；有 LUT 時由呼叫方傳入 p*。
        """
        gate_l[t-1] = orig_l.clone()
        gate_u[t-1] = orig_u.clone()
        if side == 'neg':
            gate_u[t-1][:, n].clamp_(max=split_point)
        else:
            gate_l[t-1][:, n].clamp_(min=split_point)
        self._recompute_from(t, eps, p, x, eps_idx)

    def _restore_gate(self, gate_l, gate_u, t, orig_l, orig_u):
        """
        還原 gate 到 split 前的原始值（不含 _recompute_from）。
        neg 失敗時使用：parent 的 _recompute_from 會從自己的 t 往後重算，
        所以這裡只需確保 gate 值正確，不需要自行重算 bounds。
        """
        gate_l[t-1] = orig_l
        gate_u[t-1] = orig_u

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
        with Timer(self.timing_stats, 'get_Wa_b_output'):
            result = self.get_Wa_b(
                self.W, self.b, self.seq_len, x, eps, p, save_a=False, eps_idx=eps_idx
            )
        return result

    # ------------------------------------------------------------------
    # Step 3: EVR verification main loop
    # ------------------------------------------------------------------

    def _build_model_state(self):
        """序列化 My_lstm 權重為可 pickle 的 dict，供 worker 重建 verifier。"""
        h = self.hidden_size
        return {
            'input_size':   self.input_size,
            'hidden_size':  h,
            # 重拼回 nn.LSTM 格式（順序 i/f/g/o）
            'weight_ih_l0': torch.cat([self.Wix, self.Wfx, self.Wgx, self.Wox]).detach().cpu(),
            'weight_hh_l0': torch.cat([self.Wia, self.Wfa, self.Wga, self.Woa]).detach().cpu(),
            # My_lstm 已合併 bias_ih + bias_hh，還原時把 bias_hh 給 0
            'bias_ih_l0':   torch.cat([self.bi, self.bf, self.bg, self.bo]).detach().cpu(),
            'bias_hh_l0':   torch.zeros(4 * h),
            'WF':      self.W.detach().cpu(),
            'bF':      self.b.detach().cpu(),
            'seq_len': self.seq_len,
            'lut_dir': self._lut_dir,
        }

    @staticmethod
    def _process_single_sample_worker(args):
        """Worker：建立獨立 LSTMZeroSplitVerifier 對單一樣本執行完整 EVR 流程。"""
        (sample_id, x_i, model_state, p,
         eps_values, max_splits, eps_idx,
         background_size, top_k, gate_filter) = args

        import os, sys
        _dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_dir, '..'))
        sys.path.insert(0, _dir)

        import torch
        from lstm_zerosplit_verifier import LSTMZeroSplitVerifier, _run_evr_sample

        class _Net: pass
        net = _Net()
        net.input_size   = model_state['input_size']
        net.hidden_size  = model_state['hidden_size']
        net.weight_ih_l0 = model_state['weight_ih_l0']
        net.weight_hh_l0 = model_state['weight_hh_l0']
        net.bias_ih_l0   = model_state['bias_ih_l0']
        net.bias_hh_l0   = model_state['bias_hh_l0']

        v = LSTMZeroSplitVerifier(
            net, torch.device('cpu'),
            WF=model_state['WF'], bF=model_state['bF'],
            seq_len=model_state['seq_len'], max_splits=max_splits,
            lut_dir=model_state.get('lut_dir'),
        )
        return _run_evr_sample(v, sample_id, x_i, p,
                               eps_values, max_splits, eps_idx, background_size, top_k,
                               gate_filter=gate_filter)

    def verify_evr(self, X, p, eps_range, precision=0.001,
                   max_splits=None, eps_idx=None, background_size=20, top_k=5,
                   n_workers=None, save_dir=None, gate_filter=None):
        """
        對每個樣本逐一執行 eps_range 內的驗證流程：
          1. POPQORN：compute_all_bounds → pq_verified
          2. SHAP ranking：LSTMNeuronLocator.compute_ranking
          3. ZeroSplit：_evr_recursive → zs_verified

        若 pq_verified ≠ zs_verified 則對該樣本提早結束；所有 eps 相同則標記 'equal'。

        Args:
            X:              (N, seq_len, input_size)
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
        if n_workers is None:
            n_workers = mp.cpu_count()
        logger.info(f"n_workers set to {n_workers} for verify_evr.")
        if eps_idx is None:
            eps_idx = torch.ones(self.seq_len, device=X.device)

        low, high = eps_range
        eps_values = []
        current = low
        while current <= high + precision / 2:
            eps_values.append(round(current, 8))
            current += precision

        model_state = self._build_model_state()
        eps_idx_cpu = eps_idx.cpu()

        worker_args = [
            (i, X[i:i+1].cpu(), model_state, p,
             eps_values, max_splits, eps_idx_cpu, background_size, top_k, gate_filter)
            for i in range(X.shape[0])
        ]

        _worker = type(self)._process_single_sample_worker
        if n_workers > 1:
            with mp.Pool(processes=n_workers) as pool:
                raw = pool.map(_worker, worker_args)
        else:
            raw = [_worker(a) for a in worker_args]

        raw.sort(key=lambda x: x[0])

        # aggregate timing across all workers / samples
        agg_timing = {}
        sample_records = []
        for sample_id, flag, eps_r, result, t_stats in raw:
            logger.info(f'Sample {sample_id}: {flag}, eps={eps_r:.4f}, '
                        f'pq={result["pq_verified"] if result else None}, '
                        f'zs={result["zs_verified"] if result else None}')
            sample_records.append({
                'sample_id': sample_id, 'flag': flag, 'eps': eps_r,
                'pq_verified': result['pq_verified'] if result else None,
                'zs_verified': result['zs_verified'] if result else None,
                'pq_margin': result.get('pq_margin') if result else None,
                'zs_margin': result.get('zs_margin') if result else None,
                'gain':      result.get('gain')      if result else None,
            })
            for key, vals in t_stats.items():
                e = agg_timing.setdefault(key, {'total_sec': 0.0, 'count': 0})
                e['total_sec'] += vals['total_sec']
                e['count'] += vals['count']

        for key in agg_timing:
            cnt = agg_timing[key]['count']
            agg_timing[key]['avg_ms'] = round(
                agg_timing[key]['total_sec'] / cnt * 1000, 4) if cnt > 0 else 0.0

        if save_dir is not None:
            _write_evr_json(
                save_dir=save_dir,
                sample_records=sample_records,
                agg_timing=agg_timing,
                eps_range=eps_range,
                p=p,
                max_splits=max_splits if max_splits is not None else self.max_splits,
                hidden_size=self.hidden_size,
                seq_len=self.seq_len,
            )

        return [(flag, eps_r, result) for _, flag, eps_r, result, _ in raw]

    def _evr_recursive(self, eps, p, x, true_label, eps_idx,
                       depth, split_history, max_splits, locator, ranked):
        """
        遞迴 EVR 驗證。
        使用預先計算好的 ranked list 選出 split target（避免重複計算 SHAP）。
        若目前 bounds 已驗證則回傳 True；否則選出目標，分割後遞迴驗證兩子問題。
        """
        sid = getattr(self, 'sample_id', '?')
        minimum, maximum = self._get_output_bounds(eps, p, x, eps_idx)

        # Log margin comparison vs original PQ bounds
        cur_margin = None
        if hasattr(self, '_pq_min') and self._pq_min is not None:
            L = int(true_label[0].item())
            nc = minimum.shape[1]
            mask = torch.arange(nc, device=minimum.device) != L
            cur_margin = (minimum[0, L] - maximum[0, mask].max()).item()
            pq_margin  = (self._pq_min[0, L] - self._pq_max[0, mask].max()).item()
            logger.info(
                f"  [S{sid}] depth={depth} history={split_history} "
                f"margin={cur_margin:.6f}  PQ={pq_margin:.6f}  gain={cur_margin - pq_margin:.6f}"
            )

        # 子問題是否驗證成功
        if self._is_verified(minimum, maximum, true_label) and split_history:
            if cur_margin is not None and hasattr(self, '_leaf_margins'):
                self._leaf_margins.append(cur_margin)
            logger.info(f"  [S{sid}] Verified at depth {depth}.")
            return True

        # 上述仍未驗證成功進入到這裡，代表需要繼續split；但若已達 max_splits 則不再split，直接回傳 False。
        if depth >= max_splits:
            if cur_margin is not None and hasattr(self, '_leaf_margins'):
                self._leaf_margins.append(cur_margin)
            logger.info(f"  [S{sid}] Max splits ({max_splits}) reached, unverified.")
            return False

        t, gate, n = locator.select_next_split(ranked, split_history=split_history)
        if t is None:
            already = self._is_verified(minimum, maximum, true_label)
            if cur_margin is not None and hasattr(self, '_leaf_margins'):
                self._leaf_margins.append(cur_margin)
            logger.info(
                f"  [S{sid}] No valid split target found. "
                f"Falling back to current bounds: {'verified' if already else 'unverified'}."
            )
            return already

        new_history = split_history | {(t, gate, n)} # 選好後加入history
        gate_l = getattr(self, f'y{gate}_l')
        gate_u = getattr(self, f'y{gate}_u')
        orig_l = gate_l[t-1].clone()
        orig_u = gate_u[t-1].clone()

        # 查 LUT 取最優分割點（無 LUT 時 fallback 到 0.0）
        l_val = float(orig_l[0, n].item())
        u_val = float(orig_u[0, n].item())
        split_point = self._lookup_split_point(gate, l_val, u_val)

        # --- neg branch: gate ≤ split_point ---
        logger.info(f"  [S{sid}|eps={eps:.5f}] depth={depth} neg: t={t}, gate={gate}[{n}] u→clamp({split_point:.4f})")
        self._apply_split(gate_l, gate_u, t, n, orig_l, orig_u, side='neg',
                          eps=eps, p=p, x=x, eps_idx=eps_idx, split_point=split_point)
        neg_ok = self._evr_recursive(eps, p, x, true_label, eps_idx,
                                     depth + 1, new_history,
                                     max_splits, locator, ranked)

        if not neg_ok:
            # neg 子樹失敗 → 整體必為 False，跳過 pos 子樹（DFS 剪枝）。
            # 只還原 gate 值；parent 的 _recompute_from 會重算後續所有 bounds，
            # 所以這裡不需要額外呼叫 _recompute_from。
            self._restore_gate(gate_l, gate_u, t, orig_l, orig_u)
            logger.info(f"  [S{sid}|eps={eps:.5f}] depth={depth} neg failed → skip pos, propagate False")
            return False

        # --- pos branch: gate ≥ split_point ---
        logger.info(f"  [S{sid}|eps={eps:.5f}] depth={depth} pos: t={t}, gate={gate}[{n}] l→clamp({split_point:.4f})")
        self._apply_split(gate_l, gate_u, t, n, orig_l, orig_u, side='pos',
                          eps=eps, p=p, x=x, eps_idx=eps_idx, split_point=split_point)
        pos_ok = self._evr_recursive(eps, p, x, true_label, eps_idx,
                                     depth + 1, new_history,
                                     max_splits, locator, ranked)

        # --- restore original gate + recompute（此層的 split 已處理完畢）---
        self._restore_gate(gate_l, gate_u, t, orig_l, orig_u)
        self._recompute_from(t, eps, p, x, eps_idx)
        logger.info(f"  [S{sid}|eps={eps:.5f}] depth={depth} done: neg={neg_ok} pos={pos_ok} → {pos_ok}")

        return pos_ok  # neg_ok is True here


# ------------------------------------------------------------------
# ReLU-LSTM ZeroSplit Verifier
# ------------------------------------------------------------------

class ReLULSTMZeroSplitVerifier(LSTMZeroSplitVerifier):
    """
    ZeroSplit verifier for ReLU-LSTM:
        g gate : relu(yg)   instead of tanh(yg)
        output : σ(yo)*c    instead of σ(yo)*tanh(c)

    Inherits all bound-propagation and ZeroSplit logic from
    LSTMZeroSplitVerifier; overrides forward/get_hig/get_hoc via
    My_relu_lstm through __init__.
    """

    def __init__(self, net, device, WF=None, bF=None, seq_len=None,
                 a0=None, c0=None, max_splits=1, debug=False, lut_dir=None):
        from lstm_relu import My_relu_lstm
        # Initialise with My_relu_lstm's __init__ (reads weights from net)
        My_relu_lstm.__init__(self, net, device, WF, bF, seq_len, a0, c0)
        self.max_splits  = max_splits
        self.debug       = debug
        self.split_count = 0
        self.timing_stats = {}
        self._lut_dir = lut_dir
        self._luts = {}
        if lut_dir is not None:
            self._load_luts(lut_dir)

    def _gate_activation(self, gate: str) -> str:
        """ReLU-LSTM: g gate 用 relu，其餘 sigmoid。"""
        return 'relu' if gate == 'g' else 'sigmoid'

    # Override get_hig/get_hoc/forward come from My_relu_lstm via MRO
    # (My_relu_lstm is in the MRO because __init__ calls it directly).
    # We patch the class-level method lookup so Python's attribute resolution
    # finds My_relu_lstm's overrides.  The easiest way is explicit delegation.

    def get_hig(self, m):
        from lstm_relu import My_relu_lstm
        return My_relu_lstm.get_hig(self, m)

    def get_hoc(self, m):
        from lstm_relu import My_relu_lstm
        return My_relu_lstm.get_hoc(self, m)

    def forward(self, x, a0=None, c0=None, use_x_seq_len=False):
        from lstm_relu import My_relu_lstm
        return My_relu_lstm.forward(self, x, a0=a0, c0=c0,
                                    use_x_seq_len=use_x_seq_len)

    def _build_model_state(self):
        state = super()._build_model_state()
        state['relu'] = True
        return state

    @staticmethod
    def _process_single_sample_worker(args):
        """Worker：建立獨立 ReLULSTMZeroSplitVerifier 對單一樣本執行完整 EVR 流程。"""
        (sample_id, x_i, model_state, p,
         eps_values, max_splits, eps_idx,
         background_size, top_k, gate_filter) = args

        import os, sys
        _dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_dir, '..'))
        sys.path.insert(0, _dir)

        import torch
        from lstm_zerosplit_verifier import ReLULSTMZeroSplitVerifier, _run_evr_sample

        class _Net: pass
        net = _Net()
        net.input_size   = model_state['input_size']
        net.hidden_size  = model_state['hidden_size']
        net.weight_ih_l0 = model_state['weight_ih_l0']
        net.weight_hh_l0 = model_state['weight_hh_l0']
        net.bias_ih_l0   = model_state['bias_ih_l0']
        net.bias_hh_l0   = model_state['bias_hh_l0']

        v = ReLULSTMZeroSplitVerifier(
            net, torch.device('cpu'),
            WF=model_state['WF'], bF=model_state['bF'],
            seq_len=model_state['seq_len'], max_splits=max_splits,
            lut_dir=model_state.get('lut_dir'),
        )
        return _run_evr_sample(v, sample_id, x_i, p,
                               eps_values, max_splits, eps_idx, background_size, top_k,
                               gate_filter=gate_filter)


# ------------------------------------------------------------------
# JSON result writer
# ------------------------------------------------------------------

def _write_evr_json(save_dir, sample_records, agg_timing, eps_range,
                    p, max_splits, hidden_size, seq_len):
    """驗證結果寫入 JSON，供 parse_evr_lstm.py 解析。"""
    N = len(sample_records)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (Path(save_dir) /
                   f'session_lstm_hidden{hidden_size}_ts{seq_len}_p{p}')
    session_dir.mkdir(parents=True, exist_ok=True)
    fname = (f'evr_lstm_hidden{hidden_size}_ts{seq_len}'
             f'_eps{eps_range[0]}-{eps_range[1]}_p{p}'
             f'_N{N}_splits{max_splits}.json')
    path = session_dir / fname

    zs_better   = sum(1 for r in sample_records if r['flag'] == 'zs_better')
    both_fail   = sum(1 for r in sample_records if r['flag'] == 'both_fail')
    pq_all_pass = sum(1 for r in sample_records if r['flag'] == 'pq_all_pass')

    data = {
        'experiment_info': {
            'timestamp':   ts,
            'hidden_size': hidden_size,
            'time_step':   seq_len,
            'eps_min':     eps_range[0],
            'eps_max':     eps_range[1],
            'p_norm':      p,
            'max_splits':  max_splits,
            'N_samples':   len(sample_records),
        },
        'evr_summary': {
            'zs_better':   zs_better,
            'both_fail':   both_fail,
            'pq_all_pass': pq_all_pass,
        },
        'sample_records': sample_records,
        'timing_stats': {
            k: {
                'total_sec': round(v['total_sec'], 4),
                'count':     v['count'],
                'avg_ms':    v['avg_ms'],
            }
            for k, v in agg_timing.items()
        },
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f'EVR results saved → {path}')


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

DATASET_REGISTRY = {
    'cifar10': {
        'total_dim': lambda args: 3072 if args.use_rgb else 1024,
        'sample':    lambda args, time_step, device, model:
                         sample_cifar10_data(N=args.N, time_step=time_step, device=device,
                                             data_dir=args.data_dir, train=False,
                                             use_rgb=args.use_rgb, shuffle=True, rnn=model),
    },
    'mnist': {
        'total_dim': lambda args: 784,
        'sample':    lambda args, time_step, device, model:
                         sample_mnist_data(N=args.N, seq_len=time_step, device=device,
                                           data_dir=args.data_dir, train=False,
                                           shuffle=True, rnn=model),
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='LSTM ZeroSplit Verifier — EVR verification'
    )
    parser.add_argument('--hidden-size', default=64, type=int, metavar='HS',
                        help='hidden layer size (default: 32)')
    parser.add_argument('--time-step', default=4, type=int, metavar='TS',
                        help='sequence length / time steps (default: 8)')
    parser.add_argument('--use-rgb', action='store_true',
                        help='use RGB (3072) instead of grayscale (1024)')
    parser.add_argument('--work-dir',
                        default='../models/mnist_lstm/',
                        type=str, metavar='WD',
                        help='directory containing the pretrained model')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU if available')
    parser.add_argument('--N', default=50, type=int,
                        help='number of test samples (default: 5)')
    parser.add_argument('--p', default=2, type=int,
                        help='lp-norm; p > 100 is treated as inf (default: 2)')
    parser.add_argument('--eps', default=0.005, type=float,
                        help='perturbation radius (default: 0.005)')
    parser.add_argument('--eps-min', default=0.005, type=float,
                        help='eps range lower bound for verify_evr (default: 0.001)')
    parser.add_argument('--eps-max', default=0.02, type=float,
                        help='eps range upper bound for verify_evr (default: 0.01)')
    parser.add_argument('--max-splits', default=5, type=int,
                        help='max ZeroSplit iterations (default: 5)')
    parser.add_argument('--n-workers', default=None, type=int,
                        help='number of parallel worker processes (default: auto = cpu_count)')
    parser.add_argument('--save-dir', default='./lstm/evr_results', type=str,
                        help='directory to write EVR JSON results (default: None, no save)')
    parser.add_argument('--dataset', default='mnist', choices=list(DATASET_REGISTRY),
                        help='dataset: cifar10 | mnist (default: cifar10)')
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='data directory (default: ./data)')
    parser.add_argument('--relu', action='store_true',
                        help='use ReLU-LSTM verifier (relu(yg), identity output)')
    parser.add_argument('--gate-filter', default=None, type=str,
                        help='comma-separated gates to split on, e.g. "g" or "g,i" (default: all gates)')
    parser.add_argument('--lut-dir', default=None, type=str,
                        help='directory with pre-built branching point LUTs; '
                             'if omitted, ZeroSplit falls back to p=0 (original behaviour)')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    )
    p = args.p if args.p <= 100 else float('inf')
    time_step   = args.time_step
    hidden_size = args.hidden_size
    cfg        = DATASET_REGISTRY[args.dataset]
    total_dim  = cfg['total_dim'](args)
    input_size = total_dim // time_step
    output_size = 10

    if args.relu:
        from lstm_relu import My_relu_lstm
        from train_mnist_relu_lstm import ReLULSTMClassifier
        model = ReLULSTMClassifier(input_size, hidden_size, output_size, dropout=0.0)
        model_path = os.path.join(args.work_dir,
                                  f'relu_lstm_{time_step}_{hidden_size}', 'relu_lstm')
    else:
        model = LSTMClassifier(input_size, hidden_size, output_size)
        model_path = os.path.join(args.work_dir,
                                  f'lstm_{time_step}_{hidden_size}', 'lstm')

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    logger.info(f'Loaded model from {model_path}')

    X, _, _ = cfg['sample'](args, time_step, device, model)
    logger.info(f'Sampled {X.shape[0]} {args.dataset} samples (shape {list(X.shape)})')

    if args.relu:
        verifier = ReLULSTMZeroSplitVerifier(
            model.rnn, device,
            WF=model.fc.weight, bF=model.fc.bias,
            seq_len=time_step, max_splits=args.max_splits,
            lut_dir=args.lut_dir,
        )
    else:
        verifier = LSTMZeroSplitVerifier(
            model.rnn, device,
            WF=model.fc.weight, bF=model.fc.bias,
            seq_len=time_step, max_splits=args.max_splits,
            lut_dir=args.lut_dir,
        )

    eps_idx = torch.ones(time_step, device=device)

    gate_filter = set(args.gate_filter.split(',')) if args.gate_filter else None

    logger.info(f'\nRunning verify_evr (eps_range=({args.eps_min}, {args.eps_max}), p={p}, max_splits={args.max_splits}, gate_filter={gate_filter}) ...')
    results = verifier.verify_evr(
        X, p,
        eps_range=(args.eps_min, args.eps_max),
        max_splits=args.max_splits,
        eps_idx=eps_idx,
        n_workers=args.n_workers,
        save_dir=args.save_dir,
        gate_filter=gate_filter,
    )


if __name__ == '__main__':
    main()
