#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN ZeroSplit Verifier — EVR refactored version.

Differences from zerosplit_verifier.py:
  - Inherits bound_vanilla_rnn.RNN for weight loading/extraction.
  - compute2sideBound / computeLast2sideBound are overridden to NOT clear
    self.l[k]/self.u[k] intermediate values (required for ZeroSplit in-place restore).
  - EVR DFS follows lstm_zerosplit_verifier._evr_recursive pattern:
      in-place modify → recurse → restore (+recompute on success path).
  - No refine_preh snapshots, no _early_stop global flag.
  - EVR-only interface (no fixed-eps mode).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import types
import argparse
import multiprocessing as mp
import time
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

from loguru import logger

import get_bound_for_general_activation_function as get_bound
from bound_vanilla_rnn import RNN
from locate_timestep_shap import compute_shap_ranking_once, select_timestep_from_shap
from utils.sample_data import sample_mnist_data
from utils.sample_seq_mnist import sample_seq_mnist_data
from utils.sample_cifar10 import sample_cifar10_data


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class Timer:
    """Accumulate wall-clock time under a key in a stats dict."""
    def __init__(self, stats: dict, key: str):
        self.stats = stats
        self.key = key

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        e = self.stats.setdefault(self.key, {'total_sec': 0.0, 'count': 0})
        e['total_sec'] += time.perf_counter() - self._t
        e['count'] += 1


class _TimedGetConvBound:
    """
    Context manager: monkey-patches get_bound.getConvenientGeneralActivationBound
    to record its cumulative wall-clock time into timing_stats.
    Safe to nest (restores original on exit).
    """
    def __init__(self, stats: dict):
        self.stats = stats
        self._orig = None

    def __enter__(self):
        orig = get_bound.getConvenientGeneralActivationBound
        self._orig = orig
        stats = self.stats
        key = 'getConvenientGeneralActivationBound'

        def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = orig(*args, **kwargs)
            e = stats.setdefault(key, {'total_sec': 0.0, 'count': 0})
            e['total_sec'] += time.perf_counter() - t0
            e['count'] += 1
            return result

        get_bound.getConvenientGeneralActivationBound = _timed
        return self

    def __exit__(self, *_):
        get_bound.getConvenientGeneralActivationBound = self._orig


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class RNNZeroSplitVerifier(RNN):
    """
    ZeroSplit verifier for vanilla RNN (tanh / relu).

    Inherits from bound_vanilla_rnn.RNN.
    Overrides compute2sideBound + computeLast2sideBound — same maths as the
    originals but without clearing self.l[k]/self.u[k] intermediate values,
    which is required for the ZeroSplit in-place modify-and-restore pattern.

    EVR DFS (see _evr_recursive) is analogous to lstm_zerosplit_verifier.
    """

    def __init__(self, input_size, hidden_size, output_size, time_step,
                 activation, max_splits=1, debug=False):
        RNN.__init__(self, input_size, hidden_size, output_size,
                     time_step, activation)
        self.max_splits  = max_splits
        self.debug       = debug
        self.timing_stats: dict = {}

    # ------------------------------------------------------------------
    # compute2sideBound — no clearing of self.l/u
    # ------------------------------------------------------------------

    def compute2sideBound(self, eps, p, v, X=None, Eps_idx=None):
        """
        Pre-activation bounds for hidden layer at timestep v.

        Sets self.l[v], self.u[v].
        For k = 1..v-1 also fills self.kl[k], self.ku[k], self.bl[k], self.bu[k]
        (activation-function linear-bound slopes, reused by later timesteps and by
        computeLast2sideBound).

        Mathematically identical to bound_vanilla_rnn.RNN.compute2sideBound;
        the ONLY difference is that self.l[k]/self.u[k] are NOT cleared after use,
        so they remain available for ZeroSplit restore operations.
        """
        with torch.no_grad():
            n = self.W_ax.shape[1]   # input_size
            s = self.W_ax.shape[0]   # hidden_size
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx - 1] = 1
            N = X.shape[0]
            a_0 = (torch.zeros(N, s, device=X.device)
                   if self.a_0 is None else self.a_0)
            if isinstance(eps, torch.Tensor):
                eps = eps.to(X.device)
            if p == 1:
                q = float('inf')
            elif p in ('inf', float('inf')):
                q = 1
            else:
                q = p / (p - 1)

            W_ax = self.W_ax.unsqueeze(0).expand(N, -1, -1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N, -1, -1)  # [N, s, s]
            b_ax = self.b_ax.unsqueeze(0).expand(N, -1)      # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N, -1)      # [N, s]

            # --- v-th timestep contribution ---
            if isinstance(eps, torch.Tensor):
                eps_col = eps.unsqueeze(1).expand(-1, s)
                yU = idx_eps[v-1] * eps_col * torch.norm(W_ax, p=q, dim=2)
            else:
                yU = idx_eps[v-1] * eps * torch.norm(W_ax, p=q, dim=2)
            yL = -yU.clone()

            X_v = X.view(N, 1, n) if v == 1 else X
            wx = torch.matmul(W_ax, X_v[:, v-1, :].view(N, n, 1)).squeeze(2)
            yU = yU + wx + b_aa + b_ax
            yL = yL + wx + b_aa + b_ax

            # --- k = v-1 down to 1 ---
            if v > 1:
                for k in range(v - 1, 0, -1):
                    if k == v - 1:
                        kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                            self.l[k], self.u[k], self.activation)
                        bl = bl / (kl + 1e-8)
                        bu = bu / (ku + 1e-8)
                        self.kl[k] = kl
                        self.ku[k] = ku
                        self.bl[k] = bl
                        self.bu[k] = bu
                        alpha_l = kl.unsqueeze(1).expand(-1, s, -1)
                        alpha_u = ku.unsqueeze(1).expand(-1, s, -1)
                        beta_l  = bl.unsqueeze(1).expand(-1, s, -1)
                        beta_u  = bu.unsqueeze(1).expand(-1, s, -1)
                        I      = (W_aa >= 0).float()
                        lamida = I * alpha_u + (1 - I) * alpha_l
                        omiga  = I * alpha_l + (1 - I) * alpha_u
                        Delta  = I * beta_u  + (1 - I) * beta_l
                        Theta  = I * beta_l  + (1 - I) * beta_u
                        A  = W_aa * lamida
                        Ou = W_aa * omiga
                    else:
                        alpha_l = self.kl[k].unsqueeze(1).expand(-1, s, -1)
                        alpha_u = self.ku[k].unsqueeze(1).expand(-1, s, -1)
                        beta_l  = self.bl[k].unsqueeze(1).expand(-1, s, -1)
                        beta_u  = self.bu[k].unsqueeze(1).expand(-1, s, -1)
                        AW   = torch.matmul(A,  W_aa)
                        OuW  = torch.matmul(Ou, W_aa)
                        I      = (AW  >= 0).float()
                        lamida = I * alpha_u + (1 - I) * alpha_l
                        Delta  = I * beta_u  + (1 - I) * beta_l
                        I      = (OuW >= 0).float()
                        omiga  = I * alpha_l + (1 - I) * alpha_u
                        Theta  = I * beta_l  + (1 - I) * beta_u
                        A  = AW  * lamida
                        Ou = OuW * omiga

                    AW_ax  = torch.matmul(A,  W_ax)
                    OuW_ax = torch.matmul(Ou, W_ax)
                    if isinstance(eps, torch.Tensor):
                        eps_col = eps.unsqueeze(1).expand(-1, s)
                        yU = yU + idx_eps[k-1] * eps_col * torch.norm(AW_ax,  p=q, dim=2)
                        yL = yL - idx_eps[k-1] * eps_col * torch.norm(OuW_ax, p=q, dim=2)
                    else:
                        yU = yU + idx_eps[k-1] * eps * torch.norm(AW_ax,  p=q, dim=2)
                        yL = yL - idx_eps[k-1] * eps * torch.norm(OuW_ax, p=q, dim=2)

                    xk = X[:, k-1, :].view(N, n, 1)
                    yU = yU + torch.matmul(A,  torch.matmul(W_ax, xk)).squeeze(2)
                    yL = yL + torch.matmul(Ou, torch.matmul(W_ax, xk)).squeeze(2)
                    b  = (b_aa + b_ax).view(N, s, 1)
                    yU = yU + torch.matmul(A,  b).squeeze(2) + (A  * Delta).sum(2)
                    yL = yL + torch.matmul(Ou, b).squeeze(2) + (Ou * Theta).sum(2)

                A  = torch.matmul(A,  W_aa)
                Ou = torch.matmul(Ou, W_aa)
            else:
                A  = W_aa
                Ou = W_aa

            yU = yU + torch.matmul(A,  a_0.view(N, s, 1)).squeeze(2)
            yL = yL + torch.matmul(Ou, a_0.view(N, s, 1)).squeeze(2)

            self.l[v] = yL
            self.u[v] = yU
            return yL, yU

    # ------------------------------------------------------------------
    # computeLast2sideBound — uses stored kl/ku, no clearing
    # ------------------------------------------------------------------

    def computeLast2sideBound(self, eps, p, v, X=None, Eps_idx=None):
        """
        Output logit bounds.

        For k = v-1 (= time_step): computes activation slopes fresh from
        self.l[k]/self.u[k] (same as bound_vanilla_rnn original).
        For k < v-1: uses stored self.kl[k]/self.ku[k] from compute2sideBound.
        Does NOT clear self.l/u.
        """
        with torch.no_grad():
            n = self.W_ax.shape[1]
            s = self.W_ax.shape[0]
            t = self.W_fa.shape[0]   # output_size
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx - 1] = 1
            N = X.shape[0]
            a_0 = (torch.zeros(N, s, device=X.device)
                   if self.a_0 is None else self.a_0)
            if isinstance(eps, torch.Tensor):
                eps = eps.to(X.device)
            if p == 1:
                q = float('inf')
            elif p in ('inf', float('inf')):
                q = 1
            else:
                q = p / (p - 1)

            W_ax = self.W_ax.unsqueeze(0).expand(N, -1, -1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N, -1, -1)  # [N, s, s]
            W_fa = self.W_fa.unsqueeze(0).expand(N, -1, -1)  # [N, t, s]
            b_ax = self.b_ax.unsqueeze(0).expand(N, -1)
            b_aa = self.b_aa.unsqueeze(0).expand(N, -1)
            b_f  = self.b_f.unsqueeze(0).expand(N, -1)

            yU = torch.zeros(N, t, device=X.device)
            yL = torch.zeros(N, t, device=X.device)

            for k in range(v - 1, 0, -1):
                if k == v - 1:
                    # Fresh slope computation from current self.l[k]/self.u[k]
                    kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                        self.l[k], self.u[k], self.activation)
                    bl = bl / (kl + 1e-8)
                    bu = bu / (ku + 1e-8)
                    self.kl[k] = kl
                    self.ku[k] = ku
                    self.bl[k] = bl
                    self.bu[k] = bu
                    alpha_l = kl.unsqueeze(1).expand(-1, t, -1)
                    alpha_u = ku.unsqueeze(1).expand(-1, t, -1)
                    beta_l  = bl.unsqueeze(1).expand(-1, t, -1)
                    beta_u  = bu.unsqueeze(1).expand(-1, t, -1)
                    I      = (W_fa >= 0).float()
                    lamida = I * alpha_u + (1 - I) * alpha_l
                    omiga  = I * alpha_l + (1 - I) * alpha_u
                    Delta  = I * beta_u  + (1 - I) * beta_l
                    Theta  = I * beta_l  + (1 - I) * beta_u
                    A  = W_fa * lamida
                    Ou = W_fa * omiga
                else:
                    # Reuse stored slopes
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, t, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, t, -1)
                    beta_l  = self.bl[k].unsqueeze(1).expand(-1, t, -1)
                    beta_u  = self.bu[k].unsqueeze(1).expand(-1, t, -1)
                    AW   = torch.matmul(A,  W_aa)
                    OuW  = torch.matmul(Ou, W_aa)
                    I      = (AW  >= 0).float()
                    lamida = I * alpha_u + (1 - I) * alpha_l
                    Delta  = I * beta_u  + (1 - I) * beta_l
                    I      = (OuW >= 0).float()
                    omiga  = I * alpha_l + (1 - I) * alpha_u
                    Theta  = I * beta_l  + (1 - I) * beta_u
                    A  = AW  * lamida
                    Ou = OuW * omiga

                AW_ax  = torch.matmul(A,  W_ax)
                OuW_ax = torch.matmul(Ou, W_ax)
                if isinstance(eps, torch.Tensor):
                    eps_col = eps.unsqueeze(1).expand(-1, t)
                    yU = yU + idx_eps[k-1] * eps_col * torch.norm(AW_ax,  p=q, dim=2)
                    yL = yL - idx_eps[k-1] * eps_col * torch.norm(OuW_ax, p=q, dim=2)
                else:
                    yU = yU + idx_eps[k-1] * eps * torch.norm(AW_ax,  p=q, dim=2)
                    yL = yL - idx_eps[k-1] * eps * torch.norm(OuW_ax, p=q, dim=2)

                xk = X[:, k-1, :].view(N, n, 1)
                yU = yU + torch.matmul(A,  torch.matmul(W_ax, xk)).squeeze(2)
                yL = yL + torch.matmul(Ou, torch.matmul(W_ax, xk)).squeeze(2)
                b  = (b_aa + b_ax).view(N, s, 1)
                yU = yU + torch.matmul(A,  b).squeeze(2) + (A  * Delta).sum(2)
                yL = yL + torch.matmul(Ou, b).squeeze(2) + (Ou * Theta).sum(2)

            A  = torch.matmul(A,  W_aa)
            Ou = torch.matmul(Ou, W_aa)
            yU = yU + torch.matmul(A,  a_0.view(N, s, 1)).squeeze(2) + b_f
            yL = yL + torch.matmul(Ou, a_0.view(N, s, 1)).squeeze(2) + b_f
            return yL, yU

    # ------------------------------------------------------------------
    # Bound wrappers used by ZeroSplit
    # ------------------------------------------------------------------

    def compute_all_bounds(self, eps, p, X, Eps_idx):
        """Forward pass: fill self.l[1..T], self.u[1..T], self.kl/ku/bl/bu."""
        self.clear_intermediate_variables()
        for k in range(1, self.time_step + 1):
            self.compute2sideBound(eps, p, k, X=X[:, 0:k, :], Eps_idx=Eps_idx)

    def _recompute_from(self, t, eps, p, X, Eps_idx):
        """
        Re-run compute2sideBound for timesteps t+1..T.
        self.l[t]/self.u[t] must already be set (and possibly clamped) by the caller.
        Updates self.l[t+1..T], self.kl[t..T-1].
        """
        for k in range(t + 1, self.time_step + 1):
            self.compute2sideBound(eps, p, k, X=X[:, 0:k, :], Eps_idx=Eps_idx)

    def _get_output_bounds(self, eps, p, X, Eps_idx):
        return self.computeLast2sideBound(
            eps, p, v=self.time_step + 1, X=X, Eps_idx=Eps_idx)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _is_verified(self, yL_out, yU_out, true_label):
        L = int(true_label[0].item()) if isinstance(true_label, torch.Tensor) else int(true_label)
        mask = torch.arange(yU_out.shape[1], device=yU_out.device) != L
        return bool((yL_out[0, L] > yU_out[0, mask]).all().item())

    def _margin(self, yL_out, yU_out, true_label):
        """min logit[L] - max logit[c≠L]. Positive = verified."""
        L = int(true_label[0].item()) if isinstance(true_label, torch.Tensor) else int(true_label)
        mask = torch.arange(yU_out.shape[1], device=yU_out.device) != L
        return (yL_out[0, L] - yU_out[0, mask].max()).item()

    # ------------------------------------------------------------------
    # LSTM-style split helpers
    # ------------------------------------------------------------------

    def _apply_split(self, t, n, orig_l, orig_u, side, eps, p, X, Eps_idx):
        """
        Restore self.l[t]/self.u[t] to orig, clamp neuron n on the chosen side,
        then recompute from t+1.

        side='neg' → u[t][:,n] clamp ≤ 0
        side='pos' → l[t][:,n] clamp ≥ 0
        """
        self.l[t] = orig_l.clone()
        self.u[t] = orig_u.clone()
        if side == 'neg':
            self.u[t][:, n].clamp_(max=0.0)
        else:
            self.l[t][:, n].clamp_(min=0.0)
        self._recompute_from(t, eps, p, X, Eps_idx)

    def _restore(self, t, orig_l, orig_u):
        """
        Restore self.l[t]/self.u[t] without recomputing.
        Used after neg-branch failure (parent's recompute will propagate).
        """
        self.l[t] = orig_l
        self.u[t] = orig_u

    # ------------------------------------------------------------------
    # EVR DFS
    # ------------------------------------------------------------------

    def _evr_recursive(self, eps, p, X, true_label, Eps_idx,
                       depth, split_history, max_splits, ranked, sample_id,
                       start_timestep=1):
        """
        LSTM-style ZeroSplit DFS.

        At every node:
          1. Get output bounds → check verified.
          2. If at a leaf (depth ≥ max_splits or no target): record margin, return.
          3. Select next (t, n) from SHAP ranked list (t ≥ start_timestep).
          4. neg branch: clamp u[t][:,n] ≤ 0, recompute, recurse.
             If neg fails → restore (no recompute), propagate False (DFS prune).
          5. pos branch: clamp l[t][:,n] ≥ 0, recompute, recurse.
          6. Restore + recompute for parent, return pos_ok.

        start_timestep is passed down monotonically (set to current t in children)
        to match the original zerosplit_verifier.py's unsafe_layer behaviour.
        """
        yL_out, yU_out = self._get_output_bounds(eps, p, X, Eps_idx)
        verified = self._is_verified(yL_out, yU_out, true_label)
        margin   = self._margin(yL_out, yU_out, true_label)

        logger.info(
            f"  [S{sample_id}] depth={depth} history={sorted(split_history)} "
            f"margin={margin:.6f} verified={verified}"
        )

        if verified and split_history:
            self._leaf_margins.append(margin)
            logger.info(f"  [S{sample_id}] depth={depth} → VERIFIED (leaf)")
            return True

        if depth >= max_splits:
            self._leaf_margins.append(margin)
            logger.info(f"  [S{sample_id}] depth={depth} → max_splits reached, unverified")
            return False

        t, n, _ = select_timestep_from_shap(
            self, ranked,
            start_timestep=start_timestep, refine_preh=None,
            split_history=split_history, sample_id=sample_id,
        )

        if t is None:
            self._leaf_margins.append(margin)
            status = 'verified' if verified else 'unverified'
            logger.info(
                f"  [S{sample_id}] depth={depth} → no split target, "
                f"fallback={status}"
            )
            return verified

        orig_l = self.l[t].clone()
        orig_u = self.u[t].clone()
        new_history = split_history | {(t, n)}

        # --- neg branch: u[t][:,n] ← clamp ≤ 0 ---
        logger.info(f"  [S{sample_id}|eps={eps:.5f}] depth={depth} neg: t={t} n={n}")
        self._apply_split(t, n, orig_l, orig_u, 'neg', eps, p, X, Eps_idx)
        neg_ok = self._evr_recursive(
            eps, p, X, true_label, Eps_idx,
            depth + 1, new_history, max_splits, ranked, sample_id,
            start_timestep=t,
        )

        if not neg_ok:
            # DFS prune: neg failed → whole subtree unverified. Restore gate only;
            # parent's _recompute_from will rebuild state when needed.
            self._restore(t, orig_l, orig_u)
            logger.info(
                f"  [S{sample_id}|eps={eps:.5f}] depth={depth} "
                f"neg FAILED → skip pos, return False"
            )
            return False

        # --- pos branch: l[t][:,n] ← clamp ≥ 0 ---
        logger.info(f"  [S{sample_id}|eps={eps:.5f}] depth={depth} pos: t={t} n={n}")
        self._apply_split(t, n, orig_l, orig_u, 'pos', eps, p, X, Eps_idx)
        pos_ok = self._evr_recursive(
            eps, p, X, true_label, Eps_idx,
            depth + 1, new_history, max_splits, ranked, sample_id,
            start_timestep=t,
        )

        # Restore + recompute so parent sees correct state for self.l[t..T]
        self._restore(t, orig_l, orig_u)
        self._recompute_from(t, eps, p, X, Eps_idx)
        logger.info(
            f"  [S{sample_id}|eps={eps:.5f}] depth={depth} "
            f"done: neg=True pos={pos_ok}"
        )
        return pos_ok

    # ------------------------------------------------------------------
    # EVR per-sample loop
    # ------------------------------------------------------------------

    def _run_evr_sample(self, sample_id, x_i, top1_i, p, eps_values, max_splits):
        """
        For one sample: scan eps_values low→high.

        Phase 1 (POPQORN) runs at every eps.
        If pq_verified → skip to next eps (no SHAP, no ZS).
        If pq fails → Phase 2 (SHAP) + Phase 3 (ZS):
          ZS succeeds → flag = 'zs_better', stop.
          ZS fails    → continue to next eps.
        Reaching eps_max with ZS never succeeding → flag = 'both_fail'.

        Flags:
          'pq_all_pass' — pq succeeded at all eps; ZS never ran.
          'zs_better'   — ZS rescued at the first eps where pq=F and ZS=T.
          'both_fail'   — pq failed at some eps and ZS never succeeded up to eps_max.

        Timing: 'popqorn' and 'zs_total' counts increment together only for
          eps steps where ZS ran (pq=F), so their denominators are equal.
        """
        Eps_idx = torch.arange(1, self.time_step + 1, device=x_i.device)
        last_result = None
        sample_flag = 'pq_all_pass'
        sample_eps  = eps_values[-1] if eps_values else 0.0

        with _TimedGetConvBound(self.timing_stats):
            for eps_step, eps in enumerate(eps_values, 1):
                logger.info(f"[S{sample_id}] eps={eps:.6f} ({eps_step}/{len(eps_values)})")

                # --- Phase 1: POPQORN (time only committed when ZS also runs) ---
                _pq_t0 = time.perf_counter()
                self.compute_all_bounds(eps, p, x_i, Eps_idx)
                pq_yL, pq_yU = self._get_output_bounds(eps, p, x_i, Eps_idx)
                _pq_elapsed = time.perf_counter() - _pq_t0
                pq_verified = self._is_verified(pq_yL, pq_yU, top1_i)

                logger.info(f"[S{sample_id}] eps={eps:.6f} PQ={pq_verified}")

                if pq_verified:
                    continue  # ZS not needed; don't count this PQ step

                # PQ failed: commit PQ timing (now denominators match ZS)
                _e = self.timing_stats.setdefault('popqorn', {'total_sec': 0.0, 'count': 0})
                _e['total_sec'] += _pq_elapsed
                _e['count'] += 1

                # --- pq failed: Phase 2 SHAP + Phase 3 ZeroSplit ---

                with Timer(self.timing_stats, 'shap'):
                    ranked = compute_shap_ranking_once(
                        self, x_i, top1_i, eps, p, top_k_neurons=5)

                self._leaf_margins = []
                _t0 = time.perf_counter()
                zs_verified = self._evr_recursive(
                    eps, p, x_i, top1_i, Eps_idx,
                    depth=0, split_history=set(),
                    max_splits=max_splits, ranked=ranked, sample_id=sample_id,
                )
                _e = self.timing_stats.setdefault('zs_total', {'total_sec': 0.0, 'count': 0})
                _e['total_sec'] += time.perf_counter() - _t0
                _e['count'] += 1

                L_val     = int(top1_i[0].item())
                mask_val  = torch.arange(pq_yU.shape[1], device=pq_yU.device) != L_val
                pq_margin = (pq_yL[0, L_val] - pq_yU[0, mask_val].max()).item()
                zs_margin = min(self._leaf_margins) if self._leaf_margins else None
                gain      = (zs_margin - pq_margin) if zs_margin is not None else None

                last_result = {
                    'eps':         eps,
                    'pq_verified': pq_verified,
                    'zs_verified': zs_verified,
                    'pq_margin':   round(pq_margin, 6),
                    'zs_margin':   round(zs_margin, 6) if zs_margin is not None else None,
                    'gain':        round(gain,      6) if gain      is not None else None,
                }

                logger.info(
                    f"[S{sample_id}] eps={eps:.6f} "
                    f"PQ={pq_verified} ZS={zs_verified} "
                    f"pq_margin={pq_margin:.6f} zs_margin={zs_margin}"
                )

                sample_eps  = eps
                if zs_verified:
                    sample_flag = 'zs_better'
                    break  # ZS rescued; stop scanning
                else:
                    sample_flag = 'both_fail'
                    # continue to next eps

        return sample_id, sample_flag, sample_eps, last_result, self.timing_stats

    # ------------------------------------------------------------------
    # Worker (multiprocessing)
    # ------------------------------------------------------------------

    def _build_model_state(self):
        return {
            'input_size':  self.input_size,
            'hidden_size': self.num_neurons,
            'output_size': self.output_size,
            'time_step':   self.time_step,
            'activation':  self.activation,
            'state_dict':  {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }

    @staticmethod
    def _process_single_sample_worker(args):
        (sample_id, x_i, model_state, p, eps_values, max_splits) = args

        _dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, _dir)
        sys.path.insert(0, os.path.join(_dir, '..'))

        from rnn_zerosplit_verifier import RNNZeroSplitVerifier

        ms = model_state
        v = RNNZeroSplitVerifier(
            ms['input_size'], ms['hidden_size'], ms['output_size'],
            ms['time_step'], ms['activation'], max_splits=max_splits,
        )
        v.load_state_dict(ms['state_dict'], strict=False)
        v.extractWeight(clear_original_model=False)

        with torch.no_grad():
            output = v(x_i)
            top1_i = output.argmax(dim=1)

        return v._run_evr_sample(sample_id, x_i, top1_i, p, eps_values, max_splits)

    # ------------------------------------------------------------------
    # EVR main entry
    # ------------------------------------------------------------------

    def verify_evr(self, X, p, eps_range, precision=0.001,
                   max_splits=None, n_workers=None, save_dir=None):
        """
        EVR for N samples in parallel.
        Returns list of (flag, eps, result_dict) per sample.
        Top-1 predicted class is used as the verification target (computed inside each worker).
        """
        if max_splits is None:
            max_splits = self.max_splits
        if n_workers is None:
            n_workers = mp.cpu_count()
        logger.info(f"n_workers={n_workers}")

        low, high = eps_range
        eps_values = []
        cur = low
        while cur <= high + precision / 2:
            eps_values.append(round(cur, 8))
            cur += precision

        model_state  = self._build_model_state()
        worker_args  = [
            (i, X[i:i+1].cpu(), model_state, p, eps_values, max_splits)
            for i in range(X.shape[0])
        ]

        if n_workers > 1:
            with mp.Pool(processes=n_workers) as pool:
                raw = pool.map(self._process_single_sample_worker, worker_args)
        else:
            raw = [self._process_single_sample_worker(a) for a in worker_args]

        raw.sort(key=lambda x: x[0])

        agg_timing     = {}
        sample_records = []
        for sample_id, flag, eps_r, result, t_stats in raw:
            logger.info(
                f"Sample {sample_id}: {flag}  eps={eps_r:.4f}  "
                f"pq={result['pq_verified'] if result else None}  "
                f"zs={result['zs_verified'] if result else None}"
            )
            sample_records.append({
                'sample_id':   sample_id,
                'flag':        flag,
                'eps':         eps_r,
                'pq_verified': result['pq_verified'] if result else None,
                'zs_verified': result['zs_verified'] if result else None,
                'pq_margin':   result.get('pq_margin') if result else None,
                'zs_margin':   result.get('zs_margin') if result else None,
                'gain':        result.get('gain')      if result else None,
            })
            for key, vals in t_stats.items():
                e = agg_timing.setdefault(key, {'total_sec': 0.0, 'count': 0})
                e['total_sec'] += vals['total_sec']
                e['count']     += vals['count']

        for key in agg_timing:
            cnt = agg_timing[key]['count']
            agg_timing[key]['avg_ms'] = (
                round(agg_timing[key]['total_sec'] / cnt * 1000, 4) if cnt > 0 else 0.0
            )

        self._generate_evr_report(sample_records, agg_timing)

        if save_dir is not None:
            _write_evr_json(
                save_dir, sample_records, agg_timing, eps_range, p,
                max_splits, self.num_neurons, self.time_step, self.activation,
            )

        return [(r['flag'], r['eps'], r) for r in sample_records]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _generate_evr_report(self, sample_records, agg_timing):
        N            = len(sample_records)
        zs_better    = sum(1 for r in sample_records if r['flag'] == 'zs_better')
        both_fail    = sum(1 for r in sample_records if r['flag'] == 'both_fail')
        pq_all_pass  = sum(1 for r in sample_records if r['flag'] == 'pq_all_pass')

        logger.info(f"\n{'='*80}")
        logger.info("EVR FINAL REPORT")
        logger.info(f"{'='*80}")
        logger.info(f"Total samples : {N}")
        logger.info(f"  ZS better   (PQ=F, ZS=T)      : {zs_better}/{N}")
        logger.info(f"  Both fail   (PQ=F, ZS=F)      : {both_fail}/{N}")
        logger.info(f"  PQ all pass (ZS never ran)    : {pq_all_pass}/{N}")

        logger.info("\nPer-sample results:")
        for r in sample_records:
            logger.info(
                f"  [{r['sample_id']}] flag={r['flag']:10s} "
                f"eps={r['eps']:.4f}  "
                f"pq_margin={r['pq_margin']}  "
                f"zs_margin={r['zs_margin']}  "
                f"gain={r['gain']}"
            )

        logger.info("\nTiming summary (aggregated across all samples):")
        for key, vals in agg_timing.items():
            logger.info(
                f"  {key:45s}  total={vals['total_sec']:.3f}s  "
                f"count={vals['count']}  avg={vals.get('avg_ms', 0):.2f}ms"
            )
        logger.info(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# JSON result writer (module-level, shared by main)
# ---------------------------------------------------------------------------

def _write_evr_json(save_dir, sample_records, agg_timing,
                    eps_range, p, max_splits, hidden_size, time_step, activation):
    N  = len(sample_records)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (Path(save_dir) /
                   f'session_rnn_{activation}_hidden{hidden_size}_ts{time_step}_p{p}')
    session_dir.mkdir(parents=True, exist_ok=True)

    zs_better   = sum(1 for r in sample_records if r['flag'] == 'zs_better')
    both_fail   = sum(1 for r in sample_records if r['flag'] == 'both_fail')
    pq_all_pass = sum(1 for r in sample_records if r['flag'] == 'pq_all_pass')

    fname = (f'evr_rnn_{activation}_hidden{hidden_size}_ts{time_step}'
             f'_eps{eps_range[0]}-{eps_range[1]}_p{p}'
             f'_N{N}_splits{max_splits}.json')
    path = session_dir / fname

    data = {
        'experiment_info': {
            'timestamp':   ts,
            'hidden_size': hidden_size,
            'time_step':   time_step,
            'activation':  activation,
            'eps_min':     eps_range[0],
            'eps_max':     eps_range[1],
            'p_norm':      p,
            'max_splits':  max_splits,
            'N_samples':   N,
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
                'avg_ms':    v.get('avg_ms', 0.0),
            }
            for k, v in agg_timing.items()
        },
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"EVR results saved → {path}")


# ---------------------------------------------------------------------------
# Toy RNN
# ---------------------------------------------------------------------------

def create_toy_rnn(verifier):
    """Patch a 2-timestep, 1-neuron RNN into an existing verifier instance."""
    with torch.no_grad():
        verifier.rnn = None
        verifier.a_0 = torch.tensor([0.0], dtype=torch.float32)
        verifier.W_ax = torch.tensor([[1.0]], dtype=torch.float32)
        verifier.W_aa = torch.tensor([[1.0]], dtype=torch.float32)
        verifier.W_fa = torch.tensor([[1.0]], dtype=torch.float32)
        verifier.b_ax = torch.tensor([0.0],  dtype=torch.float32)
        verifier.b_aa = torch.tensor([0.0],  dtype=torch.float32)
        verifier.b_f  = torch.tensor([0.0],  dtype=torch.float32)

        def forward(self, X):
            with torch.no_grad():
                N = X.shape[0]
                h = torch.zeros(N, self.time_step + 1, self.num_neurons, device=X.device)
                pre_h = torch.matmul(X[:, 0, :], self.W_ax.t()) + self.b_ax
                h[:, 1, :] = torch.relu(pre_h)
                pre_h = (torch.matmul(X[:, 1, :], self.W_ax.t()) + self.b_ax
                         + torch.matmul(h[:, 1, :], self.W_aa.t()))
                h[:, 2, :] = torch.relu(pre_h)
                return torch.matmul(h[:, 2, :], self.W_fa.t()) + self.b_f

        verifier.forward = types.MethodType(forward, verifier)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'mnist': {
        'input_size': lambda args: 784 // args.time_step,
        'sample': lambda args, verifier, device: sample_mnist_data(
            N=args.N, seq_len=args.time_step, device=device,
            data_dir=args.data_dir, train=False, shuffle=True, rnn=verifier,
        ),
    },
    'mnist-seq': {
        'input_size': lambda args: 3,
        'sample': lambda args, verifier, device: sample_seq_mnist_data(
            N=args.N, time_step=args.time_step, device=device,
            data_dir='./data/mnist_seq/sequences/', train=False, rnn=verifier,
        ),
    },
    'cifar10': {
        'input_size': lambda args: (3072 if args.use_rgb else 1024) // args.time_step,
        'sample': lambda args, verifier, device: sample_cifar10_data(
            N=args.N, time_step=args.time_step, device=device,
            data_dir=args.data_dir, train=False, rnn=verifier,
        ),
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='RNN ZeroSplit Verifier — EVR mode'
    )
    parser.add_argument('--hidden-size',  default=64,     type=int)
    parser.add_argument('--time-step',    default=8,      type=int)
    parser.add_argument('--activation',   default='relu', choices=['tanh', 'relu'])
    parser.add_argument('--dataset',      default='cifar10',
                        choices=list(DATASET_REGISTRY),
                        help='mnist | mnist-seq | cifar10')
    parser.add_argument('--data-dir',     default='./data', type=str)
    parser.add_argument('--work-dir',
                        default='../models/cifar10_classifier/rnn_8_64_relu/',
                        type=str)
    parser.add_argument('--model-name',   default='rnn',  type=str)
    parser.add_argument('--cuda',         action='store_true')
    parser.add_argument('--cuda-idx',     default=0,      type=int)
    parser.add_argument('--N',            default=10,     type=int)
    parser.add_argument('--p',            default=2,      type=int)
    parser.add_argument('--eps-min',      default=0.005,  type=float)
    parser.add_argument('--eps-max',      default=0.015,  type=float)
    parser.add_argument('--max-splits',   default=5,      type=int)
    parser.add_argument('--n-workers',    default=None,   type=int)
    parser.add_argument('--save-dir',     default='./evr_results', type=str)
    parser.add_argument('--toy-rnn',      action='store_true')
    parser.add_argument('--use-rgb',      action='store_true',
                        help='CIFAR10 only: use RGB (3072) instead of greyscale (1024)')
    args = parser.parse_args()

    device = (torch.device(f'cuda:{args.cuda_idx}')
              if args.cuda and torch.cuda.is_available()
              else torch.device('cpu'))
    p = args.p if args.p <= 100 else float('inf')

    if args.toy_rnn:
        logger.info("=== Toy RNN ===")
        verifier = RNNZeroSplitVerifier(
            1, 1, 2, 2, args.activation, max_splits=args.max_splits)
        create_toy_rnn(verifier)
        X = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32, device=device)
    else:
        cfg        = DATASET_REGISTRY[args.dataset]
        input_size = cfg['input_size'](args)
        verifier   = RNNZeroSplitVerifier(
            input_size, args.hidden_size, 10, args.time_step,
            args.activation, max_splits=args.max_splits,
        )
        model_path = os.path.join(args.work_dir, args.model_name)
        verifier.load_state_dict(
            torch.load(model_path, map_location='cpu'), strict=False)
        verifier.to(device)
        verifier.extractWeight(clear_original_model=False)

        X, _, _ = cfg['sample'](args, verifier, device)
        logger.info(f"Sampled {X.shape[0]} {args.dataset} samples  shape={list(X.shape)}")

    logger.info(f"Model      : {args.work_dir if not args.toy_rnn else 'toy'}")
    logger.info(
        f"Config     : eps=[{args.eps_min}, {args.eps_max}]  "
        f"p={p}  N={args.N}  max_splits={args.max_splits}"
    )

    verifier.verify_evr(
        X, p,
        eps_range=(args.eps_min, args.eps_max),
        max_splits=args.max_splits,
        n_workers=args.n_workers,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
