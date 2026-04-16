#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
branching_point_optimizer.py

Pre-computes optimal branching points p* for BaB verification (GenBaB, TACAS 2025).

For a neuron with pre-activation bound [l, u], finds p* minimising
total linear relaxation tightness after splitting into [l, p] and [p, u]:

    P(f, l, u, p) = ∫[l→p] gap₁(x) dx + ∫[p→u] gap₂(x) dx

where gap_i(x) = upper_bound_i(x) - lower_bound_i(x) on each sub-interval.
All linear relaxations use CROWN-style bounds from POPQORN's existing code.

Supports: relu, sigmoid, tanh.

Build LUT (one-time, offline):
    python branching_point_optimizer.py --func all --output_dir ./lookup_tables

Query at verification time:
    from branching_point_optimizer import load_lookup_table, query_lookup_table_1d
    table = load_lookup_table('./lookup_tables/tanh_lookup.pkl')
    p_star = query_lookup_table_1d(table, l=-1.5, u=2.3)

LUT design:
  Phase 1 – batch-compute ALL (grid[i], grid[j]) relaxations in one vectorised call
             (~500K pairs for [-5,5]/0.01; binary-search cost amortised over batch).
  Phase 2 – compute single-interval tightness losses (vectorised, instant).
  Phase 3 – fill optimal_p[i, k] = argmin_p (loss[i,p] + loss[p,k])
             via k-indexed loop; each iteration is a (k×k) numpy operation.
"""

import sys
import os
import pickle
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
# get_bound_for_general_activation_function.py lives in vanilla_rnn/
sys.path.insert(0, os.path.join(_DIR, '..'))
sys.path.insert(0, os.path.join(_DIR, '../vanilla_rnn'))

import numpy as np
import torch
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from get_bound_for_general_activation_function import getConvenientGeneralActivationBound

SUPPORTED = ('relu', 'sigmoid', 'tanh')


# ---------------------------------------------------------------------------
# Linear relaxation wrapper  (reuses POPQORN's CROWN-style bounding)
# ---------------------------------------------------------------------------

def linear_relax_batch(
    activation: str,
    l_arr: np.ndarray,
    u_arr: np.ndarray,
) -> tuple:
    """
    Batched linear relaxation: kl*x + bl <= f(x) <= ku*x + bu on [l_arr[i], u_arr[i]].

    Thin wrapper around getConvenientGeneralActivationBound so the rest of this
    module works with plain numpy arrays instead of torch tensors.

    Returns (kl, bl, ku, bu) as float64 numpy arrays, shape (N,).
    """
    l_t = torch.tensor(l_arr, dtype=torch.float32)
    u_t = torch.tensor(u_arr, dtype=torch.float32)
    with torch.no_grad():
        kl, bl, ku, bu = getConvenientGeneralActivationBound(l_t, u_t, activation)
    return (kl.numpy().astype(np.float64),
            bl.numpy().astype(np.float64),
            ku.numpy().astype(np.float64),
            bu.numpy().astype(np.float64))


# ---------------------------------------------------------------------------
# Analytical tightness loss for a single segment [l, u]
# ---------------------------------------------------------------------------

def _segment_loss_vec(
    kl: np.ndarray, bl: np.ndarray,
    ku: np.ndarray, bu: np.ndarray,
    l:  np.ndarray, u:  np.ndarray,
) -> np.ndarray:
    """
    ∫[l→u] ((ku - kl)·x + (bu - bl)) dx, vectorised over arrays.

    = (ku-kl)/2 · (u²-l²) + (bu-bl) · (u-l)
    """
    dk = ku - kl
    db = bu - bl
    return dk / 2.0 * (u**2 - l**2) + db * (u - l)


# ---------------------------------------------------------------------------
# LUT construction
# ---------------------------------------------------------------------------

def build_lookup_table_1d(
    activation: str,
    bound_range: tuple = (-5.0, 5.0),
    step_size: float = 0.01,
    save_path: str = None,
) -> dict:
    """
    Build a 1-D lookup table: optimal_p[i, j] = p* for interval [grid[i], grid[j]].

    Three-phase build:

    Phase 1  Batch-compute CROWN relaxations for every valid (l, u) pair in the
             grid — one vectorised torch call, binary search amortised over the
             whole batch (~500K pairs for the default range/step).

    Phase 2  Compute single-interval tightness losses (all-pairs, vectorised).

    Phase 3  For each (i, k) with k >= i+2, find
                 p* = argmin_{p in (i,k)} [ loss[i,p] + loss[p,k] ]
             using a k-indexed loop; the inner work per k is a (k×k) numpy op.

    Args:
        activation:  'relu', 'sigmoid', or 'tanh'
        bound_range: (lo, hi) grid extent, default (-5, 5)
        step_size:   grid spacing, default 0.01
        save_path:   if given, pickle the table to this path

    Returns:
        dict with keys:
            func_name, bound_range, step_size, l_grid, u_grid, optimal_p
        optimal_p: float32 array (N, N), NaN for invalid i >= j entries.
        l_grid == u_grid: the shared 1-D grid of N points.
    """
    assert activation in SUPPORTED, f'Unsupported activation: {activation!r}'

    N = int(round((bound_range[1] - bound_range[0]) / step_size)) + 1
    grid = np.linspace(bound_range[0], bound_range[1], N)
    n_pairs = N * (N - 1) // 2
    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: precompute relaxations for all (grid[i], grid[j]), i < j
    # ------------------------------------------------------------------
    print(f'[{activation}] Phase 1 — {n_pairs:,} relaxations (batch)…')
    t0 = time.perf_counter()

    i_idx, j_idx = np.triu_indices(N, k=1)          # (M,) each, M = n_pairs
    kl_f, bl_f, ku_f, bu_f = linear_relax_batch(
        activation, grid[i_idx], grid[j_idx]
    )

    # Store flat results back into (N, N) 2-D arrays
    kl_2d = np.full((N, N), np.nan, dtype=np.float64)
    bl_2d = np.full((N, N), np.nan, dtype=np.float64)
    ku_2d = np.full((N, N), np.nan, dtype=np.float64)
    bu_2d = np.full((N, N), np.nan, dtype=np.float64)
    kl_2d[i_idx, j_idx] = kl_f
    bl_2d[i_idx, j_idx] = bl_f
    ku_2d[i_idx, j_idx] = ku_f
    bu_2d[i_idx, j_idx] = bu_f
    print(f'[{activation}] Phase 1 done  {time.perf_counter()-t0:.1f}s')

    # ------------------------------------------------------------------
    # Phase 2: single-interval tightness losses, vectorised
    # ------------------------------------------------------------------
    print(f'[{activation}] Phase 2 — segment losses…')
    t0 = time.perf_counter()

    loss_2d = np.full((N, N), np.nan, dtype=np.float64)
    loss_2d[i_idx, j_idx] = _segment_loss_vec(
        kl_f, bl_f, ku_f, bu_f, grid[i_idx], grid[j_idx]
    )
    print(f'[{activation}] Phase 2 done  {time.perf_counter()-t0:.1f}s')

    # ------------------------------------------------------------------
    # Phase 3: fill LUT by argmin over candidate split points
    # ------------------------------------------------------------------
    print(f'[{activation}] Phase 3 — filling {N}×{N} LUT…')
    t0 = time.perf_counter()

    optimal_p = np.full((N, N), np.nan, dtype=np.float32)

    # k == i+1: sub-interval is one grid step — no interior grid point exists.
    # Fall back to midpoint; these pairs are handled by query_lookup_table_1d.
    # (optimal_p[i, i+1] stays NaN; the query function returns midpoint for NaN.)

    # k >= i+2: at least one grid-aligned candidate p = grid[i+1..k-1] exists.
    # For fixed k, build a (k × k) cost matrix:
    #   total[i, p] = loss_2d[i, p]  +  loss_2d[p, k]
    # Valid region: upper triangle (p > i).  Lower triangle → inf.
    # Row argmin over [:k-1] rows gives best p for each i.
    for k in tqdm(range(2, N), desc=f'{activation}', unit='k'):
        left  = loss_2d[:k, :k]                        # (k, k): left[i, p]
        right = loss_2d[:k, k]                         # (k,):   right[p]
        total = left + right[np.newaxis, :]            # (k, k): broadcast

        # Mask invalid positions (p <= i, includes diagonal and lower triangle)
        lower_tri = np.tril(np.ones((k, k), dtype=bool))
        total[lower_tri] = np.inf                      # in-place on the new array

        # argmin over columns → best p index for each row i in [0, k-2]
        best_p_idx = np.argmin(total[:k-1], axis=1)   # (k-1,)
        optimal_p[:k-1, k] = grid[best_p_idx].astype(np.float32)

    print(f'[{activation}] Phase 3 done  {time.perf_counter()-t0:.1f}s')
    print(f'[{activation}] Total build time: {time.perf_counter()-t_total:.1f}s')

    table = {
        'func_name':   activation,
        'bound_range': bound_range,
        'step_size':   step_size,
        'l_grid':      grid.astype(np.float32),
        'u_grid':      grid.astype(np.float32),
        'optimal_p':   optimal_p,           # float32 (N, N)
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(table, f, protocol=4)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f'[{activation}] Saved → {save_path}  ({size_mb:.1f} MB)')

    return table


# ---------------------------------------------------------------------------
# LUT query
# ---------------------------------------------------------------------------

def load_lookup_table(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def query_lookup_table_1d(table: dict, l: float, u: float) -> float:
    """
    Nearest-neighbour lookup in the pre-built table.

    Returns p* ∈ (l, u).  Falls back to midpoint when:
      - the table entry is NaN (k == i+1 pairs, or l >= u in grid)
      - the stored p* is outside (l, u) due to rounding at grid boundaries
    """
    grid = table['l_grid']
    idx_l = int(np.argmin(np.abs(grid - l)))
    idx_u = int(np.argmin(np.abs(grid - u)))

    p = float(table['optimal_p'][idx_l, idx_u])
    if np.isnan(p) or not (l < p < u):
        return (l + u) / 2.0
    return p


# ---------------------------------------------------------------------------
# Convenience: build all supported tables
# ---------------------------------------------------------------------------

def build_all_lookup_tables(
    output_dir: str = './lookup_tables',
    bound_range: tuple = (-5.0, 5.0),
    step_1d: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for act in SUPPORTED:
        path = os.path.join(output_dir, f'{act}_lookup.pkl')
        build_lookup_table_1d(act, bound_range, step_1d, path)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_correctness() -> None:
    print('\n=== Correctness Tests ===\n')

    step = 0.01
    funcs_np = {
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
        'tanh':    np.tanh,
        'relu':    lambda x: np.maximum(x, 0.0),
    }

    # ------------------------------------------------------------------
    # 1. Soundness: kl*x+bl <= f(x) <= ku*x+bu for sampled x in [l, u]
    # ------------------------------------------------------------------
    print('1. Soundness (100-point sample per interval)')
    cases = [
        ('sigmoid', -3.0,  2.0),
        ('sigmoid',  0.0,  1.0),
        ('sigmoid', -1.0, -0.1),
        ('tanh',   -2.0,  1.0),
        ('tanh',    0.0,  2.0),
        ('tanh',   -1.5, -0.2),
        ('relu',   -2.0,  3.0),
        ('relu',    0.5,  2.0),
        ('relu',   -3.0, -0.5),
    ]
    for act, l_val, u_val in cases:
        kl, bl, ku, bu = linear_relax_batch(act,
                                            np.array([l_val]),
                                            np.array([u_val]))
        xs   = np.linspace(l_val, u_val, 200)
        f_xs = funcs_np[act](xs)
        lb_ok = bool(np.all(kl[0]*xs + bl[0] <= f_xs + 1e-5))
        ub_ok = bool(np.all(ku[0]*xs + bu[0] >= f_xs - 1e-5))
        status = 'OK  ' if (lb_ok and ub_ok) else 'FAIL'
        print(f'  {status} {act:7s} [{l_val:5.1f}, {u_val:5.1f}]  '
              f'lb={lb_ok} ub={ub_ok}')
        assert lb_ok and ub_ok, f'Soundness failed: {act} [{l_val}, {u_val}]'

    # ------------------------------------------------------------------
    # 2. Optimality: p* <= loss(midpoint) for tanh cross-zero case
    # ------------------------------------------------------------------
    print('\n2. Optimality (tanh [-2, 3]): p* should beat midpoint')
    l, u = -2.0, 3.0
    p_arr = np.arange(l + step, u, step)
    kl1, bl1, ku1, bu1 = linear_relax_batch('tanh',
                                             np.full(len(p_arr), l), p_arr)
    kl2, bl2, ku2, bu2 = linear_relax_batch('tanh',
                                             p_arr, np.full(len(p_arr), u))
    losses = (
        _segment_loss_vec(kl1, bl1, ku1, bu1, np.full(len(p_arr), l), p_arr) +
        _segment_loss_vec(kl2, bl2, ku2, bu2, p_arr, np.full(len(p_arr), u))
    )
    best_idx = int(np.argmin(losses))
    p_star   = float(p_arr[best_idx])
    mid_idx  = int(np.argmin(np.abs(p_arr - (l + u) / 2.0)))
    loss_opt = losses[best_idx]
    loss_mid = losses[mid_idx]
    print(f'  midpoint  p={(l+u)/2:.3f}  loss={loss_mid:.6f}')
    print(f'  optimal   p={p_star:.3f}  loss={loss_opt:.6f}')
    assert loss_opt <= loss_mid + 1e-8, 'p* must be at least as good as midpoint'
    print('  OK: loss(p*) <= loss(midpoint)')

    # ------------------------------------------------------------------
    # 3. relu: p* ≈ 0 for cross-zero interval
    # ------------------------------------------------------------------
    print('\n3. relu cross-zero: p* should be near 0')
    l_r, u_r = -2.0, 3.0
    p_arr_r  = np.arange(l_r + step, u_r, step)
    kl1, bl1, ku1, bu1 = linear_relax_batch('relu',
                                             np.full(len(p_arr_r), l_r), p_arr_r)
    kl2, bl2, ku2, bu2 = linear_relax_batch('relu',
                                             p_arr_r, np.full(len(p_arr_r), u_r))
    losses_r = (
        _segment_loss_vec(kl1, bl1, ku1, bu1, np.full(len(p_arr_r), l_r), p_arr_r) +
        _segment_loss_vec(kl2, bl2, ku2, bu2, p_arr_r, np.full(len(p_arr_r), u_r))
    )
    p_star_r = float(p_arr_r[np.argmin(losses_r)])
    print(f'  p* = {p_star_r:.4f}  (expected ~0.00)')
    assert abs(p_star_r) < 0.05, f'relu p* should be near 0, got {p_star_r}'
    print('  OK')

    # ------------------------------------------------------------------
    # 4. Small LUT round-trip: build tiny table and query it
    # ------------------------------------------------------------------
    print('\n4. LUT round-trip (sigmoid, [-1, 1], step=0.1)')
    small_table = build_lookup_table_1d('sigmoid',
                                        bound_range=(-1.0, 1.0),
                                        step_size=0.1)
    p_q = query_lookup_table_1d(small_table, l=-0.8, u=0.7)
    assert -0.8 < p_q < 0.7, f'query result {p_q} not in (-0.8, 0.7)'
    print(f'  query(-0.8, 0.7) → p*={p_q:.4f}  OK')

    print('\n=== All tests passed ===\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build branching-point lookup tables for BaB verification')
    parser.add_argument('--func',
                        choices=[*SUPPORTED, 'all'], default='all',
                        help='Which activation to build (default: all)')
    parser.add_argument('--output_dir',  default='./lookup_tables',
                        help='Directory for pkl files (default: ./lookup_tables)')
    parser.add_argument('--bound_range', type=float, nargs=2, default=[-5.0, 5.0],
                        metavar=('LO', 'HI'),
                        help='Grid extent (default: -5 5)')
    parser.add_argument('--step_1d',     type=float, default=0.01,
                        help='Grid step size (default: 0.01)')
    parser.add_argument('--test',        action='store_true',
                        help='Run correctness tests and exit')
    args = parser.parse_args()

    if args.test:
        test_correctness()
    elif args.func == 'all':
        build_all_lookup_tables(
            args.output_dir, tuple(args.bound_range), args.step_1d
        )
    else:
        path = os.path.join(args.output_dir, f'{args.func}_lookup.pkl')
        build_lookup_table_1d(
            args.func, tuple(args.bound_range), args.step_1d, path
        )
