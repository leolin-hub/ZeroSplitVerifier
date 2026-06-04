#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse and compare GenBaB vs POPQORN results for ReLU-LSTM MNIST verification.

Reads:
  - GenBaB:   lstm/evr_results_genbab/session_genbab_hidden{h}_ts{t}_p2/*.json
  - POPQORN:  lstm/evr_results/session_lstm_hidden{h}_ts{t}_p2/*.json

Outputs a comparison Excel (or CSV) table with columns:
  hidden | ts | eps | N |
  POPQORN_cert%  POPQORN_avg_ms  POPQORN_pq_only_cert% |
  GenBaB_cert%   GenBaB_avg_ms

Usage:
  python lstm/parse_genbab_results.py
  python lstm/parse_genbab_results.py --genbab-dir lstm/evr_results_genbab \
                                       --popqorn-dir lstm/evr_results \
                                       --output comparison.xlsx
"""

import os
import json
import glob
import argparse
from pathlib import Path

HIDDEN_SIZES = [4, 8, 16, 32]
TIMESTEPS    = [1, 2, 4, 7]
EPS_VALUES   = [round(0.01 + i * 0.001, 4) for i in range(291)]  # 0.010 ~ 0.300

GENBAB_DIR   = 'lstm/evr_results_genbab'


# ── GenBaB result parsing ─────────────────────────────────────────────────────

def load_genbab_result(genbab_dir, hs, ts, eps):
    pattern = os.path.join(
        genbab_dir,
        f'session_genbab_hidden{hs}_ts{ts}_p2',
        f'genbab_lstm_hidden{hs}_ts{ts}_eps{eps:.3f}_N*.json'
    )
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)



# ── Main ──────────────────────────────────────────────────────────────────────

def build_table(genbab_dir):
    rows = []

    for hs in HIDDEN_SIZES:
        for ts in TIMESTEPS:
            for eps in EPS_VALUES:
                row = {
                    'hidden': hs, 'ts': ts, 'eps': eps,
                }

                # GenBaB
                gb = load_genbab_result(genbab_dir, hs, ts, eps)
                if gb and gb.get('status') in ('success', 'partial'):
                    ps = gb.get('per_sample', [])
                    # Per-sample index lists. Index in per_sample = sample index
                    # (abcrown processes samples sequentially from --start to --end).
                    crown_idx   = [i for i, r in enumerate(ps) if r.get('status') == 'safe-incomplete']
                    rescued_idx = [i for i, r in enumerate(ps) if r.get('status') == 'safe']
                    unsafe_idx  = [i for i, r in enumerate(ps) if 'unsafe'  in r.get('status', '')]
                    unknown_idx = [i for i, r in enumerate(ps) if 'unknown' in r.get('status', '')
                                                              or 'timed out' in r.get('status', '')]
                    crown_only  = len(crown_idx)
                    bab_rescued = len(rescued_idx)
                    n_unknown   = len(unknown_idx)
                    n_total     = len(ps) or 1
                    # Fall back to stored fields if per_sample missing (old JSONs)
                    if not ps:
                        crown_only  = gb.get('n_crown_only',  None)
                        bab_rescued = gb.get('n_bab_rescued', None)
                        n_unknown   = gb.get('n_unknown',     None)

                    # Restrict timing averages to the CROWN-failed subset
                    # (samples that triggered BaB). Reason:
                    #  - avg CROWN ms = cost of CROWN on samples it could NOT
                    #    certify (the meaningful "wasted CROWN" baseline per
                    #    refinement attempt). Excludes safe-incomplete.
                    #  - avg BaB ms   = cost of refinement per attempt,
                    #    regardless of whether BaB ultimately succeeded
                    #    (rescued / unknown / timeout all count).
                    crown_failed_rows = [r for r in ps
                                         if r.get('status') != 'safe-incomplete']
                    crown_times_failed = [r['crown_s'] for r in crown_failed_rows
                                          if r.get('crown_s') is not None]
                    bab_times_all      = [r['bab_s']   for r in crown_failed_rows
                                          if r.get('bab_s')   is not None]

                    row['genbab_cert_pct']     = gb.get('certified_pct', None)
                    row['genbab_crown_only']   = crown_only
                    row['genbab_bab_rescued']  = bab_rescued
                    row['genbab_n_unknown']    = n_unknown
                    row['genbab_N']            = gb.get('n_parsed', n_total)
                    row['genbab_avg_total_ms'] = gb.get('avg_time_s', 0) * 1000
                    row['genbab_avg_crown_ms'] = (sum(crown_times_failed) / len(crown_times_failed) * 1000
                                                  if crown_times_failed else None)
                    row['genbab_avg_bab_ms']   = (sum(bab_times_all) / len(bab_times_all) * 1000
                                                  if bab_times_all else None)
                    # Sample-level detail (so you can see exactly WHICH samples
                    # BaB rescued vs. which CROWN alone certified vs. unknown).
                    row['rescued_sample_idx']  = ','.join(map(str, rescued_idx))
                    row['crown_sample_idx']    = ','.join(map(str, crown_idx))
                    row['unknown_sample_idx']  = ','.join(map(str, unknown_idx))
                    row['unsafe_sample_idx']   = ','.join(map(str, unsafe_idx))
                else:
                    for k in ['genbab_cert_pct', 'genbab_crown_only', 'genbab_bab_rescued',
                              'genbab_n_unknown', 'genbab_N',
                              'genbab_avg_total_ms', 'genbab_avg_crown_ms', 'genbab_avg_bab_ms',
                              'rescued_sample_idx', 'crown_sample_idx',
                              'unknown_sample_idx', 'unsafe_sample_idx']:
                        row[k] = None


                rows.append(row)

    return rows


def _fmt(v, fmt='.1f', na='N/A'):
    return format(v, fmt) if v is not None else na

def print_table(rows):
    header = (
        f"{'hs':>4} {'ts':>3} {'eps':>6} | "
        f"{'cert%':>6} {'crown':>6} {'flipped':>7} {'unkn':>5} {'crown_ms':>9} {'bab_ms':>7}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        print(
            f"{r['hidden']:>4} {r['ts']:>3} {r['eps']:>6.3f} | "
            f"{_fmt(r['genbab_cert_pct']):>6} "
            f"{_fmt(r['genbab_crown_only'], '.0f'):>6} "
            f"{_fmt(r['genbab_bab_rescued'], '.0f'):>7} "
            f"{_fmt(r['genbab_n_unknown'],   '.0f'):>5} "
            f"{_fmt(r['genbab_avg_crown_ms']):>9} "
            f"{_fmt(r['genbab_avg_bab_ms']):>7}"
        )


def print_rescue_highlights(rows):
    """List per-cell which sample indices BaB rescued (CROWN failed → BaB safe)."""
    interesting = [r for r in rows
                   if r.get('genbab_bab_rescued') or r.get('genbab_n_unknown')]
    if not interesting:
        print('\nNo BaB rescues or unknowns recorded across all cells.')
        return
    print('\n' + '=' * 80)
    print('Rescue & unknown highlights — per (hs, ts, eps) cell with BaB activity')
    print('=' * 80)
    print(f"{'hs':>3} {'ts':>3} {'eps':>6} | "
          f"{'resc':>4} {'unkn':>4} | rescued sample idx | unknown sample idx")
    print('-' * 80)
    for r in interesting:
        print(
            f"{r['hidden']:>3} {r['ts']:>3} {r['eps']:>6.3f} | "
            f"{(r.get('genbab_bab_rescued') or 0):>4} "
            f"{(r.get('genbab_n_unknown') or 0):>4} | "
            f"{(r.get('rescued_sample_idx') or ''):<18} | "
            f"{(r.get('unknown_sample_idx') or '')}"
        )


def save_excel(rows, path):
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        print(f'Saved Excel → {path}')
    except ImportError:
        import csv
        csv_path = path.replace('.xlsx', '.csv')
        keys = rows[0].keys() if rows else []
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'Saved CSV (pandas not available) → {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genbab-dir', default=GENBAB_DIR)
    parser.add_argument('--output',     default='lstm/genbab_results.xlsx')
    args = parser.parse_args()

    rows = build_table(args.genbab_dir)
    print_table(rows)
    print_rescue_highlights(rows)
    save_excel(rows, args.output)


if __name__ == '__main__':
    main()
