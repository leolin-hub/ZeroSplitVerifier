#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto test script for GenBaB (α,β-CROWN) verification of ReLU-LSTM models on MNIST.
Mirrors the structure of auto_test_mnist_lstm.py for side-by-side comparison.

For each (hidden_size, timestep, eps):
  1. Runs abcrown.py in customized_data mode with L2 norm
  2. Parses stdout for per-sample Result lines
  3. Saves results to JSON (same structure as POPQORN evr_results JSONs)

Usage:
  python lstm/auto_test_genbab_mnist.py
  python lstm/auto_test_genbab_mnist.py --timestep 1        # run only one timestep
  python lstm/auto_test_genbab_mnist.py --resume            # skip completed runs
"""

import subprocess
import itertools
from datetime import datetime, timedelta
import os
import sys
import re
import json
import time

GENBAB_CONFIG = {
    'hidden_sizes': [4, 8, 16, 32],
    'timesteps':    [1, 2, 4, 7],
    'eps_values':   [round(0.01 + i * 0.001, 4) for i in range(291)],  # 0.010, 0.011, ..., 0.300
    'N':            50,
    'timeout':      3600,          # 1h hard cap; BaB limited by max_iterations=5
    'omp_threads':  4,             # conservative: 4 threads per pane
    'configs_dir':  '/home/sausage/GenBaB/benchmarks/mnist_relu_lstm/configs',
    'abcrown_py':   '/home/sausage/GenBaB/alpha-beta-CROWN/complete_verifier/abcrown.py',
    'python_bin':   os.environ.get('PYTHON_BIN', sys.executable),
    'save_dir':     'lstm/evr_results_genbab',
}

# Regex patterns for abcrown stdout
_RESULT_RE  = re.compile(r'Result:\s+([\w\-]+(?:\s+\(timed out\))?)\s+in\s+([\d.]+)\s+seconds?')
_CROWN_RE   = re.compile(r'\[TIMING\] crown:\s+([\d.]+)s')
_BAB_RE     = re.compile(r'\[TIMING\] bab:\s+([\d.]+)s')


def parse_results_from_stdout(stdout: str) -> list:
    """
    Parse per-sample results from abcrown stdout.
    Returns list of {'status', 'time_s', 'crown_s', 'bab_s'} dicts.
    TIMING lines appear in the same order as Result lines (one crown+bab pair per sample).
    """
    results   = list(_RESULT_RE.finditer(stdout))
    crowns    = [float(m.group(1)) for m in _CROWN_RE.finditer(stdout)]
    babs      = [float(m.group(1)) for m in _BAB_RE.finditer(stdout)]

    parsed = []
    for idx, m in enumerate(results):
        parsed.append({
            'status':  m.group(1).strip(),
            'time_s':  float(m.group(2)),
            'crown_s': crowns[idx] if idx < len(crowns) else None,
            'bab_s':   babs[idx]   if idx < len(babs)   else None,
        })
    return parsed


def run_one(hidden_size: int, timestep: int, eps: float) -> dict:
    config_path = os.path.join(
        GENBAB_CONFIG['configs_dir'],
        f'config_h{hidden_size}_t{timestep}.yaml'
    )
    eps_str = f'{eps:.3f}'.replace('.', 'p')

    cmd = [
        GENBAB_CONFIG['python_bin'],
        GENBAB_CONFIG['abcrown_py'],
        '--config',    config_path,
        '--epsilon',   str(eps),
        '--device',    'cpu',
        '--start',     '0',
        '--end',       str(GENBAB_CONFIG['N']),
        '--override_timeout', str(GENBAB_CONFIG['timeout']),
        '--results_file', '/dev/null',
    ]

    # Limit CPU threads per subprocess so 4 concurrent panes share 32 cores evenly.
    sub_env = os.environ.copy()
    sub_env['OMP_NUM_THREADS']  = str(GENBAB_CONFIG['omp_threads'])
    sub_env['MKL_NUM_THREADS']  = str(GENBAB_CONFIG['omp_threads'])
    sub_env['OPENBLAS_NUM_THREADS'] = str(GENBAB_CONFIG['omp_threads'])

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=GENBAB_CONFIG['timeout'] * (GENBAB_CONFIG['N'] + 5),
            cwd='/home/sausage/GenBaB/alpha-beta-CROWN/complete_verifier',
            env=sub_env,
        )
        wall = time.time() - start
        parsed = parse_results_from_stdout(proc.stdout)
        # safe-incomplete = CROWN alone; safe = BaB rescued CROWN failure
        n_crown_only  = sum(1 for r in parsed if r['status'] == 'safe-incomplete')
        n_bab_rescued = sum(1 for r in parsed if r['status'] == 'safe')
        n_safe        = n_crown_only + n_bab_rescued
        n_unsafe      = sum(1 for r in parsed if 'unsafe'  in r['status'])
        n_unknown     = sum(1 for r in parsed if 'unknown' in r['status'] or 'timed out' in r['status'])
        crowns        = [r['crown_s'] for r in parsed if r['crown_s'] is not None]
        babs_rescued  = [r['bab_s']  for r in parsed if r['status'] == 'safe' and r['bab_s'] is not None]
        # α,β-CROWN can crash mid-run (e.g. IndexError in branching_domains.add
        # on very-small-eps borderline samples). The samples completed before
        # the crash are still valid; treat the run as 'partial' rather than
        # 'failed' so they contribute to the comparison.
        if proc.returncode == 0:
            run_status = 'success'
        elif len(parsed) > 0:
            run_status = 'partial'
        else:
            run_status = 'failed'
        return {
            'hidden_size':    hidden_size,
            'timestep':       timestep,
            'eps':            eps,
            'eps_str':        eps_str,
            'status':         run_status,
            'returncode':     proc.returncode,
            'wall_time_s':    wall,
            'n_safe':         n_safe,
            'n_crown_only':   n_crown_only,   # CROWN alone certified
            'n_bab_rescued':  n_bab_rescued,  # CROWN failed, BaB rescued ← flipped
            'n_unsafe':       n_unsafe,
            'n_unknown':      n_unknown,
            'n_parsed':       len(parsed),
            'certified_pct':  n_safe / len(parsed) * 100 if parsed else 0.0,
            'avg_time_s':     sum(r['time_s'] for r in parsed) / len(parsed) if parsed else 0.0,
            'avg_crown_s':    sum(crowns)       / len(crowns)       if crowns       else 0.0,
            'avg_bab_s':      sum(babs_rescued) / len(babs_rescued) if babs_rescued else 0.0,
            'per_sample':     parsed,
            'stdout_tail':    proc.stdout[-1000:],
            'stderr_tail':    proc.stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {
            'hidden_size': hidden_size, 'timestep': timestep, 'eps': eps,
            'eps_str': eps_str, 'status': 'timeout',
            'wall_time_s': time.time() - start,
        }
    except Exception as e:
        return {
            'hidden_size': hidden_size, 'timestep': timestep, 'eps': eps,
            'eps_str': eps_str, 'status': 'error', 'error': str(e),
            'wall_time_s': time.time() - start,
        }


def save_run_json(result: dict, save_dir: str) -> None:
    hs, ts, eps = result['hidden_size'], result['timestep'], result['eps']
    sub_dir = os.path.join(save_dir, f'session_genbab_hidden{hs}_ts{ts}_p2')
    os.makedirs(sub_dir, exist_ok=True)
    fname = (f'genbab_lstm_hidden{hs}_ts{ts}'
             f'_eps{eps:.3f}_N{GENBAB_CONFIG["N"]}.json')
    with open(os.path.join(sub_dir, fname), 'w') as f:
        json.dump(result, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep', type=int, default=None,
                        help='Run only this timestep')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Run only this hidden size')
    parser.add_argument('--eps', type=float, default=None,
                        help='Run only this epsilon')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed runs (based on JSON presence)')
    args = parser.parse_args()

    hs_list  = [args.hidden_size] if args.hidden_size else GENBAB_CONFIG['hidden_sizes']
    ts_list  = [args.timestep]    if args.timestep    else GENBAB_CONFIG['timesteps']
    eps_list = [args.eps]         if args.eps         else GENBAB_CONFIG['eps_values']

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'auto_test_results_genbab_mnist_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(GENBAB_CONFIG['save_dir'], exist_ok=True)

    combos = list(itertools.product(hs_list, ts_list, eps_list))
    total  = len(combos)
    all_results = []
    start_overall = time.time()

    print(f'\n{"="*80}')
    print(f'GenBaB Auto Test — ReLU-LSTM MNIST (L2 norm, CPU)')
    print(f'  hidden_sizes : {hs_list}')
    print(f'  timesteps    : {ts_list}')
    print(f'  eps_values   : {eps_list}')
    print(f'  N per run    : {GENBAB_CONFIG["N"]}')
    print(f'  Total runs   : {total}')
    print(f'  Started at   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*80}\n')

    for idx, (hs, ts, eps) in enumerate(combos, 1):
        run_key = f'h{hs}_t{ts}_eps{eps:.3f}'

        if args.resume:
            sub_dir = os.path.join(GENBAB_CONFIG['save_dir'],
                                   f'session_genbab_hidden{hs}_ts{ts}_p2')
            fname   = f'genbab_lstm_hidden{hs}_ts{ts}_eps{eps:.3f}_N{GENBAB_CONFIG["N"]}.json'
            if os.path.exists(os.path.join(sub_dir, fname)):
                print(f'[SKIP] {run_key} already done')
                continue

        elapsed = time.time() - start_overall
        avg_per = elapsed / (idx - 1) if idx > 1 else 0
        eta_str = ''
        if idx > 1:
            eta = datetime.now() + timedelta(seconds=avg_per * (total - idx + 1))
            eta_str = f'  ETA {eta.strftime("%H:%M:%S")}'
        print(f'[{idx}/{total}] {run_key}{eta_str}')

        result = run_one(hs, ts, eps)
        all_results.append(result)
        save_run_json(result, GENBAB_CONFIG['save_dir'])

        status_str = result['status']
        if result['status'] == 'success':
            status_str = (f"cert={result.get('certified_pct', 0):.1f}%  "
                          f"safe={result.get('n_safe',0)} "
                          f"unsafe={result.get('n_unsafe',0)} "
                          f"unknown={result.get('n_unknown',0)} "
                          f"avg={result.get('avg_time_s',0):.3f}s")
        print(f'  → {status_str}  (wall {result["wall_time_s"]:.1f}s)')

        with open(f'{results_dir}/progress.json', 'w') as f:
            json.dump({
                'completed': idx, 'total': total,
                'last_update': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)

    total_wall = time.time() - start_overall
    summary = {
        'timestamp':  timestamp,
        'total_runs': total,
        'completed':  len(all_results),
        'success':    sum(1 for r in all_results if r['status'] == 'success'),
        'failed':     sum(1 for r in all_results if r['status'] not in ('success', 'timeout', 'error')),
        'total_wall_s': total_wall,
        'config': GENBAB_CONFIG,
        'results': all_results,
    }
    with open(f'{results_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*80}')
    print(f'All runs completed!  Total wall time: {timedelta(seconds=int(total_wall))}')
    print(f'Results saved to: {results_dir}/')
    print(f'JSON results at:  {GENBAB_CONFIG["save_dir"]}/')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
