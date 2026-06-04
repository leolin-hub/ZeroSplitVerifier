#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto test script for LSTM ZeroSplit Verifier (MNIST, ReLU-LSTM)
參考 auto_test_cifar10.py，改為 MNIST + ReLU-LSTM 設定。
"""

import subprocess
import itertools
from datetime import datetime, timedelta
import os
import sys
import json
import time

TEST_CONFIG = {
    'hidden_sizes': [4, 8, 16, 32],
    'timesteps': [1, 2, 4, 7],
    'dataset': 'mnist',
    'base_work_dir': '../models/mnist_relu_lstm/',
    'N': 50,
    'p_values': [2],
    'eps_min': 0.01,
    'eps_max': 0.3,
    'max_splits_map': {1: 5, 2: 5, 4: 5, 7: 5},
    'lut_dir': './lookup_tables',
    'save_dir': './lstm/evr_results',
    'n_workers': 4,
    'python_bin': os.environ.get('PYTHON_BIN', sys.executable),
    'pq_only': False,   # True: 只跑 POPQORN timing，不跑 ZeroSplit
}


def get_work_dir():
    return TEST_CONFIG['base_work_dir']


def run_single_test(hidden_size, timestep, p):
    work_dir = get_work_dir()
    max_splits = TEST_CONFIG['max_splits_map'][timestep]

    mode_tag = 'pqonly' if TEST_CONFIG['pq_only'] else 'evr'
    cmd = [
        TEST_CONFIG['python_bin'], 'lstm/lstm_zerosplit_verifier.py',
        '--hidden-size', str(hidden_size),
        '--time-step',   str(timestep),
        '--dataset',     TEST_CONFIG['dataset'],
        '--work-dir',    work_dir,
        '--N',           str(TEST_CONFIG['N']),
        '--p',           str(p),
        '--eps-min',     str(TEST_CONFIG['eps_min']),
        '--eps-max',     str(TEST_CONFIG['eps_max']),
        '--max-splits',  str(max_splits),
        '--lut-dir',     TEST_CONFIG['lut_dir'],
        '--save-dir',    TEST_CONFIG['save_dir'],
        '--n-workers',   str(TEST_CONFIG['n_workers']),
        '--relu',
    ]
    if TEST_CONFIG['pq_only']:
        cmd.append('--pq-only')

    test_name = (
        f"h{hidden_size}_t{timestep}_relu_p{p}"
        f"_{mode_tag}{TEST_CONFIG['eps_min']}-{TEST_CONFIG['eps_max']}"
    )

    print(f"\n{'='*80}")
    print(f"Running: {test_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        return {
            'test_name': test_name,
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'stdout': result.stdout[-500:] if result.stdout else '',
            'stderr': result.stderr[-500:] if result.stderr else '',
            'command': ' '.join(cmd),
            'params': {
                'hidden_size': hidden_size,
                'timestep': timestep,
                'p': p,
                'max_splits': max_splits,
            },
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            'test_name': test_name,
            'status': 'timeout',
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'params': {
                'hidden_size': hidden_size,
                'timestep': timestep,
                'p': p,
                'max_splits': max_splits,
            },
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'test_name': test_name,
            'status': 'error',
            'error': str(e),
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'params': {
                'hidden_size': hidden_size,
                'timestep': timestep,
                'p': p,
                'max_splits': max_splits,
            },
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep', type=int, default=None,
                        help='只跑這個 timestep（給 tmux 各 pane 用）')
    parser.add_argument('--resume', action='store_true',
                        help='跳過 progress.json 裡已完成的測試')
    pane_args = parser.parse_args()

    if pane_args.timestep is not None:
        TEST_CONFIG['timesteps'] = [pane_args.timestep]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = 'pqonly' if TEST_CONFIG['pq_only'] else 'evr'
    results_dir = f"auto_test_results_mnist_{mode_tag}_{timestamp}"

    # ── 斷點恢復 ────────────────────────────────────────
    completed_tests = set()
    if pane_args.resume:
        import glob as _glob
        existing = sorted(_glob.glob(f"auto_test_results_mnist_{mode_tag}_*/progress.json"))
        if existing:
            latest = existing[-1]
            results_dir = os.path.dirname(latest)
            with open(latest) as f:
                prev = json.load(f)
            completed_tests = {
                r['test_name'] for r in prev.get('results', [])
                if r['status'] == 'success'
            }
            print(f"Resume mode: 找到 {len(completed_tests)} 個已完成的測試，將跳過")

    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    start_time = time.time()

    test_combinations = list(itertools.product(
        TEST_CONFIG['hidden_sizes'],
        TEST_CONFIG['timesteps'],
        TEST_CONFIG['p_values'],
    ))

    total_tests = len(test_combinations)
    print(f"\n{'='*80}")
    print(f"Auto Test — LSTM ZeroSplit EVR (MNIST, ReLU-LSTM)")
    print(f"{'='*80}")
    print(f"Dataset   : {TEST_CONFIG['dataset']}")
    print(f"EVR range : [{TEST_CONFIG['eps_min']}, {TEST_CONFIG['eps_max']}]")
    print(f"LUT dir   : {TEST_CONFIG['lut_dir']}")
    print(f"Total tests to run: {total_tests}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    for idx, (hidden_size, timestep, p) in enumerate(test_combinations, 1):
        test_name = (
            f"h{hidden_size}_t{timestep}_relu_p{p}"
            f"_{mode_tag}{TEST_CONFIG['eps_min']}-{TEST_CONFIG['eps_max']}"
        )

        if test_name in completed_tests:
            print(f"[SKIP] {test_name} 已完成")
            continue

        print(f"\nProgress: [{idx}/{total_tests}] ({idx/total_tests*100:.1f}%)")

        if idx > 1:
            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / (idx - 1)
            remaining_tests = total_tests - idx + 1
            estimated_remaining = avg_time_per_test * remaining_tests
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"Avg time/test: {avg_time_per_test:.1f}s | ETA: {eta.strftime('%H:%M:%S')}")

        result = run_single_test(hidden_size, timestep, p)
        all_results.append(result)

        print(f"Status: {result['status']} | Time: {result['elapsed_time']:.1f}s")

        with open(f"{results_dir}/progress.json", 'w') as f:
            json.dump({
                'completed': idx,
                'total': total_tests,
                'last_update': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)

    total_elapsed = time.time() - start_time

    summary = {
        'timestamp': timestamp,
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'total_tests': total_tests,
        'total_elapsed_time': total_elapsed,
        'total_elapsed_formatted': str(timedelta(seconds=int(total_elapsed))),
        'avg_time_per_test': total_elapsed / total_tests if total_tests > 0 else 0,
        'successful': sum(1 for r in all_results if r['status'] == 'success'),
        'failed':     sum(1 for r in all_results if r['status'] == 'failed'),
        'timeout':    sum(1 for r in all_results if r['status'] == 'timeout'),
        'error':      sum(1 for r in all_results if r['status'] == 'error'),
        'config': TEST_CONFIG,
        'results': all_results,
    }

    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(f"{results_dir}/summary.txt", 'w') as f:
        f.write("Auto Test Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset  : {TEST_CONFIG['dataset']} (ReLU-LSTM)\n")
        f.write(f"EVR Range: [{TEST_CONFIG['eps_min']}, {TEST_CONFIG['eps_max']}]\n")
        f.write(f"LUT dir  : {TEST_CONFIG['lut_dir']}\n")
        f.write(f"Start time: {summary['start_time']}\n")
        f.write(f"End time  : {summary['end_time']}\n")
        f.write(f"Total elapsed time: {summary['total_elapsed_formatted']}\n")
        f.write(f"Avg time per test : {total_elapsed/total_tests:.1f}s\n")
        f.write(f"Total tests : {total_tests}\n")
        f.write(f"Successful  : {summary['successful']}\n")
        f.write(f"Failed      : {summary['failed']}\n")
        f.write(f"Timeout     : {summary['timeout']}\n")
        f.write(f"Error       : {summary['error']}\n\n")

        f.write("Test Configuration:\n")
        f.write(f"{'-'*80}\n")
        for key, value in TEST_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Results by Status:\n")
        f.write(f"{'-'*80}\n")
        for status in ['success', 'failed', 'timeout', 'error']:
            status_results = [r for r in all_results if r['status'] == status]
            if status_results:
                f.write(f"\n{status.upper()} ({len(status_results)}):\n")
                for r in status_results:
                    f.write(f"  {r['test_name']} - {r['elapsed_time_formatted']}\n")

    print(f"\n{'='*80}")
    print(f"All tests completed!")
    print(f"Total time: {summary['total_elapsed_formatted']}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()