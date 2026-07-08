import subprocess
import itertools
from datetime import datetime, timedelta
import os
import sys
import json
import time

TEST_CONFIG = {
    'hidden_sizes': [16, 32, 64, 128],
    'timesteps': [30, 35, 40, 45],
    'activations': ['relu', 'tanh'],
    'dataset': 'mnist-seq',
    'base_work_dir': os.path.join(os.environ.get('MODEL_ROOT', '../models'), 'mnist_seq_classifier') + '/',
    'N': 50,
    'p_values': [2],
    'eps_min': 0.005,
    'eps_max': 0.1,
    'max_splits_map': {30: 30, 35: 35, 40: 40, 45: 45},
    'save_dir': './evr_results',
}

def get_work_dir(timestep, hidden_size, activation):
    return f"{TEST_CONFIG['base_work_dir']}rnn_seq_{timestep}_{hidden_size}_{activation}/"

def run_single_test(hidden_size, timestep, activation, p):
    work_dir  = get_work_dir(timestep, hidden_size, activation)
    max_splits = TEST_CONFIG['max_splits_map'][timestep]

    cmd = [
        os.environ.get('PYTHON_BIN', sys.executable), 'vanilla_rnn/rnn_zerosplit_verifier.py',
        '--hidden-size',  str(hidden_size),
        '--time-step',    str(timestep),
        '--activation',   activation,
        '--dataset',      TEST_CONFIG['dataset'],
        '--work-dir',     work_dir,
        '--N',            str(TEST_CONFIG['N']),
        '--p',            str(p),
        '--eps-min',      str(TEST_CONFIG['eps_min']),
        '--eps-max',      str(TEST_CONFIG['eps_max']),
        '--max-splits',   str(max_splits),
        '--save-dir',     TEST_CONFIG['save_dir'],
    ]

    test_name = (f"h{hidden_size}_t{timestep}_{activation}_p{p}"
                 f"_evr{TEST_CONFIG['eps_min']}-{TEST_CONFIG['eps_max']}")
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
            'params': {
                'hidden_size': hidden_size,
                'timestep': timestep,
                'activation': activation,
                'p': p,
                'max_splits': max_splits,
            }
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
                'activation': activation,
                'p': p,
                'max_splits': max_splits,
            }
        }


def main():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"auto_test_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    start_time  = time.time()

    test_combinations = list(itertools.product(
        TEST_CONFIG['hidden_sizes'],
        TEST_CONFIG['timesteps'],
        TEST_CONFIG['activations'],
        TEST_CONFIG['p_values'],
    ))

    total_tests = len(test_combinations)
    print(f"\n{'='*80}")
    print(f"Auto Test — RNN ZeroSplit EVR (rnn_zerosplit_verifier.py)")
    print(f"{'='*80}")
    print(f"EVR range : [{TEST_CONFIG['eps_min']}, {TEST_CONFIG['eps_max']}]")
    print(f"Total tests: {total_tests}")
    print(f"Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    for idx, (hidden_size, timestep, activation, p) in enumerate(test_combinations, 1):
        print(f"\nProgress: [{idx}/{total_tests}] ({idx/total_tests*100:.1f}%)")

        if idx > 1:
            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / (idx - 1)
            eta = datetime.now() + timedelta(seconds=avg_time_per_test * (total_tests - idx + 1))
            print(f"Avg time/test: {avg_time_per_test:.1f}s | ETA: {eta.strftime('%H:%M:%S')}")

        result = run_single_test(hidden_size, timestep, activation, p)
        all_results.append(result)

        print(f"Status: {result['status']} | Time: {result['elapsed_time']:.1f}s")

        with open(f"{results_dir}/progress.json", 'w') as f:
            json.dump({
                'completed': idx,
                'total': total_tests,
                'last_update': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2)

    total_elapsed = time.time() - start_time

    summary = {
        'timestamp': timestamp,
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'total_tests': total_tests,
        'total_elapsed_formatted': str(timedelta(seconds=int(total_elapsed))),
        'avg_time_per_test': total_elapsed / total_tests if total_tests > 0 else 0,
        'successful': sum(1 for r in all_results if r['status'] == 'success'),
        'failed':     sum(1 for r in all_results if r['status'] == 'failed'),
        'error':      sum(1 for r in all_results if r['status'] == 'error'),
        'config': TEST_CONFIG,
        'results': all_results,
    }

    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"All tests completed!")
    print(f"Total time: {summary['total_elapsed_formatted']}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
