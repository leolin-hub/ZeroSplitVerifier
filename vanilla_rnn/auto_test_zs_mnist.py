import subprocess
import itertools
from datetime import datetime, timedelta
import os
import sys
import json
import time

# 測試參數配置
TEST_CONFIG = {
    'hidden_sizes': [4, 8, 16, 32],
    'timesteps': [1, 2, 4, 7],
    'activations': ['relu', 'tanh'],
    'base_work_dir': os.path.join(os.environ.get('MODEL_ROOT', '../models'), 'mnist_classifier') + '/',
    'N': 50,
    'p_values': [2],
    'eps_min': 0.005,
    'eps_max': 0.1,
    'eps_step': 0.001,
    'max_splits_map': {1: 1, 2: 2, 4: 4, 7: 7},
    'modes': ['shap'] # , 'max_violation'
}

# eps 掃描：0.005 → 0.1（含）step 0.001
TEST_CONFIG['eps_values'] = [
    round(TEST_CONFIG['eps_min'] + i * TEST_CONFIG['eps_step'], 6)
    for i in range(round((TEST_CONFIG['eps_max'] - TEST_CONFIG['eps_min'])
                         / TEST_CONFIG['eps_step']) + 1)
]

def get_work_dir(timestep, hidden_size, activation):
    return f"{TEST_CONFIG['base_work_dir']}rnn_{timestep}_{hidden_size}_{activation}/"

def run_single_test(hidden_size, timestep, activation, p, eps, mode):
    work_dir = get_work_dir(timestep, hidden_size, activation)
    max_splits = TEST_CONFIG['max_splits_map'][timestep]
    
    cmd = [
        os.environ.get('PYTHON_BIN', sys.executable), 'vanilla_rnn/zerosplit_verifier.py',
        '--hidden-size', str(hidden_size),
        '--time-step', str(timestep),
        '--activation', activation,
        '--work-dir', work_dir,
        '--N', str(TEST_CONFIG['N']),
        '--p', str(p),
        '--eps', str(eps),
        '--max-splits', str(max_splits),
        '--mode', str(mode),
        '--merge-results'
    ]
    
    test_name = f"h{hidden_size}_t{timestep}_{activation}_p{p}_eps{eps}_mode_{mode}"
    print(f"\n{'='*80}")
    print(f"Running: {test_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed_time = time.time() - start_time
        return {
            'test_name': test_name,
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'stdout': result.stdout[-500:] if result.stdout else '',  # ← 加這行（最後500字元）
            'stderr': result.stderr[-500:] if result.stderr else '',
            'params': {
                'hidden_size': hidden_size,
                'timestep': timestep,
                'activation': activation,
                'p': p,
                'eps': eps,
                'mode': mode,
                'max_splits': max_splits
            }
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
                'activation': activation,
                'p': p,
                'eps': eps,
                'mode': mode,
                'max_splits': max_splits
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
                'eps': eps,
                'mode': mode,
                'max_splits': max_splits
            }
        }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"auto_test_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    start_time = time.time()
    
    # 產生所有測試組合
    test_combinations = list(itertools.product(
        TEST_CONFIG['hidden_sizes'],
        TEST_CONFIG['timesteps'],
        TEST_CONFIG['activations'],
        TEST_CONFIG['p_values'],
        TEST_CONFIG['eps_values'],
        TEST_CONFIG['modes']
    ))
    
    total_tests = len(test_combinations)
    print(f"\n{'='*80}")
    print(f"Auto Test for ZeroSplit Verifier")
    print(f"{'='*80}")
    print(f"Total tests to run: {total_tests}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    for idx, (hidden_size, timestep, activation, p, eps, mode) in enumerate(test_combinations, 1):
        print(f"\nProgress: [{idx}/{total_tests}] ({idx/total_tests*100:.1f}%)")
        
        # 估計剩餘時間
        if idx > 1:
            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / (idx - 1)
            remaining_tests = total_tests - idx + 1
            estimated_remaining = avg_time_per_test * remaining_tests
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"Avg time/test: {avg_time_per_test:.1f}s | ETA: {eta.strftime('%H:%M:%S')}")
        
        result = run_single_test(hidden_size, timestep, activation, p, eps, mode)
        all_results.append(result)
        
        print(f"Status: {result['status']} | Time: {result['elapsed_time']:.1f}s")
        
        # 即時儲存進度
        with open(f"{results_dir}/progress.json", 'w') as f:
            json.dump({
                'completed': idx,
                'total': total_tests,
                'last_update': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2)
    
    total_elapsed = time.time() - start_time
    
    # 儲存最終結果
    summary = {
        'timestamp': timestamp,
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'total_tests': total_tests,
        'total_elapsed_time': total_elapsed,
        'total_elapsed_formatted': str(timedelta(seconds=int(total_elapsed))),
        'avg_time_per_test': total_elapsed / total_tests if total_tests > 0 else 0,
        'successful': sum(1 for r in all_results if r['status'] == 'success'),
        'failed': sum(1 for r in all_results if r['status'] == 'failed'),
        'timeout': sum(1 for r in all_results if r['status'] == 'timeout'),
        'error': sum(1 for r in all_results if r['status'] == 'error'),
        'config': TEST_CONFIG,
        'results': all_results
    }
    
    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(f"{results_dir}/summary.txt", 'w') as f:
        f.write(f"Auto Test Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Start time: {summary['start_time']}\n")
        f.write(f"End time: {summary['end_time']}\n")
        f.write(f"Total elapsed time: {summary['total_elapsed_formatted']}\n")
        f.write(f"Avg time per test: {total_elapsed/total_tests:.1f}s\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Successful: {summary['successful']}\n")
        f.write(f"Failed: {summary['failed']}\n")
        f.write(f"Timeout: {summary['timeout']}\n")
        f.write(f"Error: {summary['error']}\n\n")
        
        f.write(f"Test Configuration:\n")
        f.write(f"{'-'*80}\n")
        for key, value in TEST_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n")
        
        f.write(f"Results by Status:\n")
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