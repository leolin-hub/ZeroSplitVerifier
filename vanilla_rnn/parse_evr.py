"""
parse_evr.py — 解析 rnn_zerosplit_verifier.py 輸出的 EVR JSON 檔案，
生成 Excel 摘要。

JSON 格式（_write_evr_json）：
  experiment_info: {hidden_size, time_step, activation, eps_min, eps_max,
                    p_norm, max_splits, N_samples, timestamp}
  evr_summary:     {zs_better, both_fail, pq_all_pass}
  sample_records:  [{sample_id, flag, eps, pq_verified, zs_verified,
                     pq_margin, zs_margin, gain}, ...]
  timing_stats:    {key: {total_sec, count, avg_ms}, ...}

Flags：
  zs_better   — 第一個 pq 失敗的 eps，ZS 驗證成功
  both_fail   — 第一個 pq 失敗的 eps，ZS 也失敗
  pq_all_pass — 所有 eps pq 均通過，ZS 從未啟動
"""

import json
import glob
import pandas as pd
from pathlib import Path


def parse_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ei = data['experiment_info']
    es = data.get('evr_summary', {})
    sr = data.get('sample_records', [])
    ts = data.get('timing_stats', {})

    N = ei.get('N_samples', len(sr))

    # --- timing ---
    def _t(key):
        return ts.get(key, {}).get('avg_ms', 0.0)

    # --- per-sample stats ---
    zs_better_samples   = [r for r in sr if r.get('flag') == 'zs_better']
    both_fail_samples   = [r for r in sr if r.get('flag') == 'both_fail']
    pq_all_pass_samples = [r for r in sr if r.get('flag') == 'pq_all_pass']

    gains = [r['gain'] for r in sr if r.get('gain') is not None]
    avg_gain = round(sum(gains) / len(gains), 6) if gains else None

    row = {
        'file_name':    Path(json_path).stem,
        'timestamp':    ei.get('timestamp', ''),
        'hidden_size':  ei.get('hidden_size'),
        'time_step':    ei.get('time_step'),
        'activation':   ei.get('activation'),
        'eps_min':      ei.get('eps_min'),
        'eps_max':      ei.get('eps_max'),
        'p_norm':       ei.get('p_norm'),
        'max_splits':   ei.get('max_splits'),
        'N_samples':    N,

        # EVR summary
        'zs_better':    es.get('zs_better',   0),
        'both_fail':    es.get('both_fail',    0),
        'pq_all_pass':  es.get('pq_all_pass',  0),

        # derived
        'avg_gain':     avg_gain,

        # timing (avg ms per call)
        'timing_popqorn_avg_ms':  _t('popqorn'),
        'timing_shap_avg_ms':     _t('shap'),
        'timing_zs_total_avg_ms': _t('zs_total'),
        'timing_getbound_avg_ms': _t('getConvenientGeneralActivationBound'),
    }

    return row


def main(results_dir='./evr_results', output_excel='evr_summary.xlsx'):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    json_pattern = str(results_dir / 'session_rnn_*/evr_rnn_*.json')
    json_files   = glob.glob(json_pattern)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found: {json_pattern}")

    print(f"Found {len(json_files)} JSON file(s)")

    rows = []
    for i, jf in enumerate(json_files, 1):
        try:
            print(f"  [{i}/{len(json_files)}] {Path(jf).name}", end='')
            rows.append(parse_json_file(jf))
            print(" OK")
        except Exception as e:
            print(f" FAIL: {e}")
            import traceback; traceback.print_exc()

    if not rows:
        raise ValueError("No valid data extracted")

    df = pd.DataFrame(rows)
    df = df.sort_values(['activation', 'hidden_size', 'time_step', 'eps_min'])
    df.to_excel(output_excel, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(rows)} rows → {output_excel}")
    print(f"{'='*60}")
    print(f"Hidden sizes : {sorted(df['hidden_size'].unique())}")
    print(f"Time steps   : {sorted(df['time_step'].unique())}")
    print(f"Activations  : {sorted(df['activation'].unique())}")
    print(f"ZS better    : {df['zs_better'].sum()} / {df['N_samples'].sum()}")
    print(f"Both fail    : {df['both_fail'].sum()} / {df['N_samples'].sum()}")
    print(f"PQ all pass  : {df['pq_all_pass'].sum()} / {df['N_samples'].sum()}")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='./evr_results')
    parser.add_argument('--output',    default='evr_summary.xlsx')
    args = parser.parse_args()
    main(args.input_dir, args.output)
