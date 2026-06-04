#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import glob
import pandas as pd
from pathlib import Path

TIMING_KEYS = [
    'get_y',
    'get_hfc',
    'get_hig',
    'get_c',
    'get_hoc',
    'get_Wa_b_hidden',
    'get_Wa_b_output',
    'zs_total',
]


def parse_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    info    = data['experiment_info']
    summary = data.get('evr_summary', {})
    timing  = data.get('timing_stats', {})

    row = {
        'file_name':    Path(json_path).stem,
        'timestamp':    info.get('timestamp', ''),
        'hidden_size':  info['hidden_size'],
        'time_step':    info['time_step'],
        'eps_min':      info['eps_min'],
        'eps_max':      info['eps_max'],
        'p_norm':       info['p_norm'],
        'max_splits':   info['max_splits'],
        'N_samples':    info['N_samples'],
        'zs_better':    summary.get('zs_better',   0),
        'both_fail':    summary.get('both_fail',   0),
        'pq_all_pass':  summary.get('pq_all_pass', 0),
    }

    for key in TIMING_KEYS:
        entry = timing.get(key, {})
        row[f'timing_{key}_total_sec'] = entry.get('total_sec', 0)
        row[f'timing_{key}_count']     = entry.get('count', 0)
        row[f'timing_{key}_avg_ms']    = entry.get('avg_ms', 0)

    _pq_keys = ['get_y', 'get_hfc', 'get_hig', 'get_c', 'get_hoc', 'get_Wa_b_hidden']
    _pq_total = sum(timing.get(k, {}).get('total_sec', 0) for k in _pq_keys)
    _get_y_count = timing.get('get_y', {}).get('count', 0)
    _n_runs = _get_y_count / info['time_step'] if _get_y_count > 0 else None
    row['popqorn_avg_sec'] = round(_pq_total / _n_runs, 6) if _n_runs else None

    records     = data.get('sample_records', [])
    pq_margins  = [r['pq_margin'] for r in records if r.get('pq_margin') is not None]
    zs_margins  = [r['zs_margin'] for r in records if r.get('zs_margin') is not None]
    gains       = [r['gain']      for r in records if r.get('gain')      is not None]
    row['pq_margin'] = round(sum(pq_margins) / len(pq_margins), 6) if pq_margins else None
    row['zs_margin'] = round(sum(zs_margins) / len(zs_margins), 6) if zs_margins else None
    row['gain']      = round(sum(gains)      / len(gains),      6) if gains      else None

    return row


def main(results_dir='./lstm/evr_results', output_excel='evr_lstm_summary.xlsx'):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f'Directory not found: {results_dir}')

    json_files = glob.glob(str(results_dir / '**' / 'evr_lstm_*.json'), recursive=True)
    if not json_files:
        json_files = glob.glob(str(results_dir / 'evr_lstm_*.json'))
    if not json_files:
        raise FileNotFoundError(f'No evr_lstm_*.json files found in {results_dir}')

    print(f'找到 {len(json_files)} 個 JSON 檔案')

    rows = []
    for i, path in enumerate(json_files, 1):
        try:
            print(f'處理 [{i}/{len(json_files)}]: {Path(path).name}', end='')
            rows.append(parse_json_file(path))
            print(' OK')
        except Exception as e:
            print(f' FAIL: {e}')
            import traceback; traceback.print_exc()

    if not rows:
        raise ValueError('No valid data extracted')

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['hidden_size', 'time_step', 'eps_min', 'p_norm'])
    df.to_excel(output_excel, index=False)

    print(f'\n{"="*70}')
    print(f'成功解析 {len(rows)} 個檔案，結果儲存至: {output_excel}')
    print(f'{"="*70}')
    print(f'\n實驗配置:')
    print(f'  hidden_size: {sorted(df["hidden_size"].unique())}')
    print(f'  time_step:   {sorted(df["time_step"].unique())}')
    print(f'  eps_min:     {sorted(df["eps_min"].unique())}')
    print(f'  p_norm:      {sorted(df["p_norm"].unique())}')

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse LSTM EVR JSON results to Excel')
    parser.add_argument('--input-dir', default='./lstm/evr_results',
                        help='directory containing evr_lstm_*.json files')
    parser.add_argument('--output', default='evr_lstm_summary.xlsx',
                        help='output Excel file')
    args = parser.parse_args()
    main(args.input_dir, args.output)
