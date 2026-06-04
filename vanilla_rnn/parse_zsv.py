import json
import glob
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

def parse_final_report_from_logs(logs):
    """從 logs 中提取 FINAL VERIFICATION REPORT 的統計資訊"""
    stats = {}
    
    in_final_report = False
    in_statistics = False
    
    for entry in logs:
        msg = entry['message']
        
        if "FINAL VERIFICATION REPORT" in msg:
            in_final_report = True
            continue
        
        if in_final_report:
            if "Total samples:" in msg:
                parts = msg.split("Total samples:")[1].strip()
                stats['total_samples'] = int(parts)
            
            elif "Verified samples:" in msg:
                parts = msg.split("Verified samples:")[1].strip()
                verified, total = parts.split("/")[0], parts.split("/")[1].split("(")[0].strip()
                stats['zerosplit_verified'] = int(verified)
                
            elif "Failed samples:" in msg:
                parts = msg.split("Failed samples:")[1].strip()
                failed = parts.split("/")[0]
                stats['zerosplit_failed'] = int(failed)
        
        if "STATISTICS SUMMARY" in msg:
            in_statistics = True
            continue
        
        if in_statistics:
            if "Average splits per sample:" in msg:
                val = msg.split("Average splits per sample:")[1].strip()
                stats['avg_splits'] = float(val)
            
            elif "Maximum splits:" in msg:
                val = msg.split("Maximum splits:")[1].strip()
                stats['max_splits_actual'] = int(val)
            
            elif "Total splits across all samples:" in msg:
                val = msg.split("Total splits across all samples:")[1].strip()
                stats['total_splits'] = int(val)
            
            elif "Samples with strict tighten:" in msg:
                val = msg.split("Samples with strict tighten:")[1].strip()
                stats['strict_tighten_count'] = int(val)
            
            elif "Average top-1 gap improvement (fail):" in msg:
                val = msg.split("Average top-1 gap improvement (fail):")[1].strip()
                stats['avg_fail_top1_gap_imp'] = float(val)
            
            elif "Samples with top-1 gap improvement (fail):" in msg:
                parts = msg.split("Samples with top-1 gap improvement (fail):")[1].strip()
                improved = parts.split("/")[0]
                stats['samples_top1_gap_improved_fail'] = int(improved)
            
            elif "Average top-1 lower bound improvement (fail, refined only):" in msg:
                val = msg.split("Average top-1 lower bound improvement (fail, refined only):")[1].strip()
                stats['avg_fail_top1_imp'] = float(val)
            
            elif "Samples with top-1 lower bound improvement (fail, refined only):" in msg:
                parts = msg.split("Samples with top-1 lower bound improvement (fail, refined only):")[1].strip()
                improved = parts.split("/")[0]
                stats['samples_top1_imp_fail'] = int(improved)
            
            elif "Average other upper bound reduction (fail, refined only):" in msg:
                val = msg.split("Average other upper bound reduction (fail, refined only):")[1].strip()
                stats['avg_fail_other_reduc'] = float(val)
            
            elif "Samples with other upper bound reduction (fail, refined only):" in msg:
                parts = msg.split("Samples with other upper bound reduction (fail, refined only):")[1].strip()
                improved = parts.split("/")[0]
                stats['samples_other_reduc_fail'] = int(improved)
    
    return stats

def extract_first_unsafe_layers(logs):
    """從 final report 中提取每個樣本的 first_unsafe_layer"""
    first_unsafe_layers = []
    first_unsafe_layer_methods = []
    current_sample_id = None
    in_final_report = False
    
    for entry in logs:
        msg = entry['message']
        
        # 檢測是否進入 final report
        if "FINAL VERIFICATION REPORT" in msg:
            in_final_report = True
            continue
        
        # 只在 final report 中處理
        if not in_final_report:
            continue
        
        # 檢測樣本邊界
        if msg.strip().startswith("SAMPLE "):
            try:
                sample_str = msg.strip().split()[1].rstrip(':')
                current_sample_id = int(sample_str)
            except:
                pass

        # 提取 first_unsafe_layer
        if "First unsafe layer selected:" in msg and current_sample_id is not None:
            try:
                content = msg.split("First unsafe layer selected:")[1].strip()
                if content.startswith('['):
                    import ast
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, list) and len(parsed) == 2:
                        layer, neuron = parsed
                        first_unsafe_layers.append((current_sample_id, (layer, neuron)))
                else:
                    layer = int(content)
                    first_unsafe_layers.append((current_sample_id, layer))
            except:
                pass

        # 提取 first_unsafe_layer_method
        if "First unsafe layer method:" in msg and current_sample_id is not None:
            try:
                method = msg.split("First unsafe layer method:")[1].strip()
                first_unsafe_layer_methods.append((current_sample_id, method))
            except:
                pass
    
    return first_unsafe_layers, first_unsafe_layer_methods

def flatten_list(data):
    """遞迴攤平所有嵌套的列表"""
    flat = []
    for item in data:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def compute_group_stats(sample_records, group_size=50):
    groups = []
    total_samples = len(sample_records)
    
    for start_idx in range(0, total_samples, group_size):
        end_idx = min(start_idx + group_size, total_samples)
        group_samples = sample_records[start_idx:end_idx]
        
        # 基本驗證統計
        pq_safe = sum(1 for s in group_samples if s.get('popqorn_verified'))
        zs_safe = sum(1 for s in group_samples if s.get('is_verified'))
        
        # Split 統計
        total_splits_list = [s.get('total_splits', 0) for s in group_samples]
        max_splits = max(total_splits_list) if total_splits_list else 0
        total_splits = sum(total_splits_list)
        
        # Improvement 統計（只針對 PQ 失敗樣本）
        pq_failed = [s for s in group_samples if not s.get('popqorn_verified')]
        pq_failed_count = len(pq_failed)
        
        gap_improvements = []
        top1_improvements = []
        other_reductions = []
        strict_tighten_count = 0
        
        for s in pq_failed:
            imp = s.get('improvements')
            if imp:
                if imp.get('gap_improvement') is not None:
                    gap_improvements.append(imp['gap_improvement'])
                if imp.get('top1_improvement') is not None:
                    top1_improvements.append(imp['top1_improvement'])
                if imp.get('other_reduction') is not None:
                    other_reductions.append(imp['other_reduction'])
                if imp.get('strict_tighten'):
                    strict_tighten_count += 1
        
        # 計算平均值
        avg_fail_top1_gap_imp = sum(gap_improvements) / len(gap_improvements) if gap_improvements else 0
        samples_top1_gap_improved_fail = sum(1 for g in gap_improvements if g > 0)
        
        avg_fail_top1_imp = sum(top1_improvements) / len(top1_improvements) if top1_improvements else 0
        samples_top1_imp_fail = sum(1 for t in top1_improvements if t > 0)
        
        avg_fail_other_reduc = sum(other_reductions) / len(other_reductions) if other_reductions else 0
        samples_other_reduc_fail = sum(1 for o in other_reductions if o > 0)
        
        # First unsafe layers
        first_unsafe_layers = [tuple(s.get('first_unsafe_layer')) if isinstance(s.get('first_unsafe_layer'), list) 
                      else s.get('first_unsafe_layer')
                      for s in group_samples if s.get('first_unsafe_layer') is not None]
        methods = [s.get('first_unsafe_layer_method') for s in group_samples if s.get('first_unsafe_layer_method') is not None]
        
        # Split timesteps 分析
        split_timesteps_list = [s.get('split_timesteps', []) for s in group_samples]
        choosed_timesteps = [ts for ts in split_timesteps_list if ts]
        
        all_timesteps = []
        for ts in split_timesteps_list:
            if isinstance(ts, list):
                all_timesteps.extend(flatten_list(ts))
            else:
                all_timesteps.append(ts)
        
        timestep_frequency = dict(Counter(all_timesteps)) if all_timesteps else {}
        avg_split_timesteps = sum(len(ts) for ts in split_timesteps_list) / len(split_timesteps_list) if split_timesteps_list else 0
        
        # SHAP values
        shap_values_list = [s.get('shap_values') for s in group_samples if s.get('shap_values') is not None]
        
        # PQ failed -> ZS success
        pq_failed_zs_success = [s for s in group_samples if not s.get('popqorn_verified') and s.get('is_verified')]
        pq_failed_zs_success_timesteps = [s.get('split_timesteps', []) for s in pq_failed_zs_success]

        # 新增：Split times 統計
        group_split_times = []
        for s in group_samples:
            split_times = s.get('split_times', [])
            group_split_times.extend(split_times)
        
        # 計算每個 depth 的平均時間
        # 按 (depth, sample_id) 分組
        depth_sample_times = defaultdict(lambda: defaultdict(float))
        for item in group_split_times:
            if len(item) == 3:
                depth, elapsed, sample_id = item
            else:
                depth, elapsed = item
                sample_id = 0
            depth_sample_times[depth][sample_id] += elapsed
        
        group_depth_times = {}
        for depth, sample_dict in sorted(depth_sample_times.items()):
            total_time = sum(sample_dict.values())
            num_samples = len(sample_dict)
            group_depth_times[depth] = total_time / num_samples
        
        group_avg_split_time = (
            sum(item[1] for item in group_split_times) / len(group_split_times)
            if group_split_times else 0
        )
        
        groups.append({
            'group_id': f"{start_idx}-{end_idx-1}",
            'group_size': len(group_samples),
            'popqorn_safe': pq_safe,
            'zerosplit_safe': zs_safe,
            'verify_imp': zs_safe - pq_safe,
            'max_splits_actual': max_splits,
            'total_splits': total_splits,
            'strict_tighten_count': strict_tighten_count,
            'avg_fail_top1_gap_imp': avg_fail_top1_gap_imp,
            'samples_top1_gap_improved_fail': samples_top1_gap_improved_fail,
            'avg_fail_top1_imp': avg_fail_top1_imp,
            'samples_top1_imp_fail': samples_top1_imp_fail,
            'avg_fail_other_reduc': avg_fail_other_reduc,
            'samples_other_reduc_fail': samples_other_reduc_fail,
            'first_unsafe_layers': str(first_unsafe_layers),
            'methods': str(methods),
            'choosed_timesteps': str(choosed_timesteps),
            'shap_values': str(shap_values_list) if shap_values_list else '',
            'avg_split_timesteps': avg_split_timesteps,
            'timestep_frequency': str(timestep_frequency),
            'pq_failed_zs_success_timesteps': str(pq_failed_zs_success_timesteps),
            'pq_failed_zs_success_count': len(pq_failed_zs_success),

            # 新增：Split time analysis
            'avg_split_time': group_avg_split_time,
            'avg_time_per_depth': str(group_depth_times),
            'total_split_time': sum(item[1] for item in group_split_times) if group_split_times else 0,
        })
    
    return groups

def parse_json_file(json_path, group_size=50):
    """解析單個 JSON 檔案，返回單行資料"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exp_info = data['experiment_info']
    ver_results = data['verification_results']
    logs = data['all_logs']
    sample_records = data.get('sample_records', [])
    
    # 從 final report 提取統計資訊
    final_stats = parse_final_report_from_logs(logs)
    
    # 提取 first_unsafe_layers 和 methods
    first_unsafe_layers_list, first_unsafe_layer_methods_list = extract_first_unsafe_layers(logs)
    
    # POPQORN 原先成功數
    popqorn_safe = ver_results.get('popqorn_safe_count', 0)
    
    # ZeroSplit 最終成功數
    zerosplit_safe = ver_results.get('zerosplit_safe_count', final_stats.get('zerosplit_verified', 0))
    
    # 總樣本數
    total_samples = ver_results.get('total_samples', final_stats.get('total_samples', 0))

    # NEW: extract split_timesteps from all samples
    split_timesteps_list = [
        r.get('split_timesteps', []) for r in sample_records
    ]

    # Extrac split_timesteps only from POPQORN failed -> ZeroSplit succeeded samples
    pq_failed_zs_success_timesteps = [
        r.get('split_timesteps', []) 
        for r in sample_records 
        if r.get('pq_failed_zs_success', False)
    ]

    shap_vals_list = [
        r.get('shap_values', None) for r in sample_records 
        if r.get('shap_values') is not None
    ]

    # splits_timestep distribution
    all_timesteps_used = []
    for timesteps in split_timesteps_list:
        # 修改：使用 flatten_list 確保沒有嵌套列表
        if isinstance(timesteps, list):
            all_timesteps_used.extend(flatten_list(timesteps))
        else:
            all_timesteps_used.append(timesteps)

    # Compute the frequency of each timestep being used for splitting
    timestep_frequency = {}
    if all_timesteps_used:
        from collections import Counter
        timestep_counts = Counter(all_timesteps_used)
        timestep_frequency = dict(timestep_counts)

    # Compute the average number of splits per sample
    avg_split_timesteps = (
        sum(len(ts) for ts in split_timesteps_list) / len(split_timesteps_list)
        if split_timesteps_list else 0
    )

    # 新增：提取所有樣本的 split_times
    all_split_times = []
    for record in sample_records:
        split_times = record.get('split_times', [])
        all_split_times.extend(split_times)

    # 按 (depth, sample_id) 分組，計算每個 sample 在該 depth 的總時間
    depth_sample_times = defaultdict(lambda: defaultdict(float))
    sample_max_depth = {}  # 記錄每個 sample 的最大深度
    for item in all_split_times:
        if len(item) == 3:
            depth, elapsed, sample_id = item
        else:
            depth, elapsed = item
            sample_id = 0  # backward compatibility
        depth_sample_times[depth][sample_id] += elapsed

        # 更新 sample 的最大深度
        if sample_id not in sample_max_depth:
            sample_max_depth[sample_id] = depth
        else:
            sample_max_depth[sample_id] = max(sample_max_depth[sample_id], depth)
    
    # 計算每個 depth「到達」該深度的 sample 數
    # 到達 depth d = sample 的 max_depth >= d
    samples_reaching_depth = defaultdict(int)
    for sample_id, max_d in sample_max_depth.items():
        for d in range(1, max_d + 1):
            samples_reaching_depth[d] += 1

    # per-sample 平均：sum(在該 depth 終止的時間) / 到達該 depth 的 sample 數
    avg_time_per_depth = {}
    num_samples_per_depth = {}
    for depth in sorted(depth_sample_times.keys()):
        total_time = sum(depth_sample_times[depth].values())
        num_samples = samples_reaching_depth[depth]
        avg_time_per_depth[depth] = total_time / num_samples if num_samples > 0 else 0
        num_samples_per_depth[depth] = num_samples

    avg_split_time = (
        sum(item[1] for item in all_split_times) / len(all_split_times)
        if all_split_times else 0
    )

    row = {
        'file_name': Path(json_path).stem,
        'timestamp': exp_info.get('timestamp', ''),
        'mode': exp_info.get('mode', ''),
        'hidden_size': exp_info['hidden_size'],
        'time_step': exp_info['time_step'],
        'activation': exp_info['activation'],
        'eps': exp_info['eps'],
        'p_norm': exp_info['p_norm'],
        'N_samples': total_samples,
        'max_splits_config': exp_info['max_splits'],
        
        # POPQORN 結果
        'popqorn_safe': popqorn_safe,
        
        # ZeroSplit 結果
        'zerosplit_safe': zerosplit_safe,
        
        # 改善
        'verify_imp': zerosplit_safe - popqorn_safe,
        
        # Split 統計
        'max_splits_actual': final_stats.get('max_splits_actual', 0),
        'total_splits': final_stats.get('total_splits', 0),
        
        # Gap improvement
        'strict_tighten_count': final_stats.get('strict_tighten_count', 0),
        'avg_fail_top1_gap_imp': final_stats.get('avg_fail_top1_gap_imp', 0),
        'samples_top1_gap_improved_fail': final_stats.get('samples_top1_gap_improved_fail', 0),
        
        # Top-1 improvement
        'avg_fail_top1_imp': final_stats.get('avg_fail_top1_imp', 0),
        'samples_top1_imp_fail': final_stats.get('samples_top1_imp_fail', 0),
        
        # Other reduction
        'avg_fail_other_reduc': final_stats.get('avg_fail_other_reduc', 0),
        'samples_other_reduc_fail': final_stats.get('samples_other_reduc_fail', 0),
        
        # First unsafe layers
        'first_unsafe_layers': str([layer for _, layer in first_unsafe_layers_list]),
        'methods': str([method for _, method in first_unsafe_layer_methods_list]),

        # Split timesteps analysis
        'choosed_timesteps': str([ts for ts in split_timesteps_list if ts]),  # 只記錄非空的
        'shap_values': str(shap_vals_list) if shap_vals_list else '',
        'avg_split_timesteps': avg_split_timesteps,
        'timestep_frequency': str(timestep_frequency),  # {1: 5, 2: 8, 3: 12, ...}
        'pq_failed_zs_success_timesteps': str(pq_failed_zs_success_timesteps), # Fail to success timesteps
        'pq_failed_zs_success_count': len(pq_failed_zs_success_timesteps),

        # 新增：Split time analysis
        'avg_split_time': avg_split_time,
        'avg_time_per_depth': str(avg_time_per_depth),
        'num_samples_per_depth': str(num_samples_per_depth),
        'total_split_time': sum(item[1] for item in all_split_times) if all_split_times else 0,
    }

    group_stats = compute_group_stats(sample_records, group_size)
    group_rows = []
    for g in group_stats:
        group_row = {
            'file_name': Path(json_path).stem,
            'group_id': g['group_id'],
            'hidden_size': exp_info['hidden_size'],
            'time_step': exp_info['time_step'],
            'activation': exp_info['activation'],
            'eps': exp_info['eps'],
            'p_norm': exp_info['p_norm'],
            'N_samples': g['group_size'],
            'popqorn_safe': g['popqorn_safe'],
            'zerosplit_safe': g['zerosplit_safe'],
            'verify_imp': g['verify_imp'],
            'max_splits_actual': g['max_splits_actual'],
            'total_splits': g['total_splits'],
            'strict_tighten_count': g['strict_tighten_count'],
            'avg_fail_top1_gap_imp': g['avg_fail_top1_gap_imp'],
            'samples_top1_gap_improved_fail': g['samples_top1_gap_improved_fail'],
            'avg_fail_top1_imp': g['avg_fail_top1_imp'],
            'samples_top1_imp_fail': g['samples_top1_imp_fail'],
            'avg_fail_other_reduc': g['avg_fail_other_reduc'],
            'samples_other_reduc_fail': g['samples_other_reduc_fail'],
            'first_unsafe_layers': g['first_unsafe_layers'],
            'methods': g['methods'],
            'choosed_timesteps': g['choosed_timesteps'],
            'shap_values': g['shap_values'],
            'avg_split_timesteps': g['avg_split_timesteps'],
            'timestep_frequency': g['timestep_frequency'],
            'pq_failed_zs_success_timesteps': g['pq_failed_zs_success_timesteps'],
            'pq_failed_zs_success_count': g['pq_failed_zs_success_count'],

            # 新增：Split time analysis
            'avg_split_time': g['avg_split_time'],
            'avg_time_per_depth': g['avg_time_per_depth'],
            'total_split_time': g['total_split_time'],
        }
        group_rows.append(group_row)
    
    return row, group_rows

def main(results_dir='./verification_results', output_excel='zerosplit_summary.xlsx', group_size=50):
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")
    
    json_pattern = str(results_dir / "session_*/zerosplit_*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found matching pattern: {json_pattern}")
    
    print(f"找到 {len(json_files)} 個 JSON 檔案")

    overall_rows = []
    grouped_rows = []
    
    # all_rows = []
    for i, json_file in enumerate(json_files, 1):
        try:
            print(f"處理 [{i}/{len(json_files)}]: {Path(json_file).name}", end='')
            row, group_rows = parse_json_file(json_file, group_size)
            overall_rows.append(row)
            grouped_rows.extend(group_rows)
            print(f" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not overall_rows:
        raise ValueError("No valid data extracted from JSON files")
    
    df = pd.DataFrame(overall_rows)
    df_grouped = pd.DataFrame(grouped_rows)
    # 排序
    df = df.sort_values(by=['hidden_size', 'time_step', 'eps', 'p_norm'])
    df_grouped = df_grouped.sort_values(by=['file_name', 'group_id'])

    # 儲存 Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Overall', index=False)
        df_grouped.to_excel(writer, sheet_name='Grouped', index=False)
    
    print(f"\n{'='*80}")
    print(f"成功解析 {len(json_files)} 個 JSON 檔案")
    print(f"結果已儲存至: {output_excel}")
    print(f"  - Sheet 'Overall': 整體統計（所有欄位）")
    print(f"  - Sheet 'Grouped': 每 {group_size} 樣本分組統計")
    print(f"{'='*80}")
    
    # 統計摘要
    print(f"\n實驗配置摘要:")
    print(f"  Hidden sizes: {sorted(df['hidden_size'].unique())}")
    print(f"  Time steps: {sorted(df['time_step'].unique())}")
    print(f"  Epsilon values: {sorted(df['eps'].unique())}")
    print(f"  P-norms: {sorted(df['p_norm'].unique())}")

    return df, df_grouped

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse ZeroSplit verification JSON results to Excel')
    parser.add_argument('--output', default='zerosplit_results.xlsx',
                        help='Output Excel file name')
    parser.add_argument('--group-size', type=int, default=50)
    args = parser.parse_args()
    results_dir = "./verification_results"
    output_dir = "./verification_results/zerosplit_results.xlsx"
    df = main(results_dir, output_dir, args.group_size)