import torch
import time
import argparse
import os
import sys

import json
import csv
from loguru import logger
from datetime import datetime

from bound_vanilla_rnn import RNN
from zerosplit_verifier import ZeroSplitVerifier
from utils.sample_data import sample_mnist_data
from utils.sample_stock_data import prepare_stock_tensors_split

def setup_performance_logging(hidden_size, activation):
    """設置performance測試的資料夾結構，按hidden_size分類"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 在當前目錄(vanilla_rnn)下建立experiment_results
    current_dir = os.path.dirname(os.path.abspath(__file__))  # vanilla_rnn目錄
    
    # 按hidden_size建立分類資料夾
    hidden_size_dir = os.path.join(current_dir, "experiment_results", f"hidden_{hidden_size}_{activation}")
    session_dir = os.path.join(hidden_size_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir, hidden_size_dir

def setup_logging(session_dir):
    """設置loguru日誌記錄(修改版)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(session_dir, f"experiment_log_{timestamp}.txt")

    # 移除默認handler
    logger.remove()
    
    # 添加控制台輸出
    logger.add(
        sys.stderr,
        format="{message}",
        level="INFO"
    )
    
    # 添加文件輸出
    logger.add(
        log_file,
        format="{time:HH:mm:ss} - {message}",
        level="INFO",
        encoding="utf-8"
    )
    
    return log_file

def save_results(all_results, session_dir, args):
    """保存實驗結果(修改版)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 建立檔案名稱，包含配置資訊
    base_name = f"comparison_h{args.hidden_size}_{args.activation}_p{args.p}_{timestamp}"
    
    # 1. 保存詳細結果為JSON
    json_file = os.path.join(session_dir, f"{base_name}.json")
    
    # 準備保存的數據
    save_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'session_dir': session_dir,
            'hidden_size': args.hidden_size,
            'activation': args.activation,
            'p_norm': args.p,
            'model_base_dir': args.model_base_dir,
            'total_experiments': len(all_results),
            'experiment_type': 'performance_comparison'
        },
        'results': []
    }
    
    # 轉換tensor為可序列化的格式
    for result in all_results:
        serializable_result = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                serializable_result[key] = {
                    'tensor_data': value.tolist(),
                    'tensor_shape': list(value.shape),
                    'tensor_dtype': str(value.dtype)
                }
            else:
                serializable_result[key] = value
        save_data['results'].append(serializable_result)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 2. 保存統計摘要為CSV
    csv_file = os.path.join(session_dir, f"{base_name}.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 寫入標題
        writer.writerow([
            'Dataset', 'Time_Step', 'N_Samples', 'Epsilon', 
            'POPQORN_Success', 'ZeroSplit_Success',
            'POPQORN_Time', 'ZeroSplit_Time', 'Speed_Ratio',
            'Initial_Time', 'Split_Time', 'Total_Splits'
        ])
        
        # 寫入數據
        for result in all_results:
            speed_ratio = result['popqorn_time'] / result['zerosplit_time'] if result['zerosplit_time'] > 0 else float('inf')
            
            timing_data = result.get('timing_data', {})
            
            writer.writerow([
                result['dataset'],
                result['time_step'],
                result['N'],
                result['eps'],
                result['popqorn_success'],
                result['zerosplit_success'],
                f"{result['popqorn_time']:.4f}",
                f"{result['zerosplit_time']:.4f}",
                f"{speed_ratio:.2f}",
                f"{timing_data.get('initial_time', 0):.4f}",
                f"{timing_data.get('split_time', 0):.4f}",
                timing_data.get('total_splits', 0)
            ])
    
    # 3. 保存TXT摘要報告
    txt_file = os.path.join(session_dir, f"{base_name}_summary.txt")
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Performance Comparison Report\n")
        f.write(f"{'='*80}\n")
        f.write(f"Session: {os.path.basename(session_dir)}\n")
        f.write(f"Timestamp: {save_data['experiment_info']['timestamp']}\n")
        f.write(f"Configuration: hidden_size={args.hidden_size}, activation={args.activation}, p_norm={args.p}\n")
        f.write(f"Total experiments: {len(all_results)}\n\n")
        
        # 成功率統計
        successful_results = [r for r in all_results if r['popqorn_success'] and r['zerosplit_success']]
        f.write(f"Success Statistics:\n")
        f.write(f"{'-'*40}\n")
        f.write(f"Total experiments: {len(all_results)}\n")
        f.write(f"Both successful: {len(successful_results)}\n")
        f.write(f"Success rate: {len(successful_results)/len(all_results)*100:.1f}%\n\n")
        
        if successful_results:
            # 按dataset分組統計
            for dataset in ['stock', 'mnist']:
                dataset_results = [r for r in successful_results if r['dataset'] == dataset]
                if not dataset_results:
                    continue
                    
                f.write(f"{dataset.upper()} Dataset Performance:\n")
                f.write(f"{'-'*40}\n")
                
                avg_popqorn = sum(r['popqorn_time'] for r in dataset_results) / len(dataset_results)
                avg_zerosplit = sum(r['zerosplit_time'] for r in dataset_results) / len(dataset_results)
                avg_speedup = avg_popqorn / avg_zerosplit if avg_zerosplit > 0 else float('inf')
                
                f.write(f"  Experiments: {len(dataset_results)}\n")
                f.write(f"  Avg POPQORN time: {avg_popqorn:.4f}s\n")
                f.write(f"  Avg ZeroSplit time: {avg_zerosplit:.4f}s\n")
                f.write(f"  Avg speedup: {avg_speedup:.2f}x\n")
                
                # Split統計
                total_splits = sum(r.get('timing_data', {}).get('total_splits', 0) for r in dataset_results)
                f.write(f"  Total splits: {total_splits}\n")
                f.write(f"  Avg splits per experiment: {total_splits/len(dataset_results):.1f}\n\n")
    
    return json_file, csv_file, txt_file

def build_model_path(base_dir, dataset, time_step, hidden_size, activation):
    """根據資料集類型和模型參數動態構建模型路徑"""
    if dataset == 'stock':
        model_name = f"stock_rnn_{time_step}_{hidden_size}_{activation}"
        model_path = os.path.join(base_dir, "stock_classifier", model_name, "rnn")
    else:  # mnist
        model_name = f"rnn_{time_step}_{hidden_size}_{activation}"
        model_path = os.path.join(base_dir, "mnist_classifier", model_name, "rnn")
    
    return model_path

def load_model(model_class, model_path, input_size, hidden_size, output_size, time_step, activation, device):
    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = model_class(input_size, hidden_size, output_size, time_step, activation)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.extractWeight(clear_original_model=False)
    return model

def compute_popqorn_bounds(model, X, eps, p, Eps_idx=None):
    start_time = time.time()
    try:
        yL, yU = model.getLastLayerBound(eps, p, X=X, clearIntermediateVariables=True, Eps_idx=Eps_idx)
        compute_time = time.time() - start_time
        bounds_width = (yU - yL)
        success = True
    except Exception as e:
        compute_time = time.time() - start_time
        bounds_width = float('inf')
        success = False
        yL, yU = None, None
    return bounds_width, compute_time, success

def compute_zerosplit_bounds(model, X, eps, max_splits=3, merge_results=False):
    start_time = time.time()

    timing_data = {
        'initial_time': 0,
        'split_time': 0,
        'total_splits': 0,
        'total_time': 0
    }

    try:
        
        # 1. 初始驗證時間
        initial_start = time.time()
        initial_verified, top1_class, _, _ = model.verify_robustness(X, eps)
        timing_data['initial_time'] = time.time() - initial_start
        
        if initial_verified:
            # 無需split
            timing_data['total_time'] = time.time() - start_time
            yL_out, yU_out = model.computeLast2sideBound(
                eps, p=2, v=model.time_step+1, X=X,
                Eps_idx=torch.arange(1, model.time_step+1)
            )
            bounds_width = (yU_out - yL_out)
            return bounds_width, timing_data['total_time'], True, timing_data
        
        # 2. 執行split驗證時間
        split_start = time.time()
        split_verified = model._recursive_split_verify(
            X, eps, top1_class, split_count=0, max_splits=max_splits,
            start_timestep=1, refine_preh=None
        )
        timing_data['split_time'] = time.time() - split_start
        timing_data['total_splits'] = getattr(model, 'split_count', 0)
        timing_data['total_time'] = time.time() - start_time
        
        return bounds_width, timing_data['total_time'], split_verified, timing_data
    except Exception as e:
        timing_data['total_time'] = time.time() - start_time
        bounds_width = float('inf')
        return bounds_width, timing_data['total_time'], False, timing_data

def run_single_experiment(config, base_args):
    device = torch.device("cuda" if base_args.cuda and torch.cuda.is_available() else "cpu")

    logger.info(f"\n{'='*80}")
    logger.info(f"Timing Analysis Experiment")
    logger.info(f"Dataset: {config['dataset']} | Time_step: {config['time_step']} | N: {config['N']}")
    logger.info(f"Epsilon values: {config['eps_list']}")
    logger.info(f"{'='*80}")

    if config['dataset'] == 'stock':
        input_size = 1
        output_size = 3
        data_path = base_args.stock_data_path
    else:
        input_size = int(28*28 / config['time_step'])  
        output_size = 10
        data_path = base_args.mnist_data_path

    # 動態構建模型路徑
    model_path = build_model_path(
        base_args.model_base_dir, 
        config['dataset'], 
        config['time_step'], 
        base_args.hidden_size, 
        base_args.activation
    )
    
    logger.info(f"Loading model from: {model_path}")

    try:
        # Load models
        popqorn_model = load_model(RNN, model_path, input_size, base_args.hidden_size, 
                                  output_size, config['time_step'], base_args.activation, device)

        zerosplit_model = load_model(ZeroSplitVerifier, model_path, input_size, base_args.hidden_size,
                                    output_size, config['time_step'], base_args.activation, device)
        zerosplit_model.debug = False
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return []

    # Load data
    if config['dataset'] == 'stock':
        X_train, _, _, _, _, _ = prepare_stock_tensors_split(
            data_path, config['time_step'], device=device
        )
        X = X_train[:config['N']]
    else:
        X, _, _ = sample_mnist_data(config['N'], config['time_step'], device, num_labels=output_size, 
                                  data_dir=data_path, train=False, shuffle=False, 
                                  rnn=popqorn_model, x=None, y=None)
        
    experiment_results = []

    for eps in config['eps_list']:
        logger.info(f"\nTesting epsilon = {eps}")
        logger.info(f"{'─'*50}")
        
        # POPQORN
        logger.info(f"Running POPQORN...")
        popqorn_model.clear_intermediate_variables()
        width_pop, time_pop, success_pop = compute_popqorn_bounds(
            popqorn_model, X, eps, base_args.p)
        
        logger.info(f"POPQORN: time={time_pop:.4f}s, success={success_pop}")
        
        # ZeroSplit時間分析
        logger.info(f"Running ZeroSplit (Timing Analysis)...")
        zerosplit_model.clear_intermediate_variables()
        width_zero, time_zero, success_zero, timing_data = compute_zerosplit_bounds(
            zerosplit_model, X, eps, max_splits=127, merge_results=False)
        
        # 時間分析日誌
        logger.info(f"ZeroSplit: time={time_zero:.4f}s, success={success_zero}")
        logger.info(f"  Initial verification: {timing_data['initial_time']:.4f}s")
        logger.info(f"  Split operations: {timing_data['split_time']:.4f}s")
        logger.info(f"  Total splits: {timing_data['total_splits']}")
        
        if timing_data['total_splits'] > 0:
            avg_split_time = timing_data['split_time'] / timing_data['total_splits']
            logger.info(f"  Average time per split: {avg_split_time:.4f}s")
        
        # 效率比較
        if success_pop and success_zero:
            # improvement = (width_pop - width_zero)
            # zerosplit_better = (improvement >= 0).all()
            speedup = time_pop / time_zero if time_zero > 0 else float('inf')
            
            logger.info(f"Comparison results:")
            # logger.info(f"  ZeroSplit better bounds: {zerosplit_better}")
            logger.info(f"  Speed ratio: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
        
        # 保存結果
        result = {
            'dataset': config['dataset'],
            'time_step': config['time_step'],
            'N': config['N'],
            'eps': eps,
            'popqorn_width': width_pop,
            'zerosplit_width': width_zero,
            'popqorn_time': time_pop,
            'zerosplit_time': time_zero,
            'popqorn_success': success_pop,
            'zerosplit_success': success_zero,
            'timing_data': timing_data
        }
        experiment_results.append(result)

    return experiment_results

def generate_experiment_configs():
    # Experiment configurations
    N_values = [10, 50, 100]

    stock_time_steps = [10]
    stock_eps_configs = [
        [0.5, 1.0],
        [1.5, 2.0],
        [3.0, 4.0, 5.0]
    ]

    mnist_time_steps = [2, 4, 7]
    mnist_eps_configs = [
        [0.01, 0.05, 0.1],
        [0.1, 0.2, 0.3],
        [1.0, 1.5, 2.0]
    ]

    configs = []

    # Stock experiments
    for time_step in stock_time_steps:
        for N in N_values:
            for eps_list in stock_eps_configs:
                configs.append({
                    'dataset': 'stock',
                    'time_step': time_step,
                    'N': N,
                    'eps_list': eps_list
                })

    # MNIST experiments
    for time_step in mnist_time_steps:
        for N in N_values:
            for eps_list in mnist_eps_configs:
                configs.append({
                    'dataset': 'mnist',
                    'time_step': time_step,
                    'N': N,
                    'eps_list': eps_list
                })

    return configs

def summarize_results(all_results):
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")

    # Group by dataset and time_step
    for dataset in ['stock', 'mnist']:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if not dataset_results:
            continue
            
        logger.info(f"\n{dataset.upper()} Dataset:")
        
        # Group by time_step
        time_steps = sorted(set(r['time_step'] for r in dataset_results))
        for ts in time_steps:
            ts_results = [r for r in dataset_results if r['time_step'] == ts]
            successful_results = [r for r in ts_results if r['popqorn_success'] and r['zerosplit_success']]
            
            if successful_results:
                total_tests = len(successful_results)
                # zerosplit_wins = sum(1 for r in successful_results if (r['improvement'] >= 0).all())
                avg_time_popqorn = sum(r['popqorn_time'] for r in successful_results) / total_tests
                avg_time_zerosplit = sum(r['zerosplit_time'] for r in successful_results) / total_tests
                
                # logger.info(f"  Time_step {ts}: {zerosplit_wins}/{total_tests} wins ({zerosplit_wins/total_tests*100:.1f}%)")
                logger.info(f"    Avg time - POPQORN: {avg_time_popqorn:.4f}s, ZeroSplit: {avg_time_zerosplit:.4f}s")

def main():
    parser = argparse.ArgumentParser(description='Compare POPQORN vs ZeroSplit')

    # Model Parameters (可調整的)
    parser.add_argument('--hidden-size', default=64, type=int, 
                       help='Hidden layer size (affects model path)')
    parser.add_argument('--activation', default='relu', type=str,
                       help='Activation function (affects model path)')
    parser.add_argument('--p', default=2, type=int)
    parser.add_argument('--cuda', action='store_true')

    # Dataset Parameters
    parser.add_argument('--model-base-dir', default="C:/Users/zxczx/models/", 
                       help='Base directory for model files')
    parser.add_argument('--stock-data-path', default='C:/Users/zxczx/POPQORN/vanilla_rnn/utils/A1_bin.csv')
    parser.add_argument('--mnist-data-path', default='../data/mnist', 
                       help='Path to MNIST data directory')

    # Experiment Control
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run only a subset of experiments for quick testing')
    
    args = parser.parse_args()

    # === 修改的部分：建立按hidden_size分類的資料夾 ===
    session_dir, hidden_size_dir = setup_performance_logging(args.hidden_size, args.activation)
    
    # 設置loguru日誌
    log_file = setup_logging(session_dir)
    
    logger.info(f"Performance Comparison Experiment Started")
    logger.info(f"{'='*80}")
    logger.info(f"Model configuration: hidden_size={args.hidden_size}, activation={args.activation}")
    logger.info(f"Results organized by: {os.path.basename(hidden_size_dir)}")
    logger.info(f"Current session: {os.path.basename(session_dir)}")
    logger.info(f"Log file: {os.path.basename(log_file)}")

    # Generate all experiment configurations
    all_configs = generate_experiment_configs()

    if args.quick_test:
        all_configs = all_configs[:4]
        logger.info(f"Quick test mode: running {len(all_configs)} experiments")
    else:
        logger.info(f"Full experiment mode: running {len(all_configs)} experiments")

    # Run all experiments
    all_results = []
    for i, config in enumerate(all_configs):
        logger.info(f"\nProgress: {i+1}/{len(all_configs)}")
        try:
            experiment_results = run_single_experiment(config, args)
            all_results.extend(experiment_results)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            continue

    # Summarize all results
    summarize_results(all_results)
    
    # === 修改的部分：使用新的save_results函數 ===
    if all_results:
        json_file, csv_file, txt_file = save_results(all_results, session_dir, args)
        logger.info(f"\nResults saved to session: {os.path.basename(session_dir)}")
        logger.info(f"  Detailed results (JSON): {os.path.basename(json_file)}")
        logger.info(f"  Summary results (CSV): {os.path.basename(csv_file)}")
        logger.info(f"  Report summary (TXT): {os.path.basename(txt_file)}")
        logger.info(f"  Log file: {os.path.basename(log_file)}")
    else:
        logger.warning("No results to save!")
    
    logger.info(f"\nExperiment completed at {datetime.now()}")

if __name__ == '__main__':
    main()