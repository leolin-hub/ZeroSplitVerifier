import torch
import torch.nn as nn
import numpy as np
import shap
from loguru import logger


class TimestepSHAPLocator:
    """
    Use SHAP to locate import timesteps and neurons (timesteps, neurons) in models like RNN.
    """

    def __init__(self, verifier, background_size=20, eps=None, p=2):
        self.verifier = verifier
        self.background_size = background_size
        self.device = next(verifier.parameters()).device
        self.eps = eps
        self.p = p

    def compute_shap_ranking(self, X, top1_class):
        """
        Compute all (timestep, neuron) importance using SHAP.
        Args:
            X: [N, time_step, input_size] - input samples
            top1_class: [N] ground truth labels
        
        Returns:
            ranked: [((t, n), importance), ...] sorted by importance descending
        """
        background = self._generate_background(X)
        ts_neuron_importance = {}

        # Compute SHAP importance for each timestep
        for t in range(1, self.verifier.time_step + 1):
            importance = self._compute_ts_importance(X, background, t, top1_class)

            # Save importance for each neuron at timestep t
            for n in range(self.verifier.num_neurons):
                ts_neuron_importance[(t, n)] = importance[n]

        ranked = sorted(ts_neuron_importance.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def select_next_split(self, ranked, start_timestep, refine_preh, split_history=None):
        """
        Select the next (timestep, neuron) to split based on ranked importance.
        Conditions:
        1. timestep >= start_timestep (ensure no split repeatedly)
        2. pre-activation on this (timestep, neuron) cross zero

        Args:
            ranked: [((t, n), score), ...] sorted by importance descending
            start_timestep: int, the minimum timestep to consider for splitting
            refine_preh: (l_state, u_state), previous refinement pre-activation bounds
        
        Returns:
            (timestep, neuron, cross_zero_mask) or (None, None, None) if not found
        """
        if split_history is None:
            split_history = []

        for (t, n), score in ranked:
            # Condition 1: timestep >= start_timestep
            if t < start_timestep:
                continue

            if (t, n) in split_history:
                continue
            
            # Condition 2: pre-activation cross zero
            if refine_preh is not None:
                l_t = refine_preh[0][t]
                u_t = refine_preh[1][t]
            else:
                l_t = self.verifier.l[t]
                u_t = self.verifier.u[t]
            
            if l_t is None or u_t is None:
                continue

            is_cross_zero = (l_t[:, n] < 0) & (u_t[:, n] > 0)

            if is_cross_zero.any():
                cross_zero = torch.zeros_like(l_t, dtype=torch.bool)
                cross_zero[:, n] = is_cross_zero
                return t, n, cross_zero
            
        return None, None, None  # No valid split found
    
    def _generate_background(self, X):
        # Gausian
        # background = X.repeat(self.background_size, 1, 1)
        # noise = torch.randn_like(background) * 0.1
        # background = background + noise

        N = self.background_size
        background = X.repeat(N, 1, 1)

        if self.eps is None or self.eps == 0:
            return background.detach()
        
        if self.p == 2:
            delta = torch.randn_like(background)
            norms = delta.view(N, -1).norm(p=2, dim=1, keepdim=True)
            delta = delta / norms.view(N, 1, 1)
            r = torch.rand(N, 1, 1, device=self.device) ** (1.0 / (X.shape[1] * X.shape[2]))
            delta = delta * r * self.eps
        elif self.p == float('inf'):
            delta = (torch.rand_like(background) * 2 - 1) * self.eps
        elif self.p == 1:
            delta = torch.randn_like(background)
            abs_delta = torch.abs(delta)
            abs_delta = abs_delta / (abs_delta.sum(dim=(1,2), keepdim=True) + 1e-8)
            signs = torch.sign(delta)
            r = torch.rand(N, 1, 1, device=self.device)
            delta = signs * abs_delta * r * self.eps
        else:
            raise ValueError(f"Unsupported p={self.p}")

        background = background + delta
        return background.detach()
    
    def _compute_ts_importance(self, X, background, timestep, top1_class):

        wrapper = self._create_wrapper_model(timestep, X)

        with torch.no_grad():
            h_bg = self._forward_to_timestep(background, timestep)
            h_X = self._forward_to_timestep(X, timestep)

        # Debug: 檢查wrapper的輸出
        # with torch.no_grad():
        #     test_output = wrapper(h_X)
        #     logger.info(f"Timestep {timestep}: output range = [{test_output.min():.2f}, {test_output.max():.2f}]")
        
        try:
            explainer = shap.GradientExplainer(wrapper, h_bg)
            shap_values = explainer.shap_values(h_X)

            # Debug: 檢查shap_values
            # if isinstance(shap_values, list):
            #     logger.info(f"Timestep {timestep}: shap_values type=list, len={len(shap_values)}")
            #     for i, sv in enumerate(shap_values):
            #         logger.info(f"  Class {i}: shape={sv.shape}, sum={np.abs(sv).sum():.4f}")
            # else:
            #     logger.info(f"Timestep {timestep}: shap_values shape={shap_values.shape}, sum={np.abs(shap_values).sum():.4f}")
            
            importance = np.zeros(self.verifier.num_neurons)

            if isinstance(shap_values, list):
                # shap_values: [N, hidden_size, num_classes]
                for i in range(X.shape[0]):
                    class_idx = top1_class[i].item()
                    importance += np.abs(shap_values[class_idx][i])
            else:
                # shap_values: [N, hidden_size, n_classes]
                for i in range(X.shape[0]):
                    class_idx = top1_class[i].item()
                    # shap_values[i]: [hidden_size, n_classes]
                    # shap_values[i, :, class_idx]: [hidden_size]
                    importance += np.abs(shap_values[i, :, class_idx])

            importance /= X.shape[0]

        except Exception as e:
            print(f"WARNING: SHAP failed for timestep {timestep}: {e}")
            importance = np.zeros(self.verifier.num_neurons)

        return importance
    
    def _forward_to_timestep(self, X, timestep):
        N = X.shape[0]
        h = torch.zeros(1, N, self.verifier.num_neurons, device=self.device)
        
        for t in range(timestep):
            x_t = X[:, t, :]  # [N, input_size]
            
            # 手動計算 RNN step
            z = torch.matmul(x_t, self.verifier.W_ax.t()) + \
                torch.matmul(h.squeeze(0), self.verifier.W_aa.t()) + \
                self.verifier.b_aa + self.verifier.b_ax
            
            if self.verifier.activation == 'tanh':
                h = torch.tanh(z).unsqueeze(0)
            elif self.verifier.activation == 'relu':
                h = torch.relu(z).unsqueeze(0)
            elif self.verifier.activation == 'sigmoid':
                h = torch.sigmoid(z).unsqueeze(0)
        
        return h.squeeze(0)  # [N, hidden_size]
    
    def _create_wrapper_model(self, from_timestep, X):
        verifier = self.verifier
        
        class HiddenToOutput(nn.Module):
            def __init__(self, from_t, X_sample):
                super().__init__()
                self.from_t = from_t
                self.X_sample = X_sample  # [1, time_step, input_size] 或 [N, ...]
                
            def forward(self, h):
                """
                h: [batch_size, hidden_size]
                batch_size 可能是 background_size (如20) 或 test_size (如1)
                """
                batch_size = h.shape[0]
                h_current = h.unsqueeze(0)  # [1, batch_size, hidden_size]
                
                # 重複X以匹配batch size
                if batch_size > self.X_sample.shape[0]:
                    # Background case: 需要重複
                    repeat_factor = batch_size // self.X_sample.shape[0]
                    if batch_size % self.X_sample.shape[0] != 0:
                        repeat_factor += 1
                    X_expanded = self.X_sample.repeat(repeat_factor, 1, 1)[:batch_size]
                else:
                    # Test case: 直接用
                    X_expanded = self.X_sample[:batch_size]
                
                # Forward剩餘timesteps
                for t in range(self.from_t, verifier.time_step):
                    x_t = X_expanded[:, t, :]  # [batch_size, input_size]
                    
                    z = torch.matmul(x_t, verifier.W_ax.t()) + \
                        torch.matmul(h_current.squeeze(0), verifier.W_aa.t()) + \
                        verifier.b_aa + verifier.b_ax
                    
                    if verifier.activation == 'tanh':
                        h_current = torch.tanh(z).unsqueeze(0)
                    elif verifier.activation == 'relu':
                        h_current = torch.relu(z).unsqueeze(0)
                    elif verifier.activation == 'sigmoid':
                        h_current = torch.sigmoid(z).unsqueeze(0)
                
                # 最後到output
                output = torch.matmul(h_current.squeeze(0), verifier.W_fa.t()) + verifier.b_f
                return output
        
        return HiddenToOutput(from_timestep, X)
    
def compute_shap_ranking_once(verifier, X, top1_class, eps, p, top_k_neurons=5):
    """
    Compute SHAP ranking once for given input X, filtering cross-zero neurons only.

    Args:
        verifier: ZeroSplitVerifier instance
        X: [1, time_step, input_size]
        top1_class: [1] or scalar label
        top_k_neurons: int, select top-k most important cross-zero neurons

    Returns:
        selected_neurons: [(t, n, importance), ...] sorted by timestep ascending
    """
    locator = TimestepSHAPLocator(verifier, background_size=20, eps=eps, p=p)

    neuron_importance = []

    for t in range(1, verifier.time_step + 1):
        importance = locator._compute_ts_importance(X, locator._generate_background(X), t, top1_class)

        l_t = verifier.l[t]
        u_t = verifier.u[t]

        if l_t is None or u_t is None:
            continue
        
        for n in range(verifier.num_neurons):
            is_cross_zero = (l_t[:, n] < 0) & (u_t[:, n] > 0)
            if is_cross_zero.any():
                neuron_importance.append((t, n, importance[n]))

    # Sort by importance descending
    neuron_importance.sort(key=lambda x: x[2], reverse=True)

    selected = neuron_importance[:top_k_neurons]

    selected.sort(key=lambda x: x[0])

    logger.info(f"  SHAP selected top-{len(selected)} neurons:")
    for t, n, imp in selected[:5]:
        logger.info(f"    t={t}, n={n+1}, importance={imp:.4f}")
    return selected

def select_timestep_from_shap(verifier, selected_neurons, start_timestep, refine_preh, split_history=None, sample_id=None):
    """
    Select the next (timestep, neuron) to split based on SHAP ranking.

    Args:
        verifier: ZeroSplitVerifier instance
        selected_neurons: [(t, n, importance), ...] sorted by timestep ascending
        start_timestep: int, minimum timestep to consider for splitting
        refine_preh: (l_state, u_state), refinement pre-activation bounds
        split_history: set of (timestep, neuron) tuples that have been split
        sample_id: for logging

    Returns:
        (timestep, neuron_idx, cross_zero_mask) or (None, None, None) if not found.
        cross_zero_mask: [N, hidden_size] bool tensor with only the selected neuron=True
    """
    if split_history is None:
        split_history = []

    logger.info(f"  Sample {sample_id+1}: Selecting from {len(selected_neurons)} ranked neurons")
    logger.info(f"  start_timestep={start_timestep}, split_history={sorted(list(split_history))}")

    for t, n, imp in selected_neurons:
        # Condition 1: timestep >= start_timestep
        if t < start_timestep:
            continue

        if (t, n) in split_history:
            continue
        
        # Condition 2: pre-activation cross zero
        if refine_preh is not None:
            l_t = refine_preh[0][t]
            u_t = refine_preh[1][t]
        else:
            l_t = verifier.l[t]
            u_t = verifier.u[t]
        
        if l_t is None or u_t is None:
            continue

        # Check if this specific neuron crosses zero
        is_cross_zero = (l_t[:, n] < 0) & (u_t[:, n] > 0)

        if is_cross_zero.any():
            # Create mask with only this neuron set to True
            cross_zero_mask = torch.zeros_like(l_t, dtype=torch.bool)
            cross_zero_mask[:, n] = is_cross_zero
            
            logger.info(f"  Selected t={t}, n={n+1}, importance={imp:.4f}")
            return t, n, cross_zero_mask
        
    logger.info(f"  No valid (timestep, neuron) found for sample {sample_id+1}")
    return None, None, None

# ============ 測試程式碼 ============
_DATASET_CFGS = {
    'cifar10': {
        'base_model_dir': 'C:/Users/zxczx/models/cifar10_classifier/',
        'model_prefix':   'rnn',
        'data_dir':       'C:/Users/zxczx/POPQORN/vanilla_rnn/data/cifar-10-batches-py/',
        'input_size_fn':  lambda ts: int(32 * 32 * 3 / ts),
        'num_classes':    10,
    },
    'mnist_seq': {
        'base_model_dir': 'C:/Users/zxczx/models/mnist_seq_classifier/',
        'model_prefix':   'rnn_seq',
        'data_dir':       'C:/Users/zxczx/POPQORN/vanilla_rnn/data/mnist_seq/sequences/',
        'input_size_fn':  lambda ts: 3,
        'num_classes':    10,
    },
}

def _shap_timing_worker(args):
    """Module-level worker — 處理單一 config 的 sample[sample_start:sample_end]。"""
    import os, time, torch, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from zerosplit_verifier import ZeroSplitVerifier
    from utils.sample_cifar10 import sample_cifar10_data
    from utils.sample_seq_mnist import sample_seq_mnist_data

    dataset_name, ts, h, act, eps_values, N, P, sample_start, sample_end = args
    cfg = _DATASET_CFGS[dataset_name]
    device = torch.device('cpu')

    model_file = os.path.join(cfg['base_model_dir'], f"{cfg['model_prefix']}_{ts}_{h}_{act}", 'rnn')
    if not os.path.exists(model_file):
        print(f"[SKIP] {dataset_name} ts={ts} h={h} act={act}: model not found", flush=True)
        return []

    input_size = cfg['input_size_fn'](ts)
    verifier = ZeroSplitVerifier(input_size, h, cfg['num_classes'], ts, act, max_splits=5)
    verifier.load_state_dict(torch.load(model_file, map_location='cpu'))
    verifier.to(device)
    verifier.extractWeight(clear_original_model=False)

    if dataset_name == 'cifar10':
        X, y, top1 = sample_cifar10_data(
            N=N, time_step=ts, device=device,
            data_dir=cfg['data_dir'], train=False, rnn=verifier,
        )
    else:
        X, y, top1 = sample_seq_mnist_data(
            N=N, time_step=ts, device=device,
            data_dir=cfg['data_dir'], train=False, rnn=verifier,
        )

    # 每個 eps 只回傳這個 chunk 的 sample_times，由主程式聚合
    partial = []  # list of (eps, [t0, t1, ...])
    for eps in eps_values:
        sample_times = []
        for i in range(sample_start, sample_end):
            X_i    = X[i:i+1]
            top1_i = top1[i:i+1]
            verifier.verify_robustness(X_i, eps)
            start  = time.time()
            compute_shap_ranking_once(verifier, X_i, top1_i, eps, P)
            sample_times.append(time.time() - start)
        partial.append((eps, sample_times))
        print(f"[{dataset_name}] ts={ts} h={h} act={act} eps={eps:.3f} "
              f"samples[{sample_start}:{sample_end}] done", flush=True)

    return dataset_name, ts, h, act, partial


if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from multiprocessing import Pool
    from collections import defaultdict
    import multiprocessing as mp
    import os, sys

    def initialize_multiprocessing():
        if sys.platform != 'win32':
            try:
                mp.set_start_method('spawn', force=True)
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
            except RuntimeError:
                pass

    initialize_multiprocessing()

    ALL_DATASETS = {
        'cifar10':   {'timesteps': [8, 12, 24, 32],  'hidden_sizes': [16, 32, 64, 128], 'activations': ['relu', 'tanh']},
        'mnist_seq': {'timesteps': [35, 40, 45, 50], 'hidden_sizes': [16, 32, 64, 128], 'activations': ['relu', 'tanh']},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, choices=list(ALL_DATASETS.keys()),
                        default=None,   help='限定單一 dataset (不指定則跑全部)')
    parser.add_argument('--timesteps',  type=int, nargs='+',
                        default=None,   help='限定 timestep，例如 --timesteps 8 12')
    parser.add_argument('--hidden_sizes', type=int, nargs='+',
                        default=None,   help='限定 hidden_size，例如 --hidden_sizes 16 32')
    parser.add_argument('--activations', type=str, nargs='+', choices=['relu', 'tanh'],
                        default=None,   help='限定 activation，例如 --activations relu')
    args = parser.parse_args()

    WORKERS_PER_CONFIG = 4
    N         = 50
    P         = 2
    EPS_VALUES = list(np.round(np.arange(0.005, 0.101, 0.001), 3))

    # 套用 CLI 過濾
    datasets_to_run = {args.dataset: ALL_DATASETS[args.dataset]} if args.dataset else ALL_DATASETS
    configs = [
        (ds, ts, h, act)
        for ds, cfg in datasets_to_run.items()
        for ts  in (args.timesteps   or cfg['timesteps'])
        for h   in (args.hidden_sizes or cfg['hidden_sizes'])
        for act in (args.activations  or cfg['activations'])
        if ts in cfg['timesteps'] and h in cfg['hidden_sizes']
    ]

    chunk = N // WORKERS_PER_CONFIG

    print(f"Total configs : {len(configs)}")
    print(f"EPS values    : {EPS_VALUES[0]} ~ {EPS_VALUES[-1]} ({len(EPS_VALUES)} values)")
    print(f"Started       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows = []
    for idx, (ds, ts, h, act) in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] {ds} ts={ts} h={h} act={act}")

        sub_tasks = [
            (ds, ts, h, act, EPS_VALUES, N, P,
             j * chunk,
             N if j == WORKERS_PER_CONFIG - 1 else (j + 1) * chunk)
            for j in range(WORKERS_PER_CONFIG)
        ]

        with Pool(WORKERS_PER_CONFIG) as pool:
            results_list = pool.map(_shap_timing_worker, sub_tasks)

        # 聚合同一 config 的 4 份 partial times
        aggregated = defaultdict(list)
        for result in results_list:
            if not result:
                continue
            _, _, _, _, partial = result
            for eps, times in partial:
                aggregated[eps].extend(times)

        for eps, times in sorted(aggregated.items()):
            avg_t = sum(times) / len(times)
            std_t = (sum((t - avg_t) ** 2 for t in times) / len(times)) ** 0.5
            all_rows.append({
                'dataset':     ds,
                'time_step':   ts,
                'hidden_size': h,
                'activation':  act,
                'eps':         eps,
                'N':           len(times),
                'avg_time_s':  round(avg_t, 6),
                'std_time_s':  round(std_t, 6),
                'min_time_s':  round(min(times), 6),
                'max_time_s':  round(max(times), 6),
            })
            print(f"  eps={eps:.3f}: avg={avg_t:.4f}s", flush=True)

    if not all_rows:
        print("No results collected.")
    else:
        df = pd.DataFrame(all_rows)
        ds_tag = '_'.join(sorted(set(r['dataset']   for r in all_rows)))
        ts_tag = '-'.join(str(t) for t in sorted(set(r['time_step'] for r in all_rows)))
        out_file = (f"C:/Users/zxczx/POPQORN/vanilla_rnn/"
                    f"shap_timing_{ds_tag}_ts{ts_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(out_file, index=False)
        print(f"\nSaved {len(all_rows)} rows to: {out_file}")