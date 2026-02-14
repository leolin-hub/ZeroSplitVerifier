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
if __name__ == "__main__":
    import os
    import time
    import itertools
    import pandas as pd
    from datetime import datetime
    from zerosplit_verifier import ZeroSplitVerifier
    from utils.sample_cifar10 import sample_cifar10_data

    # === Config ===
    CONFIG = {
        'timesteps': [12],
        'eps_values': [0.03],
        'activations': ['relu'],
        'hidden_size': 64,
        'N': 500,
        'p': 2,
        'max_splits': {
            8: 8,
            12: 12,
        },
        'base_work_dir': 'C:/Users/zxczx/models/cifar10_classifier/',
        'data_dir': './data/cifar-10-batches-py/',
    }

    device = torch.device('cpu')
    results = []

    combinations = list(itertools.product(
        CONFIG['timesteps'],
        CONFIG['eps_values'],
        CONFIG['activations']
    ))

    print(f"\n{'='*60}")
    print(f"SHAP Timing Benchmark")
    print(f"{'='*60}")
    print(f"Samples per config: {CONFIG['N']}")
    print(f"Total configs: {len(combinations)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for ts, eps, act in combinations:
        work_dir = f"{CONFIG['base_work_dir']}rnn_{ts}_{CONFIG['hidden_size']}_{act}/"
        input_size = int(32 * 32 * 3 / ts)
        
        print(f"Config: ts={ts}, eps={eps}, act={act}", end=" ... ")

        verifier = ZeroSplitVerifier(
            input_size, CONFIG['hidden_size'], 10, ts, act, max_splits=CONFIG['max_splits'][ts]
        )
        model_file = os.path.join(work_dir, "rnn")
        
        try:
            verifier.load_state_dict(torch.load(model_file, map_location='cpu'))
        except FileNotFoundError:
            print("SKIP (model not found)")
            continue
            
        verifier.to(device)
        verifier.extractWeight(clear_original_model=False)

        X, y, top1 = sample_cifar10_data(
            N=CONFIG['N'], time_step=ts, device=device,
            data_dir=CONFIG['data_dir'], train=False, rnn=verifier
        )

        sample_times = []
        for i in range(CONFIG['N']):
            X_i = X[i:i+1]
            top1_i = top1[i:i+1]
            
            start = time.time()
            _, _ = compute_shap_ranking_once(verifier, X_i, top1_i, eps, CONFIG['p'])
            elapsed = time.time() - start
            sample_times.append(elapsed)

        avg_time = sum(sample_times) / len(sample_times)
        std_time = (sum((t - avg_time)**2 for t in sample_times) / len(sample_times)) ** 0.5
        
        results.append({
            'timestep': ts,
            'eps': eps,
            'activation': act,
            'N': CONFIG['N'],
            'avg_time': round(avg_time, 4),
            'std_time': round(std_time, 4),
            'min_time': round(min(sample_times), 4),
            'max_time': round(max(sample_times), 4),
        })
        
        print(f"avg={avg_time:.4f}s, std={std_time:.4f}s")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    output_file = f"shap_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nSaved to: {output_file}")