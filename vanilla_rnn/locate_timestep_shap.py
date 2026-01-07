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
            split_history = set()

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
    
def compute_shap_ranking_once(verifier, X, top1_class, eps, p, top_k_ratio=0.5, shap_threshold=0.8):
    """
    Compute SHAP ranking once for given input X.

    Args:
        verifier: ZeroSplitVerifier instance
        X: [1, time_step, input_size]
        top1_class: [1] or scalar label
        top_k_ratio: Select top k ratio of timesteps

    Returns:
        selected_timesteps: [t1, t2, ...] ascending order
        timestep_importance: {t: importance}
    """
    # ============ [實驗控制區: Anchor + SHAP Block] ============
    # 如果想恢復原狀，將此處改為 False 即可
    USE_HYBRID_STRATEGY = False 
    
    # 參數設定 (僅在 True 時生效)
    ANCHOR_RATIO = 0.6  # 預算的 60% 強制給最後面 (例如選5個，這代表 3個給 Anchor)
    BLOCK_SIZE = 2      # 剩下的預算，找到 Peak 後要切成多大的連續區塊 (1代表只切Peak本身)
    # =========================================================
    locator = TimestepSHAPLocator(verifier, background_size=20, eps=eps, p=p)

    timestep_importance = {}

    for t in range(1, verifier.time_step + 1):
        importance = locator._compute_ts_importance(X, locator._generate_background(X), t, top1_class)
        timestep_importance[t] = importance.sum()

    # Rank timesteps by importance
    ranked = sorted(timestep_importance.items(), key=lambda x: x[1], reverse=True)

    # Select top k ratio timesteps
    k = max(1, int(verifier.time_step * top_k_ratio))
    # selected = [t for t, _ in ranked[:k]] # 原本的選法
    selected = []

    # ============ 策略分支 ============
    if USE_HYBRID_STRATEGY:
        # --- Part A: Anchor (保底機制) ---
        # 強制選取最後的 num_anchor 個 steps
        num_anchor = int(k * ANCHOR_RATIO)
        
        # 確保至少留 1 個給 SHAP Block (除非 k 很小)
        if k > 1 and num_anchor == k:
            num_anchor = k - 1
            
        anchor_start = verifier.time_step - num_anchor + 1
        anchor_selection = list(range(anchor_start, verifier.time_step + 1))
        selected.extend(anchor_selection)
        
        # --- Part B: SHAP Block (狙擊機制) ---
        remaining_budget = k - len(selected)
        
        if remaining_budget > 0:
            # 在 "排除 Anchor 區域" 的範圍內，找 SHAP 最高的那個點 (Peak)
            # 範圍: 1 ~ (anchor_start - 1)
            search_range = [t for t in range(1, anchor_start) if t in timestep_importance]
            
            if search_range:
                # 找到最大值的 Time Step
                peak_t = max(search_range, key=lambda t: timestep_importance[t])
                
                # 建立 Block: 從 peak_t 開始往後抓，最多抓 remaining_budget 個
                # 例如 peak_t=12, remaining=2 -> [12, 13]
                # 這裡做一個簡單的 Block 擴展 (也可以改成 peak_t 前後擴展，這裡先寫向後)
                block_selection = []
                for i in range(remaining_budget):
                    t_candidate = peak_t + i
                    if t_candidate < anchor_start: # 不能撞到 Anchor 區
                        block_selection.append(t_candidate)
                
                selected.extend(block_selection)
                logger.info(f" [Hybrid] Anchor: {anchor_selection}, SHAP Peak at {peak_t}, Block: {block_selection}")
            else:
                # 萬一沒有搜尋範圍 (例如 k=time_step)，就全選了
                pass

    else:
        # ============ 原始版本 (Pure Top-K) ============
        ranked = sorted(timestep_importance.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:k]]
        logger.info(f" [Original] Pure SHAP Top-{k}")

    selected = sorted(list(set(selected)))
    logger.info(f" Final Selection (Hybrid={USE_HYBRID_STRATEGY}): {selected}")
    # Method 2: Select timesteps above threshold
    # total_importance = sum(imp for _, imp in ranked)
    # cum_imp = 0
    # selected_by_t = []

    # for t, imp in ranked:
    #     cum_imp += imp
    #     selected_by_t.append(t)
        
    #     if cum_imp / total_importance >= shap_threshold:
    #         break
    
    # if len(selected_by_t) < len(selected):
    #     selected = selected_by_t
    #     logger.info(f"  SHAP selection: threshold method selected {len(selected)} timesteps "
    #                 f"(cumulative importance: {cum_imp/total_importance:.2%})")
    # else:
    #     logger.info(f"  SHAP selection: ratio method selected {len(selected)} timesteps "
    #                 f"(top {top_k_ratio:.0%})")
    # selected.sort()
    # selected_by_t.sort()
    
    return selected, timestep_importance

def select_timestep_from_shap(verifier, selected_timesteps, start_timestep, refine_preh, split_history=None, sample_id=None):
    """
    Select the next timestep to split based on SHAP ranking.

    Args:
        verifier: ZeroSplitVerifier instance
        selected_timesteps: [t1, t2, ...] sorted by importance descending
        start_timestep: int, minimum timestep to consider for splitting
        refine_preh: (l_state, u_state), refinement pre-activation bounds
        split_history: set of timesteps that have been split
        sample_id: for logging

    Returns:
        (timestep, cross_zero_mask) or (None, None) if not found.
    """
    if split_history is None:
        split_history = set()

    logger.info(f"  Sample {sample_id+1}: Selecting timestep from {selected_timesteps}")
    logger.info(f"  start_timestep={start_timestep}, split_history={sorted(list(split_history))}")

    for t in selected_timesteps:
        # Condition 1: timestep >= start_timestep
        if t < start_timestep:
            continue

        if t in split_history:
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

        is_cross_zero = (l_t < 0) & (u_t > 0)

        if is_cross_zero.any():
            num_cross = is_cross_zero.sum().item()
            logger.info(f"  Selected timestep {t} with {num_cross} cross-zero neurons")
            return t, is_cross_zero
        
    logger.info(f"  No valid timestep found for sample {sample_id+1}")
    return None, None  # No valid split found

# ============ 測試程式碼 ============
if __name__ == "__main__":
    import os
    from zerosplit_verifier import ZeroSplitVerifier
    from utils.sample_data import sample_mnist_data

    time_step = 2
    input_size = 392
    output_size = 10
    N = 1
    p = 2
    hidden_size = 16
    activation = 'relu'
    max_splits = 1
    eps = 0.1
    work_dir = "C:/Users/zxczx/models/mnist_classifier/rnn_2_16_relu/"
    
    device = torch.device('cpu')

    verifier = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, 
                                   activation, max_splits=max_splits, debug=False)

    model_file = os.path.join(work_dir, "rnn")
    verifier.load_state_dict(torch.load(model_file, map_location='cpu'))
    verifier.to(device)
    
    X, y, target_label = sample_mnist_data(
        N=N, seq_len=time_step, device=device,
        data_dir='../data/mnist', train=False, shuffle=True, rnn=verifier
    )
    
    verifier.extractWeight(clear_original_model=False)

    is_verified_i, top1_i, yL_out_i, yU_out_i = verifier.verify_robustness(X, eps)