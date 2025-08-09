import types
import torch
from bound_vanilla_rnn import RNN
import torch.nn as nn
import argparse
import os
from loguru import logger
import json
import sys
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

import torch
from torchvision import datasets, transforms
from utils.sample_data import sample_mnist_data
from utils.sample_stock_data import prepare_stock_tensors_split
import get_bound_for_general_activation_function as get_bound

class LoggerCapture:
    """捕獲logger輸出的類"""
    def __init__(self):
        self.captured_logs = []
        
    def capture_handler(self, message):
        """loguru handler來捕獲log訊息"""
        # 簡化版本：只保存message內容和時間戳
        self.captured_logs.append({
            'timestamp': datetime.now().isoformat(),
            'message': str(message).strip()
        })

def setup_result_logging(work_dir, eps_value, args):
    """設置結果保存的資料夾結構和logger捕獲"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 建立時間子資料夾 - 在當前目錄(vanilla_rnn)下建立verification_results
    current_dir = os.path.dirname(os.path.abspath(__file__))  # vanilla_rnn目錄
    session_dir = os.path.join(current_dir, "verification_results", f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 建立本次實驗的檔案名稱
    base_name = f"zerosplit_eps{eps_value}_N{args.N}_maxsplit{args.max_splits}"
    json_file = os.path.join(session_dir, f"{base_name}.json")
    txt_file = os.path.join(session_dir, f"{base_name}.txt")
    
    # 設置logger捕獲
    log_capture = LoggerCapture()
    
    # 添加捕獲handler到logger
    logger.add(log_capture.capture_handler, level="INFO", format="{message}")
    
    return json_file, txt_file, log_capture, session_dir

def extract_bounds_analysis_info(captured_logs):
    """從捕獲的logs中提取bounds analysis資訊"""
    bounds_analyses = []
    final_report = {}
    
    current_analysis = None
    in_final_report = False
    
    for log_entry in captured_logs:
        message = log_entry['message']
        
        # 檢測bounds analysis開始
        if "Bounds Analysis" in message and "---" in message:
            if current_analysis:
                bounds_analyses.append(current_analysis)
            
            stage_name = message.split("---")[1].strip().split(" ")[0]  # 提取stage名稱
            current_analysis = {
                'stage': stage_name,
                'timestamp': log_entry['timestamp'],
                'metrics': {}
            }
            
        # 檢測final report開始
        elif "FINAL BOUNDS IMPROVEMENT REPORT" in message:
            in_final_report = True
            final_report['start_timestamp'] = log_entry['timestamp']
            final_report['content'] = []
            
        # 收集當前analysis的指標
        elif current_analysis and any(keyword in message for keyword in [
            "Samples with bound improvement:",
            "Average bounds improvement:",
            "Potential false positive samples:",
            "Original safe samples:",
            "Unsafe samples before split:",
            "Safe samples after split:",
            "Unsafe samples to safe after split:",
            "Current stage verification rate:"
        ]):
            # 解析指標
            for keyword in ["Samples with bound improvement:", "Average bounds improvement:", 
                          "Potential false positive samples:", "Original safe samples:",
                          "Unsafe samples before split:", "Safe samples after split:",
                          "Unsafe samples to safe after split:", "Current stage verification rate:"]:
                if keyword in message:
                    value = message.split(keyword)[1].strip()
                    metric_key = keyword.replace(":", "").replace(" ", "_").lower()
                    current_analysis['metrics'][metric_key] = value
                    break
        
        # 收集final report內容 - 擴展關鍵字匹配
        elif in_final_report and any(keyword in message for keyword in [
            "指標", "Gap Reduction", "Improvement Sources", "Status Change", 
            "├─", "│", "└─", "總體效能評估", "⚠️", "RISK ASSESSMENT",
            "Samples with", "Average", "Maximum", "contribution", "effectiveness",
            "Safe samples", "Unsafe samples", "conversions", "success rate",
            "Overall", "enhancement", "suppression", "reduction"
        ]):
            final_report['content'].append({
                'timestamp': log_entry['timestamp'],
                'message': message
            })
    
    # 添加最後一個analysis
    if current_analysis:
        bounds_analyses.append(current_analysis)
    
    return bounds_analyses, final_report

def save_verification_results(json_file, txt_file, captured_logs, args, results_data, session_dir):
    """保存驗證結果，重點保存logger分析資訊"""
    
    # 提取bounds analysis和final report資訊
    bounds_analyses, final_report = extract_bounds_analysis_info(captured_logs)
    
    # 準備保存的數據
    save_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'session_dir': session_dir,
            'model_path': args.work_dir,
            'model_name': args.model_name,
            'hidden_size': args.hidden_size,
            'time_step': args.time_step,
            'activation': args.activation,
            'eps': args.eps,
            'p_norm': args.p,
            'N_samples': args.N,
            'max_splits': args.max_splits
        },
        'verification_results': results_data,
        'bounds_analysis': bounds_analyses,
        'final_report': final_report,
        'all_logs': [log for log in captured_logs]  # 保留完整log以備查
    }
    
    # 保存JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 保存TXT摘要 - 重點整理bounds analysis
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"ZeroSplit Verification Analysis Report\n")
        f.write(f"{'='*80}\n")
        f.write(f"Session: {os.path.basename(session_dir)}\n")
        f.write(f"Timestamp: {save_data['experiment_info']['timestamp']}\n")
        f.write(f"Model: {args.work_dir}\n")
        f.write(f"Parameters: hidden_size={args.hidden_size}, time_step={args.time_step}, activation={args.activation}\n")
        f.write(f"Test config: eps={args.eps}, p={args.p}, N={args.N}, max_splits={args.max_splits}\n\n")
        
        # 基本驗證結果
        f.write(f"Basic Verification Results:\n")
        f.write(f"{'-'*40}\n")
        f.write(f"Final verification: {'SUCCESS' if results_data['is_verified'] else 'FAILED'}\n")
        f.write(f"Total split count: {results_data['split_count']}\n")
        f.write(f"Predicted class: {results_data['top1_class']}\n")
        if not results_data['is_verified'] and results_data['unsafe_layer']:
            f.write(f"Unsafe layer: {results_data['unsafe_layer']}\n")
        f.write(f"\n")
        
        # Bounds Analysis摘要
        f.write(f"Bounds Analysis Summary:\n")
        f.write(f"{'-'*40}\n")
        for i, analysis in enumerate(bounds_analyses):
            f.write(f"Stage {i+1} - {analysis['stage']}:\n")
            for metric, value in analysis['metrics'].items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value}\n")
            f.write(f"\n")
        
        # Final Report摘要 - 顯示更多詳細指標
        if final_report and 'content' in final_report:
            f.write(f"Final Report Detailed Metrics:\n")
            f.write(f"{'-'*40}\n")
            
            # 按指標分組顯示
            current_metric = None
            for entry in final_report['content']:
                message = entry['message']
                
                # 檢測指標標題
                if "指標一" in message or "Gap Reduction" in message:
                    current_metric = "Gap Reduction"
                    f.write(f"\n{current_metric}:\n")
                elif "指標二" in message or "Improvement Sources" in message:
                    current_metric = "Improvement Sources"  
                    f.write(f"\n{current_metric}:\n")
                elif "指標三" in message or "Status Change" in message:
                    current_metric = "Status Change"
                    f.write(f"\n{current_metric}:\n")
                elif "總體效能評估" in message:
                    current_metric = "Overall Performance"
                    f.write(f"\n{current_metric}:\n")
                elif "RISK ASSESSMENT" in message:
                    current_metric = "Risk Assessment"
                    f.write(f"\n{current_metric}:\n")
                
                # 顯示詳細數據（去掉樹狀符號以便閱讀）
                if any(symbol in message for symbol in ["├─", "│", "└─"]):
                    clean_message = message.replace("├─", "").replace("│", "").replace("└─", "").strip()
                    if clean_message and "===" not in clean_message:
                        f.write(f"  {clean_message}\n")
                elif "⚠️" in message or "These samples changed" in message:
                    f.write(f"  {message}\n")

class ZeroSplitVerifier(RNN):
    def __init__(self, input_size, hidden_size, output_size, time_step, activation, max_splits=1, debug=False):
        RNN.__init__(self, input_size, hidden_size, output_size, time_step, activation) # 1 layer
        self.max_splits = max_splits
        self.split_count = 0
        self.debug = debug
        # 新增：output bounds tracking
        self.bound_tracker = {
            'output_bounds': None,
            'split_history': [],
            'current_split_count': 0,
        }
        
    def _compute_basic_bounds(self, eps, p, v, X, N, s, n, idx_eps, return_for_split=False):
        """計算當前timestep input的bounds
        
        Args:
            eps: float or Tensor, 擾動範圍
            p: float, p-norm (1, 2, or inf)
            v: int, 當前timestep
            X: Tensor, 輸入數據
            N: int, batch size
            s: int, hidden size
            n: int, input size
            idx_eps: Tensor, epsilon的索引
            
        Returns:
            tuple: (yU, yL) upper和lower bounds
        """
        with torch.no_grad():
            # 初始化bounds
            yU = torch.zeros(N, s, device=X.device)  # [N,s]
            yL = torch.zeros(N, s, device=X.device)  # [N,s]
            
            # 準備權重矩陣
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]   
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)  # [N, s]
            
            # 計算q norm
            if p == 1:
                q = float('inf')
            elif p == 'inf' or p == float('inf'):
                q = 1 
            else:
                q = p / (p-1)
                
            # 第一項: eps ||A^{<v>} W_ax||q
            if isinstance(eps, torch.Tensor):
                # eps is a tensor of N
                yU = yU + idx_eps[v-1] * eps.unsqueeze(1).expand(-1, s) * torch.norm(W_ax, p=q, dim=2)
                yL = yL - idx_eps[v-1] * eps.unsqueeze(1).expand(-1, s) * torch.norm(W_ax, p=q, dim=2)
            else:
                # eps is a number
                yU = yU + idx_eps[v-1] * eps * torch.norm(W_ax, p=q, dim=2)
                yL = yL - idx_eps[v-1] * eps * torch.norm(W_ax, p=q, dim=2)
                
            # 第二項: A^{<v>} W_ax x^{<v>} and Ou^{<v>} W_ax x^{<v>}
            if v == 1:
                X = X.view(N, 1, n)  # 確保維度正確           
            yU = yU + torch.matmul(W_ax, X[:, v-1, :].view(N, n, 1)).squeeze(2)  # [N, s]
            yL = yL + torch.matmul(W_ax, X[:, v-1, :].view(N, n, 1)).squeeze(2)  # [N, s]

            if return_for_split:
                input_yL = yL.clone().detach()
                input_yU = yU.clone().detach()
            
            # 第三項: A^{<v>} (b_a + Delta^{<v>}) and Ou^ {<v>} (b_a + Theta^{<v>})
            yU = yU + b_aa + b_ax  # [N, s]
            yL = yL + b_aa + b_ax  # [N, s]
            
            if self.debug:
                print(f"\nCurrent Timestep {v} input impact:")
                print(f"Current Timestep {v} input lower: {yL}")
                print(f"Current Timestep {v} input upper: {yU}")
            
            if return_for_split:
                return yU, yL, input_yU, input_yL
            else:
                return yU, yL
        
    def computePreactivationBounds(self, eps, p, X = None, Eps_idx = None, unsafe_layer=None, merge_results=True, cross_zero=None):
        """計算hidden layer的preactivation bounds"""
        if X is None:
            X = self.X
        
        if Eps_idx is None:
            Eps_idx = torch.arange(1, self.time_step + 1)
        
        results = []
        split_done = False
        
        # 依次計算每一層
        for v in range(1, self.time_step + 1):
            # 檢查是否已經執行過分割
            if unsafe_layer is not None and v > unsafe_layer:
                split_done = True
                break
                
            yL, yU = self.compute2sideBound(
                eps, p, v, X=X[:, 0:v, :], Eps_idx=Eps_idx,
                unsafe_layer=unsafe_layer, merge_results=merge_results,
                split_done=split_done, cross_zero=cross_zero
            )
            
            # 如果是問題層並且不合併結果，返回兩個子問題的結果
            if unsafe_layer == v and not merge_results:
                return yL, yU  # 這裡的yL, yU實際上是兩個子問題的結果
            
            results.append((yL, yU))
        
        # 返回最後一層的結果
        return results[-1]
        
    def compute2sideBound(self, eps, p, v, X = None, Eps_idx = None, unsafe_layer=None,
                           merge_results=True, split_done=False, cross_zero=None,
                           return_refine_inputs=False, refine_preh=None):

        # 用自定義pre-activation bounds因應有refine，否則就是current self.l/u
        if refine_preh is not None:

            refine_preh_l, refine_preh_u = refine_preh
            # 確保所有timestep都是子區間的正確狀態
            for i in range(len(refine_preh_l)):
                if refine_preh_l[i] is not None:
                    self.l[i] = refine_preh_l[i].clone()
                if refine_preh_u[i] is not None:
                    self.u[i] = refine_preh_u[i].clone()

        # X here is of size [batch_size, layer_index m, input_size]
        #eps could be a real number, or a tensor of size N
        #p is a real number
        #m is an integer
        with torch.no_grad():
            n = self.W_ax.shape[1] # input size
            s = self.W_ax.shape[0] # hidden size
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx - 1] = 1
            if X is None:
                X = self.X
            N = X.shape[0] # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device=X.device)
            else:
                a_0 = self.a_0
                
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]    
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)  # [N, s]
           
            # v-th terms基本計算
            if unsafe_layer == v and not split_done:
                yU, yL, input_yU, input_yL = self._compute_basic_bounds(eps, p, v, X, N, s, n, idx_eps, return_for_split=True)
                self.input_yL = input_yL
                self.input_yU = input_yU
            else:
                yU, yL = self._compute_basic_bounds(eps, p, v, X, N, s, n, idx_eps)
            
            if not (v == 1):
                self.split_count = 0
                # 用遞迴去算前面timestep的影響
                yL, yU, A, Ou = self._recursive_bound_compute(v-1, v, eps, p, X, idx_eps, yL, yU, 0)
                
                # 查看在k在每次往後推時的yL跟yU
                if self.debug:
                    print(f"Step {v-1} yL & yU in zsv:")
                    print(f"yL in step {v-1}: {yL}")
                    print(f"yU in step {v-1}: {yU}")
                
                # compute A^{<0>}
                A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
                Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
            
            else:
                A = W_aa  # A^ {<0>}, [N, s, s]
                Ou = W_aa  # Ou^ {<0>}, [N, s, s]
                
            yU = yU + torch.matmul(A, a_0.view(N, s, 1)).squeeze(2)  # [N, s]
            yL = yL + torch.matmul(Ou, a_0.view(N, s, 1)).squeeze(2) # [N, s]
            
            # 查看timestep = v時最後的yL跟yU
            if self.debug:
                print(f"Final timestep {v} in zsv:")
                print(f"yL in final timestep {v}: {yL}")
                print(f"yU in final timestep {v}: {yU}")
            
            self.l[v] = yL
            self.u[v] = yU
            if refine_preh is not None:
                refine_preh_l[v] = yL.clone().detach()
                refine_preh_u[v] = yU.clone().detach()

            # 檢查是否需要在current timestep進行split
            if unsafe_layer == v and not split_done:
                if not merge_results:
                    # 不合併結果，分別計算兩個子問題
                    result = self._split_and_compute_separate(eps, p, v, Eps_idx, cross_zero, return_refine_inputs)
                    
                    return result
                else:
                    # 合併結果
                    return self._split_and_merge(eps, p, v, Eps_idx, cross_zero)

            return yL, yU
        
    def adjust_eps_input_for_split(self, cur_v_input_l, cur_v_input_u, original_X, eps, v, p):

        # 取當前input bounds的中點
        mid = (cur_v_input_l + cur_v_input_u) / 2  # [N, hidden_size]

        # 計算正負區間的中心點，用於更新input
        neg_center = (cur_v_input_l + mid) / 2
        pos_center = (cur_v_input_u + mid) / 2
        
        # 用原始W_ax計算pseudo inverse
        W_ax_pinv = torch.pinverse(self.W_ax)  # [input_size, hidden_size]
        
        X_pos = original_X.clone().detach()  # [N, timestep, input_size]
        X_neg = original_X.clone().detach()
        
        # 只更新第v-1個時間步
        # neg_center: [N, hidden_size] @ W_ax_pinv.t(): [hidden_size, input_size] = [N, input_size]
        X_neg[:, v-1, :] = torch.matmul(neg_center, W_ax_pinv.t())
        X_pos[:, v-1, :] = torch.matmul(pos_center, W_ax_pinv.t())
        
        # pos and neg epsilon計算
        half_width = torch.abs(cur_v_input_u - pos_center) # (eps * ||W_ax||_q) / 2

        if p == 1:
            q = float('inf')
        elif p == float('inf'):
            q = 1
        else:
            q = p / (p-1)

        W_ax_norm = torch.norm(self.W_ax, p=q, dim=1)  # [hidden_size]
        W_ax_norm_inv = torch.reciprocal(W_ax_norm + 1e-8)  # [hidden_size]
        eps = (half_width * W_ax_norm_inv)[:, 0]  # [N, 1] -> [N]
        print(f"eps shape: {eps.shape}, eps: {eps}")
            
        return X_pos, X_neg, eps, eps

    def _split_and_compute_separate(self, eps, p, v, Eps_idx, cross_zero=None, return_refine_inputs=False):
        """分割當前層並分別計算兩個子問題"""
        # 保存原始bounds
        orig_l = self.l[v].clone().detach()
        orig_u = self.u[v].clone().detach()
        full_X = self.original_X.clone().detach()
        print(f"full_X shape: {full_X.shape}")

        cur_v_input_l = self.input_yL
        cur_v_input_u = self.input_yU
        
        if self.debug:
            print(f"Splitting layer {v}, cross_zero count: {cross_zero.sum().item()}")
        
        # 結果儲存
        pos_yL, pos_yU = None, None
        neg_yL, neg_yU = None, None
        
        # 正區間: x >= 0
        pos_l = orig_l.clone().detach()
        pos_u = orig_u.clone().detach()
        pos_l[cross_zero] = 0

        # 得到正負區間新的input和epsilon
        X_pos, X_neg, eps_pos, eps_neg = self.adjust_eps_input_for_split(cur_v_input_l, cur_v_input_u, full_X, eps,  v, p)
        print(f"X_pos shape: {X_pos.shape}")
        # self.l[v] = pos_l
        # self.u[v] = pos_u

        # 計算正區間 - 創建正區間的pre-activation bounds
        pos_l_state = [bound.clone() if bound is not None else None for bound in self.l]
        pos_l_state[v] = pos_l
        pos_u_state = [bound.clone() if bound is not None else None for bound in self.u] 
        pos_u_state[v] = pos_u
        
        # 根據v是否為最後一個隱藏層選擇計算方式
        if v < self.time_step:
            # 從v+1層開始計算到最後一個隱藏層
            for k in range(v+1, self.time_step+1):
                pos_yL, pos_yU = self.compute2sideBound(
                    eps_pos, p, k, X=X_pos[:, 0:k, :], Eps_idx=Eps_idx,
                    refine_preh=(pos_l_state, pos_u_state)
                ) # eps換成eps_pos
                # print(f"第 {k} timestep的pre-activation bounds: 正區間下界: {pos_yL}, 正區間上界: {pos_yU}")
            
            # 計算輸出層
            pos_yL, pos_yU = self.computeLast2sideBound(
                eps_pos, p, v=self.time_step+1, X=X_pos, Eps_idx=Eps_idx
            )
            # print(f"正區間的final bounds: {pos_yL}, {pos_yU}")
            print(f"正區間的bounds差異: {pos_yU - pos_yL}")
        else:
            # 確保一直到最後一timestep才切的時候，self.l和self.u有被正確更新
            if pos_l_state is not None:

                # 只更新對應的timestep
                for i in range(len(pos_l_state)):
                    if pos_l_state[i] is not None:
                        self.l[i] = pos_l_state[i].clone()
                    if pos_u_state[i] is not None:
                        self.u[i] = pos_u_state[i].clone()

            # v已經是最後一個隱藏層，直接計算輸出層
            pos_yL, pos_yU = self.computeLast2sideBound(
                eps_pos, p, v=self.time_step+1, X=X_pos, Eps_idx=Eps_idx
            )
            # print(f"正區間的final bounds: {pos_yL}, {pos_yU}")
            print(f"正區間的bounds差異: {pos_yU - pos_yL}")
        
        # 負區間: x <= 0
        neg_l = orig_l.clone().detach()
        neg_u = orig_u.clone().detach()
        neg_u[cross_zero] = 0
        
        # self.l[v] = neg_l
        # self.u[v] = neg_u

        # 計算負區間 - 創建負區間的pre-activation bounds
        neg_l_state = [bound.clone() if bound is not None else None for bound in self.l]
        neg_l_state[v] = neg_l
        neg_u_state = [bound.clone() if bound is not None else None for bound in self.u]
        neg_u_state[v] = neg_u
        
        # 根據v是否為最後一個隱藏層選擇計算方式
        if v < self.time_step:
            # 從v+1層開始計算到最後一個隱藏層
            for k in range(v+1, self.time_step+1):
                neg_yL, neg_yU = self.compute2sideBound(
                    eps_neg, p, k, X=X_neg[:, 0:k, :], Eps_idx=Eps_idx,
                    refine_preh=(neg_l_state, neg_u_state)
                ) # eps換eps_neg
                # print(f"第 {k} timestep的pre-activation bounds: 負區間下界: {neg_yL}, 負區間上界: {neg_yU}")
            
            # 計算輸出層
            neg_yL, neg_yU = self.computeLast2sideBound(
                eps_neg, p, v=self.time_step+1, X=X_neg, Eps_idx=Eps_idx
            )
            # print(f"負區間的final bounds: {neg_yL}, {neg_yU}")
            print(f"負區間的bounds差異: {neg_yU - neg_yL}")
        else:
            # 確保一直到最後一timestep才切的時候，self.l和self.u有被正確更新
            if neg_l_state is not None:

                # 只更新對應的timestep
                for i in range(len(neg_l_state)):
                    if neg_l_state[i] is not None:
                        self.l[i] = neg_l_state[i].clone()
                    if neg_u_state[i] is not None:
                        self.u[i] = neg_u_state[i].clone()

            # v已經是最後一個隱藏層，直接計算輸出層
            neg_yL, neg_yU = self.computeLast2sideBound(
                eps_neg, p, v=self.time_step+1, X=X_neg, Eps_idx=Eps_idx
            )
            # print(f"負區間的final bounds: {neg_yL}, {neg_yU}")
            print(f"負區間的bounds差異: {neg_yU - neg_yL}")
        
        # 恢復原始bounds
        # self.l[v] = orig_l
        # self.u[v] = orig_u
        
        # 返回兩個子問題的最終結果
        if return_refine_inputs:
            return (pos_yL, pos_yU), (neg_yL, neg_yU), \
                   (X_pos, eps_pos, (pos_l_state, pos_u_state)), \
                   (X_neg, eps_neg, (neg_l_state, neg_u_state))
        else:
            return (pos_yL, pos_yU), (neg_yL, neg_yU)

    def _split_and_merge(self, eps, p, v, Eps_idx, cross_zero=None):
        """分割當前層並合併兩個子問題的結果"""
        # 獲取分別計算的結果
        (pos_yL, pos_yU), (neg_yL, neg_yU) = self._split_and_compute_separate(
            eps, p, v, Eps_idx, cross_zero=cross_zero
        )
        
        # 計算每個區間的bounds width
        pos_width = pos_yU - pos_yL  # [N, output_size]
        neg_width = neg_yU - neg_yL  # [N, output_size]
        
        # Element-wise選擇更鬆的區間(worst case)
        pos_worse = pos_width >= neg_width  # [N, output_size]
        
        # 創建合併結果
        yL = torch.where(pos_worse, pos_yL, neg_yL)
        yU = torch.where(pos_worse, pos_yU, neg_yU)

        print(f"Merged bounds: {yL}, {yU}")
        print(f"Bounds difference: {yU - yL}")
        
        return yL, yU
        
    def  computeLast2sideBound(self, eps, p, v, X=None, Eps_idx=None):
        with torch.no_grad():
            n = self.W_ax.shape[1] # input size
            s = self.W_ax.shape[0] # hidden size
            t = self.W_fa.shape[0] # output size
            
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx - 1] = 1
            if X is None:
                X = self.X
            N = X.shape[0] # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device=X.device)
            else:
                a_0 = self.a_0
                
            if p == 1:
                q = float('inf')
            elif p == 'inf' or p == float('inf'):
                q = 1
            else:
                q = p / (p-1)
                
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]
            b_f = self.b_f.unsqueeze(0).expand(N,-1)  # [N, t]
            
            yU = torch.zeros(N, t, device=X.device)  # [N, t]
            yL = torch.zeros(N, t, device=X.device)  # [N, t]
            
            self.split_count = 0
            
            yL, yU, A, Ou = self._recursive_bound_compute(v-1, v, eps, q, X, idx_eps, yL, yU, 0, is_last_layer=True)
            
            # compute A^{<0>}
            A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
            Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
            yU = yU + torch.matmul(A,a_0.view(N,s,1)).squeeze(2)  # A^ {<0>} * a_0
            yL = yL + torch.matmul(Ou,a_0.view(N,s,1)).squeeze(2)  # Ou^ {<0>} * a_0
            yU = yU + b_f
            yL = yL + b_f
            
            return yL, yU    
            
            
        
    def _recursive_bound_compute(self, k, v, eps, p, X, idx_eps, yL, yU, split_count, 
                            is_last_layer=False, prev_A=None, prev_Ou=None):
        """處理 k from v-1 to 1的部分"""
        # 基本情況: k<1時停止遞歸
        if k < 1:
            return yL, yU, prev_A, prev_Ou
        
        # 正常處理，無需考慮分割
        cur_yL, cur_yU, A, Ou = self._compute_bounds_without_split(
            k, v, eps, p, X, yL, yU, idx_eps, is_last_layer, 
            prev_A=prev_A, prev_Ou=prev_Ou
        )
        
        if k == 1:
            return cur_yL, cur_yU, A, Ou
        else:
            return self._recursive_bound_compute(
                k-1, v, eps, p, X, idx_eps, cur_yL, cur_yU, split_count,
                is_last_layer, prev_A=A, prev_Ou=Ou
            )
    
    def _compute_bounds_without_split(self, k, v, eps, p, X, yL, yU, idx_eps, is_last_layer=False, prev_A=None, prev_Ou=None):
        """計算沒有split時的bounds
        Args:
            k: int, timestep
            v: int, 當前時間步
            eps: float or Tensor, 擾動範圍
            p: float, p-norm (1, 2, or inf)
            X: Tensor, 輸入數據
            yL: Tensor, lower bounds
            yU: Tensor, upper bounds
            idx_eps: Tensor, eps的index
        """

        s = self.W_ax.shape[0]
        n = self.W_ax.shape[1]
        t = self.W_fa.shape[0]
        N = X.shape[0]
        W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]
        W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
        W_fa = self.W_fa.unsqueeze(0).expand(N,-1,-1)  # [N, t, s]
        b_aa = self.b_aa.unsqueeze(0).expand(N,-1)  # [N, s]
        b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
        
        if k == v-1:
            # 計算slopes和intercepts
            kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                self.l[k], self.u[k], self.activation)
                
            # 避免除以0
            bl = bl/ (kl + 1e-8)
            bu = bu/ (ku + 1e-8)
            
            # 查看第k層的kl, ku, bl, bu
            if self.debug:
                print(f"Layer {k} in zsv:")
                print(f"kl in layer {k}: {kl}")
                print(f"ku in layer {k}: {ku}")
                print(f"bl in layer {k}: {bl}")
                print(f"bu in layer {k}: {bu}")
            
            # 將斜率跟截距存著
            self.kl[k] = kl  # [N, s]
            self.ku[k] = ku  # [N, s]
            self.bl[k] = bl  # [N, s]
            self.bu[k] = bu  # [N, s]
            
            # 剩餘計算與原始POPQORN相同
            if is_last_layer:
                alpha_l = kl.unsqueeze(1).expand(-1, t, -1)
                alpha_u = ku.unsqueeze(1).expand(-1, t, -1)
                beta_l = bl.unsqueeze(1).expand(-1, t, -1)
                beta_u = bu.unsqueeze(1).expand(-1, t, -1)
            else:
                alpha_l = kl.unsqueeze(1).expand(-1, s, -1)
                alpha_u = ku.unsqueeze(1).expand(-1, s, -1)
                beta_l = bl.unsqueeze(1).expand(-1, s, -1)
                beta_u = bu.unsqueeze(1).expand(-1, s, -1)
            
            # Element-wise
            if is_last_layer:
                I = (W_fa >= 0).float()
                lamida = I*alpha_u + (1-I)*alpha_l
                omiga = I*alpha_l + (1-I)*alpha_u
                Delta = I*beta_u + (1-I)*beta_l
                Theta = I*beta_l + (1-I)*beta_u
                
                # 最後一層使用W_fa
                A = W_fa * lamida
                Ou = W_fa * omiga
            else:
                I = (W_aa >= 0).float()
                lamida = I*alpha_u + (1-I)*alpha_l
                omiga = I*alpha_l + (1-I)*alpha_u
                Delta = I*beta_u + (1-I)*beta_l
                Theta = I*beta_l + (1-I)*beta_u
                
                A = W_aa * lamida
                Ou = W_aa * omiga
        else:
            
            if is_last_layer:
                # 使用之前存的參數計算
                alpha_l = self.kl[k].unsqueeze(1).expand(-1, t, -1)
                alpha_u = self.ku[k].unsqueeze(1).expand(-1, t, -1)
                beta_l = self.bl[k].unsqueeze(1).expand(-1, t, -1)
                beta_u = self.bu[k].unsqueeze(1).expand(-1, t, -1)
            else:
                # 使用之前存的參數計算
                alpha_l = self.kl[k].unsqueeze(1).expand(-1, s, -1)
                alpha_u = self.ku[k].unsqueeze(1).expand(-1, s, -1)
                beta_l = self.bl[k].unsqueeze(1).expand(-1, s, -1)
                beta_u = self.bu[k].unsqueeze(1).expand(-1, s, -1)
            
            # 用目前的A和Ou來計算新的參數
            I = (torch.matmul(prev_A, W_aa) >= 0).float()
            lamida = I*alpha_u + (1-I)*alpha_l
            Delta = I*beta_u + (1-I)*beta_l
            
            I = (torch.matmul(prev_Ou, W_aa) >= 0).float()
            omiga = I*alpha_l + (1-I)*alpha_u
            Theta = I*beta_l + (1-I)*beta_u
            
            A = torch.matmul(prev_A, W_aa) * lamida
            Ou = torch.matmul(prev_Ou, W_aa) * omiga
        
        ## first term
        if type(eps) == torch.Tensor:
            if is_last_layer:
                yU = yU + idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                            t)*torch.norm(torch.matmul(A,W_ax),p=p,dim=2)  # eps ||A^ {<k>} W_ax||q2
                yL = yL - idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                            t)*torch.norm(torch.matmul(Ou,W_ax),p=p,dim=2)  # eps ||Ou^ {<k>} W_ax||q2
            
            else:                
                #eps is a tensor of size N 
                yU = yU + idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                            s)*torch.norm(torch.matmul(A,W_ax),p=p,dim=2)  # eps ||A^ {<k>} W_ax||q    
                yL = yL - idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                            s)*torch.norm(torch.matmul(Ou,W_ax),p=p,dim=2)  # eps ||Ou^ {<k>} W_ax||q      
        else:
            yU = yU + idx_eps[k-1]*eps*torch.norm(torch.matmul(A,W_ax),p=p,dim=2)  # eps ||A^ {<k>} W_ax||q   
            yL = yL - idx_eps[k-1]*eps*torch.norm(torch.matmul(Ou,W_ax),p=p,dim=2)  # eps ||Ou^ {<k>} W_ax||q  
        ## second term
        yU = yU + torch.matmul(A,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # A^ {<k>} W_ax x^{<k>}            
        yL = yL + torch.matmul(Ou,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # Ou^ {<k>} W_ax x^{<k>}       
        ## third term
        yU = yU + torch.matmul(A,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(A*Delta).sum(2)  # A^ {<k>} (b_a + Delta^{<k>})
        yL = yL + torch.matmul(Ou,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(Ou*Theta).sum(2)  # Ou^ {<k>} (b_a + Theta^{<k>})
        
        return yL, yU, A, Ou
    
    def verify_robustness(self, X, eps):
        # 1. 得到原始top1預測
        with torch.no_grad():
            output = self(X)
            top1_class = output.argmax(dim=1)  # [N]

        # 2. 計算bound
        yL, yU = self.computePreactivationBounds(eps, p=2, X=X, 
                                        Eps_idx=torch.arange(1,self.time_step+1))
        print(f"不做split的隱藏層bounds to verify: {yL}, {yU}")
        
        yL_out, yU_out = self.computeLast2sideBound(eps, p=2, v=self.time_step+1,
                                                X=X, Eps_idx=torch.arange(1,self.time_step+1))
        
        print(f"不做split的輸出層 bounds to verify: {yL_out}, {yU_out}")
        print(f"First time difference between yL and yU: {yU_out - yL_out}")

        # 3. 檢查robustness
        N = X.shape[0]
        robust = True
        # top1_class = 1
        for i in range(N):
            top1 = top1_class[i]
            other_classes = [j for j in range(self.output_size) if j != top1]
            
            # 檢查top1 class的lower bound是否大於其他所有class的upper bound
            if not all(yL_out[i,top1] > yU_out[i,j] for j in other_classes):
                robust = False
                break
        
        # 多返回yL_out和yU_out
        return robust, top1_class, yL_out, yU_out

    def has_violation(self, layer_bounds):
        """檢查當前layer的bound是否有violation
        
        Returns:
            - violation_detected: bool, 是否檢測到violation
            - cross_zero: tensor, 哪些部分跨越0
        """
        yL, yU = layer_bounds 
        
        # 檢查跨越0的部分
        cross_zero = (yL < 0) & (yU > 0)
        
        # 如果有跨越0，就視為有violation需要split
        return cross_zero.any(), cross_zero

    def locate_unsafe_layer(self, start_timestep=1, refine_preh=None):
        """用Sequential(原為Binary) search找到第一個出現violation的layer(需要精進)"""
        
        # 依序檢查從start_timestep開始的每一層
        for timestep in range(start_timestep, self.time_step+1):
            # 如果有提供refine_preh，則使用它來獲取當前layer的bounds
            # 否則初始使用self.l和self.u
            if refine_preh is not None:
                preh_lower = refine_preh[0][timestep]
                preh_upper = refine_preh[1][timestep]
                layer_bounds = (preh_lower, preh_upper)
            else:
                layer_bounds = (self.l[timestep], self.u[timestep])
            violation, cross_zero = self.has_violation(layer_bounds)
            if violation:
                return timestep, cross_zero
        
        return None, None

    def verify_network(self, X, eps, max_splits=None, merge_results=True):
        """整體的驗證流程"""

        self.original_X = X.clone().detach()
        
        # 第一次驗證，不做split
        is_verified, top1_class = self.verify_robustness(X, eps)
        # if is_verified:
        #     return True, None, top1_class
            
        # 若驗證失敗，找出問題layer並進行split
        unsafe_layer, cross_zero = self.locate_unsafe_layer()
        if unsafe_layer is None:
            return False, None, top1_class
        
        print(f"Found unsafe layer: {unsafe_layer} with {cross_zero.sum()} dims crossing zero")
        
        # 清空中間變數
        self.clear_intermediate_variables()
        
        # 根據merge_results來決定是否要合併結果
        if merge_results:
            # 合併驗證(這裡回傳的是已經調用過computeLast2sideBound的結果)
            yL_out, yU_out = self.computePreactivationBounds(
                eps, p=2, X=X, Eps_idx=torch.arange(1, self.time_step + 1),
                unsafe_layer=unsafe_layer, merge_results=True, cross_zero=cross_zero
            )
            
            # 計算最後一層
            # yL_out, yU_out = self.computeLast2sideBound(
            #     eps, p=2, v=self.time_step + 1, X=X,
            #     Eps_idx=torch.arange(1, self.time_step + 1)
            # )
            
            # 驗證
            N = X.shape[0]
            is_verified = True
            
            # for i in range(N):
            #     top1 = top1_class[i]
            #     other_classes = [j for j in range(self.output_size) if j != top1]
                
            #     if not all(yL_out[i, top1] > yU_out[i, j] for j in other_classes):
            #         is_verified = False
            #         break
                    
            return is_verified, unsafe_layer, top1_class, yL_out, yU_out
        
        else:
            # 分開驗證兩個子問題模式
            # 首先計算到問題層
            for k in range(1, unsafe_layer):
                yL, yU = self.compute2sideBound(
                    eps, p=2, v=k, X=X[:, 0:k, :], 
                    Eps_idx=torch.arange(1, self.time_step + 1)
                )
            
            # 在問題層執行分割並獲取兩個子問題的結果
            (pos_bounds, neg_bounds) = self.compute2sideBound(
                eps, p=2, v=unsafe_layer, X=X[:, 0:unsafe_layer, :], 
                Eps_idx=torch.arange(1, self.time_step + 1),
                unsafe_layer=unsafe_layer, merge_results=False, cross_zero=cross_zero
            )
            
            # 從split_and_compute_separate獲取的結果格式為((pos_yL, pos_yU), (neg_yL, neg_yU))
            (pos_yL_out, pos_yU_out) = pos_bounds
            (neg_yL_out, neg_yU_out) = neg_bounds
            print(f"正區間的final bounds: {pos_yL_out}, {pos_yU_out}")
            print(f"正區間的bounds差異: {pos_yU_out - pos_yL_out}")
            print(f"負區間的final bounds: {neg_yL_out}, {neg_yU_out}")
            print(f"負區間的bounds差異: {neg_yU_out - neg_yL_out}")
            
            # 驗證兩個子問題
            N = X.shape[0]
            pos_verified = True
            neg_verified = True
            
            for i in range(N):
                top1 = top1_class[i]
                other_classes = [j for j in range(self.output_size) if j != top1]
                
                # 檢查正區間
                if not all(pos_yL_out[i, top1] > pos_yU_out[i, j] for j in other_classes):
                    pos_verified = False
                
                # 檢查負區間
                if not all(neg_yL_out[i, top1] > neg_yU_out[i, j] for j in other_classes):
                    neg_verified = False
            
            # 驗證標準：只要有一個子問題驗證成功，整個問題就驗證成功
            is_verified = pos_verified and neg_verified
            
            print(f"正區間驗證結果: {pos_verified}")
            print(f"負區間驗證結果: {neg_verified}")
            print(f"整體驗證結果: {is_verified}")
            
            return is_verified, unsafe_layer, top1_class
        
    def verify_network_recursive(self, X, eps, max_splits=3):
        self.original_X = X.clone().detach()

        # 第一次驗證
        logger.info("=== Bounds Tracking Analysis ===")
        is_verified, top1_class, yL_out, yU_out = self.verify_robustness(X, eps)

        self.bound_tracker['output_bounds'] = {
            'yL': yL_out,
            'yU': yU_out,
            'verified': is_verified
        }

        self.log_bounds_analysis(yL_out, yU_out, top1_class, "Original")
        if is_verified:
            logger.info("第一次就驗證成功，無須split")
            return True, None, top1_class
        
        # 若驗證失敗，找出問題layer並進行split
        logger.info("第一次驗證失敗，開始split並尋找unsafe layer")
        is_verified = self._recursive_split_verify(X, eps, top1_class, split_count=0, max_splits=max_splits,
                                                    start_timestep=1, refine_preh=None)
        
        # 輸出最終分析報告
        self.generate_final_report()
        
        return is_verified, None, top1_class
    
    def _initialize_bounds_tracking(self):
        """初始化bounds tracking"""
        self.bound_tracker = {
            'output_bounds': None,
            'split_history': [],
            'current_split_count': 0,
        }
    
    def _recursive_split_verify(self, X, eps, top1_class, split_count, max_splits, start_timestep=1, refine_preh=None):
        
        self.original_X = X.clone().detach()

        # 終止條件判斷
        if split_count >= max_splits:
            logger.info(f"達到最大 split 數{max_splits}次，驗證失敗")
            return False
        
        # 每次往後找第一個unsafe layer
        unsafe_layer, cross_zero = self.locate_unsafe_layer(start_timestep, refine_preh)
        if unsafe_layer is None:
            logger.info("沒有找到unsafe layer")
            return True

        logger.info(f"Found unsafe layer: {unsafe_layer} with {cross_zero.sum()} dims crossing zero")

        # 在unsafe layer split為正負兩區域 (Input取到unsafe_layer，之後會調整X和eps)
        pos_bounds, neg_bounds, pos_input, neg_input = self.compute2sideBound(
            eps, p=2, v=unsafe_layer, X=X[:, 0:unsafe_layer, :],
            Eps_idx=torch.arange(1, self.time_step + 1),
            unsafe_layer=unsafe_layer, merge_results=False, cross_zero=cross_zero,
            return_refine_inputs=True, refine_preh=refine_preh
        )

        (pos_yL_out, pos_yU_out) = pos_bounds
        (neg_yL_out, neg_yU_out) = neg_bounds
        (X_pos, eps_pos, pos_bounds_state) = pos_input
        (X_neg, eps_neg, neg_bounds_state) = neg_input

        # 紀錄split完的output bounds
        self.bound_tracker['current_split_count'] = split_count + 1
        self.record_split_bounds(pos_yL_out, pos_yU_out, neg_yL_out, neg_yU_out, 
                                    unsafe_layer, top1_class, "Pos_Neg_split")

        # 驗證並遞迴正區間
        pos_verified = self._check_and_recurse(
        pos_yL_out, pos_yU_out, top1_class, X_pos, eps_pos,
        split_count, max_splits, unsafe_layer + 1, pos_bounds_state, "Positive"
        )

        # 驗證並遞迴負區間
        neg_verified = self._check_and_recurse(
            neg_yL_out, neg_yU_out, top1_class, X_neg, eps_neg,
            split_count, max_splits, unsafe_layer + 1, neg_bounds_state, "Negative"
        )   

        success = pos_verified and neg_verified
        logger.info(f"Split {split_count + 1} results: Positive={pos_verified}, Negative={neg_verified}")
        return success
    
    def _check_and_recurse(self, yL_out, yU_out, top1_class, X, eps, 
                      split_count, max_splits, next_timestep, refine_preh, region_type):
        """檢查驗證結果，如果失敗則遞迴分割"""
        N = yL_out.shape[0]
        
        # 檢查是否有任何樣本驗證失敗
        failed_samples = []
        for i in range(N):
            top1 = top1_class[i]
            other_classes = [j for j in range(self.output_size) if j != top1]
            
            # 如果當前樣本驗證失敗，需要進一步分割
            if not all(yL_out[i, top1] > yU_out[i, j] for j in other_classes):
                failed_samples.append(i)

        if failed_samples:
            logger.info(f"Region {region_type}: {len(failed_samples)}個 samples 驗證失敗, 從下一個 timestep {next_timestep}往下尋找切分timestep")

            # 紀錄遞迴前的output bounds
            # self.record_split_bounds(yL_out, yU_out, None, None, 
            #                         next_timestep-1, top1_class, f"{region_type}_before_split")
            
            return self._recursive_split_verify(
                X, eps, top1_class, split_count + 1, max_splits,
                start_timestep=next_timestep, refine_preh=refine_preh
            )

        # 所有樣本都驗證成功
        logger.info(f"Region {region_type}: 當前子區間所有樣本驗證成功")
        return True
    
    def record_split_bounds(self, pos_yL_out, pos_yU_out, neg_yL_out, neg_yU_out,
                            timestep, top1_class, split_type):
        """紀錄split後的output bounds"""
        split_record = {
            'split_count': self.bound_tracker['current_split_count'], # 目前切了幾次
            'timestep': timestep,
            'split_type': split_type, # 目前在哪個階段：有分Original, Positive, Negative 以及 Pos_Neg_Split(獲得正負區間的output bounds之後)
            'pos_output_bounds': {'pos_yL': pos_yL_out.clone(), 'pos_yU': pos_yU_out.clone()}, # 正區間的output bounds
            'improvements': self.compute_bounds_improve(pos_yL_out, pos_yU_out, top1_class), # List
        }

        if neg_yL_out is not None and neg_yU_out is not None:
            split_record['neg_output_bounds'] = {
                'neg_yL': neg_yL_out.clone(),
                'neg_yU': neg_yU_out.clone()
            }
            split_record['neg_improvements'] = self.compute_bounds_improve(neg_yL_out, neg_yU_out, top1_class)

        self.bound_tracker['split_history'].append(split_record)

        # 即時log當前改進狀況
        self.log_bounds_analysis(pos_yL_out, pos_yU_out,
            top1_class, f"Split {self.bound_tracker['current_split_count']}"
        )

    def compute_bounds_improve(self, yL_out, yU_out, top1_class):
        """計算相對於原始output bounds的改進狀況"""
        if self.bound_tracker['output_bounds'] is None:
            return {}
        
        original_yL = self.bound_tracker['output_bounds']['yL']
        original_yU = self.bound_tracker['output_bounds']['yU']

        N = yL_out.shape[0]
        improvements = []

        for i in range(N):
            top1 = top1_class[i]
            other_classes = [j for j in range(self.output_size) if j != top1]

            sample_improvement = {
                'sample_idx': i,
                'top1_class': top1.item(),
                'original_safe': all(original_yL[i, top1] > original_yU[i, j] for j in other_classes),
                'split_safe': all(yL_out[i, top1] > yU_out[i, j] for j in other_classes),
            }

            # 1.計算安全邊界(Worst-Case Gap Reduction)變化
            original_margin = original_yL[i, top1] - max([original_yU[i, j] for j in other_classes])
            current_margin = yL_out[i, top1] - max([yU_out[i, j] for j in other_classes])
            sample_improvement['margin_improvement'] = (current_margin - original_margin)

            # 2.將Improvements拆成：top1 yL提高量 + 其他class yU降低量
            original_yL_top1_imp = yL_out[i, top1] - original_yL[i, top1]
            sample_improvement['top1_yL_improvement'] = original_yL_top1_imp.item()
            other_class_yU_decrease = [original_yU[i, j] - yU_out[i, j] for j in other_classes]
            total_other_yU_decrease = sum(other_class_yU_decrease).item()
            sample_improvement['other_class_yU_decrease'] = total_other_yU_decrease

            improvements.append(sample_improvement)

        return improvements
    
    def log_bounds_analysis(self, yL_out, yU_out, top1_class, stage_name):

        logger.info(f"\n--- {stage_name} Bounds Analysis ---")

        if self.bound_tracker['output_bounds'] is None:
            logger.info("Recording original bounds as baseline")
            return
        
        original_yL = self.bound_tracker['output_bounds']['yL']
        original_yU = self.bound_tracker['output_bounds']['yU']

        N = yL_out.shape[0]
        improved_samples = 0 # 改善的sample數量
        total_improvements = 0 # bound對每個維度的改善總和
        false_positive_risk = 0 # 原先POPQORN無法驗證但split完之後有驗證成功
        N_safe_original = 0
        N_safe_split = 0
        N_unsafe_original = 0
        N_unsafe_to_safe = 0
        
        # 整體的improvements check
        for i in range(N):
            imp_check_list = [] # for每個sample裡面的top1 class與其他class的output bounds改善程度
            top1 = top1_class[i]
            other_classes = [j for j in range(self.output_size) if j != top1]

            # 未split前的output bounds安全結果
            original_safe = all(original_yL[i, top1] > original_yU[i, j] for j in other_classes)
            N_safe_original += int(original_safe)
            # split後的output bounds安全結果
            split_safe = all(yL_out[i, top1] > yU_out[i, j] for j in other_classes)
            N_safe_split += int(split_safe)
            N_unsafe_original += int(not original_safe)
            N_unsafe_to_safe += int(split_safe and not original_safe)

            # 檢查改進
            for j in other_classes:
                original_violation = original_yU[i, j] - original_yL[i, top1]
                split_violation = yU_out[i, j] - yL_out[i, top1]
                improvement = original_violation - split_violation
                imp_check_list.append(improvement)

            # 全部都是正數代表此sample有改善
            if all(imp > 1e-6 for imp in imp_check_list):
                improved_samples += 1
                total_improvements += sum(imp_check_list)

            # 檢查是否有false positive風險
            if split_safe and not original_safe:
                false_positive_risk += 1
                logger.warning(f"Sample {i} has false positive risk after split: ")

        # 統計報告
        logger.info(f"Samples with bound improvement: {improved_samples}/{N}")
        if improved_samples > 0:
            avg_improvement = total_improvements / improved_samples
            logger.info(f"Average bounds improvement: {avg_improvement:.6f}")

        if false_positive_risk > 0:
            logger.warning(f"Potential false positive samples: {false_positive_risk}/{N}")

        logger.info(f"Original safe samples: {N_safe_original}/{N}")
        logger.info(f"Unsafe samples before split: {N_unsafe_original}/{N}")
        logger.info(f"Safe samples after split: {N_safe_split}/{N}")
        logger.info(f"Unsafe samples to safe after split: {N_unsafe_to_safe}/{N}")

        logger.info(f"Current stage verification rate: {sum(1 for i in range(N) if all(yL_out[i, top1_class[i]] > yU_out[i, j] for j in range(self.output_size) if j != top1_class[i]))}/{N}")

    def generate_final_report(self):
        """生成最終的驗證報告"""
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL BOUNDS IMPROVEMENT REPORT")
        logger.info(f"{'='*80}")

        if not self.bound_tracker['split_history']:
            logger.info("No splits were performed")
            return
        
        # 總體統計
        total_splits = len(self.bound_tracker['split_history'])
        final_split = self.bound_tracker['split_history'][-1]

        logger.info(f"Total splits performed: {total_splits}")
        logger.info(f"Final split timestep: {final_split['timestep']}")

        # 分析最終改進 - 使用三大指標
        if 'improvements' in final_split:
            improvements = final_split['improvements']
            N = len(improvements)

            # 指標一：最差情況間隙縮減量 (Worst-Case Gap Reduction)
            worst_gap_improvements = []
            samples_gap_improved = 0
            samples_unsafe_to_safe = 0
            
            # 指標二：改進來源分解 (Decomposition of Improvement Sources)
            total_top1_yL_improvement = 0
            total_other_yU_decrease = 0
            samples_with_top1_improvement = 0
            samples_with_other_decrease = 0
            
            # 指標三：驗證狀態轉變率 (Verification Status Change Rate)
            N_safe_original = 0
            N_safe_split = 0
            N_unsafe_original = 0
            false_positive_count = 0

            for imp in improvements:
                # 指標一：計算最差情況間隙縮減
                margin_improvement = imp['margin_improvement'].item()
                worst_gap_improvements.append(margin_improvement)
                
                if margin_improvement > 1e-6:
                    samples_gap_improved += 1
                
                # 指標二：改進來源分解
                top1_improvement = imp['top1_yL_improvement']
                other_decrease = imp['other_class_yU_decrease'] 
                
                if top1_improvement > 1e-6:
                    total_top1_yL_improvement += top1_improvement
                    samples_with_top1_improvement += 1
                    
                if other_decrease > 1e-6:
                    total_other_yU_decrease += other_decrease
                    samples_with_other_decrease += 1
                
                # 指標三：驗證狀態統計
                original_safe = imp['original_safe']
                split_safe = imp['split_safe']
                
                N_safe_original += int(original_safe)
                N_safe_split += int(split_safe)
                N_unsafe_original += int(not original_safe)
                
                # 檢查從不安全轉為安全
                if split_safe and not original_safe:
                    samples_unsafe_to_safe += 1
                    
                # 檢查false positive
                if split_safe and not original_safe:
                    false_positive_count += 1

            # === 報告三大指標 ===
            logger.info(f"\n=== 指標一：最差情況間隙縮減量 (Worst-Case Gap Reduction) ===")
            logger.info(f"  ├─ Samples with gap reduction: {samples_gap_improved}/{N}")
            if samples_gap_improved > 0:
                avg_gap_reduction = sum(imp for imp in worst_gap_improvements if imp > 1e-6) / samples_gap_improved
                max_gap_reduction = max(worst_gap_improvements)
                logger.info(f"  ├─ Average gap reduction: {avg_gap_reduction:.6f}")
                logger.info(f"  └─ Maximum gap reduction: {max_gap_reduction:.6f}")
            else:
                logger.info(f"  └─ No significant gap reduction achieved")

            logger.info(f"\n=== 指標二：改進來源分解 (Decomposition of Improvement Sources) ===")
            logger.info(f"  ├─ Top-1 lower bound improvements:")
            logger.info(f"  │   ├─ Samples improved: {samples_with_top1_improvement}/{N}")
            if samples_with_top1_improvement > 0:
                avg_top1_improvement = total_top1_yL_improvement / samples_with_top1_improvement
                logger.info(f"  │   └─ Average improvement: {avg_top1_improvement:.6f}")
            else:
                logger.info(f"  │   └─ Average improvement: 0")
                
            logger.info(f"  ├─ Other classes upper bound reductions:")
            logger.info(f"  │   ├─ Samples improved: {samples_with_other_decrease}/{N}")
            if samples_with_other_decrease > 0:
                avg_other_decrease = total_other_yU_decrease / samples_with_other_decrease
                logger.info(f"  │   └─ Average reduction: {avg_other_decrease:.6f}")
            else:
                logger.info(f"  │   └─ Average reduction: 0")
                
            # 改進來源比例分析
            total_improvement_sources = samples_with_top1_improvement + samples_with_other_decrease
            if total_improvement_sources > 0:
                top1_contribution_rate = samples_with_top1_improvement / total_improvement_sources * 100
                other_contribution_rate = samples_with_other_decrease / total_improvement_sources * 100
                logger.info(f"  └─ Improvement source analysis:")
                logger.info(f"      ├─ Top-1 enhancement contribution: {top1_contribution_rate:.1f}%")
                logger.info(f"      └─ Other classes suppression contribution: {other_contribution_rate:.1f}%")

            logger.info(f"\n=== 指標三：驗證狀態轉變率 (Verification Status Change Rate) ===")
            logger.info(f"  ├─ Original verification status:")
            logger.info(f"  │   ├─ Safe samples: {N_safe_original}/{N} ({N_safe_original/N*100:.1f}%)")
            logger.info(f"  │   └─ Unsafe samples: {N_unsafe_original}/{N} ({N_unsafe_original/N*100:.1f}%)")
            logger.info(f"  ├─ After split verification status:")
            logger.info(f"  │   └─ Safe samples: {N_safe_split}/{N} ({N_safe_split/N*100:.1f}%)")
            logger.info(f"  ├─ Verification improvement:")
            logger.info(f"  │   ├─ Unsafe to safe conversions: {samples_unsafe_to_safe}/{N_unsafe_original}")
            
            if N_unsafe_original > 0:
                success_rate = samples_unsafe_to_safe / N_unsafe_original * 100
                logger.info(f"  │   └─ Conversion success rate: {success_rate:.1f}%")
            else:
                logger.info(f"  │   └─ Conversion success rate: N/A (no unsafe samples initially)")
                
            logger.info(f"  └─ Overall improvement rate: {(N_safe_split - N_safe_original)/N*100:.1f}%")

            # === 風險評估 ===
            if false_positive_count > 0:
                logger.warning(f"\n⚠️  RISK ASSESSMENT:")
                logger.warning(f"   └─ Potential false positive samples: {false_positive_count}/{N} ({false_positive_count/N*100:.1f}%)")
                logger.warning(f"       These samples changed from unsafe to safe - please verify manually!")
            else:
                logger.info(f"\n✅ RISK ASSESSMENT: No false positive risks detected")

            # === 總體評估 ===
            logger.info(f"\n=== 總體效能評估 ===")
            overall_effectiveness = (samples_gap_improved + samples_unsafe_to_safe) / (2 * N) * 100
            logger.info(f"  ├─ Gap reduction effectiveness: {samples_gap_improved/N*100:.1f}%")
            logger.info(f"  ├─ Safety conversion effectiveness: {samples_unsafe_to_safe/N*100:.1f}%")
            logger.info(f"  └─ Overall refinement effectiveness: {overall_effectiveness:.1f}%")

        else:
            logger.warning("No improvement data available in final split record")

def create_toy_rnn(verifier):
    with torch.no_grad():
        verifier.rnn = None
        
        verifier.a_0 = torch.tensor([0.0], dtype=torch.float32)
        
        verifier.W_ax = torch.tensor([
            [1.0],
        ], dtype=torch.float32)
        
        verifier.W_aa = torch.tensor([
            [1.0]
        ], dtype=torch.float32)
        
        verifier.W_fa = torch.tensor([
            [1.0]
        ], dtype=torch.float32)
        
        verifier.b_ax = torch.tensor([0.0], dtype=torch.float32)
        verifier.b_aa = torch.tensor([0.0], dtype=torch.float32)
        verifier.b_f = torch.tensor([0.0], dtype=torch.float32)
        
        def forward(self, X):
            with torch.no_grad():
                N = X.shape[0]
                h = torch.zeros(N, self.time_step+1, self.num_neurons, device=X.device)
                
                pre_h = torch.matmul(X[:,0,:], self.W_ax.t()) + self.b_ax
                h[:,1,:] = torch.relu(pre_h)
                
                pre_h = torch.matmul(X[:,1,:], self.W_ax.t()) + self.b_ax + torch.matmul(h[:,1,:], self.W_aa.t())
                h[:,2,:] = torch.relu(pre_h)
                
                output = torch.matmul(h[:,2,:], self.W_fa.t()) + self.b_f
                return output
        
        verifier.forward = types.MethodType(forward, verifier)

    
def main():

    parser = argparse.ArgumentParser(description='ZeroSplit Verifier for RNN robustness verification')

    parser.add_argument('--hidden-size', default=2, type=int, metavar='HS',
                        help='hidden layer size (default: 2)')
    parser.add_argument('--time-step', default=2, type=int, metavar='TS',
                        help='number of time steps (default: 2)')
    parser.add_argument('--activation', default='relu', type=str, metavar='A',
                        help='activation function: tanh or relu (default: relu)')
    parser.add_argument('--work-dir', default='../models/mnist_classifier/rnn_1_2_relu/', type=str, metavar='WD',
                        help='directory with pretrained model')
    parser.add_argument('--model-name', default='rnn', type=str, metavar='MN',
                        help='pretrained model name (default: rnn)')
    
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU')
    parser.add_argument('--cuda-idx', default=0, type=int, metavar='CI',
                        help='GPU index (default: 0)')

    parser.add_argument('--N', default=100, type=int,
                        help='number of samples (default: 5)')
    parser.add_argument('--p', default=2, type=int,
                        help='p norm (default: 2)')
    parser.add_argument('--eps', default=0.5, type=float,
                        help='perturbation epsilon (default: 0.5)')
    parser.add_argument('--max-splits', default=3, type=int,
                        help='maximum splits (default: 3)')
    parser.add_argument('--merge-results', action='store_true',
                        help='merge split results')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug output')
    parser.add_argument('--toy-rnn', action='store_true',
                        help='use toy RNN instead of MNIST model')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.cuda:
        device = torch.device(f'cuda:{args.cuda_idx}')
    else:
        device = torch.device('cpu')

    N = args.N
    p = args.p
    if p > 100:
        p = float('inf')
    eps = args.eps
    
    if args.toy_rnn:
        print("=== Toy RNN ===")
        input_size = 1
        hidden_size = 1
        output_size = 2
        time_step = 2

        verifier = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, 
                                   args.activation, max_splits=args.max_splits, debug=args.debug)
        
        create_toy_rnn(verifier)

        X = torch.tensor([
            [[1], [1]]
        ], dtype=torch.float32).to(device)

    else:
        input_size = int(28*28 / args.time_step)
        # input_size = 1
        hidden_size = args.hidden_size
        output_size = 10
        # output_size = 3
        time_step = args.time_step
        
        verifier = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, 
                                   args.activation, max_splits=args.max_splits, debug=args.debug)
        
        model_file = os.path.join(args.work_dir, args.model_name)
        verifier.load_state_dict(torch.load(model_file, map_location='cpu'))
        verifier.to(device)
        
        X, y, target_label = sample_mnist_data(
            N=N, seq_len=time_step, device=device,
            data_dir='../data/mnist', train=False, shuffle=True, rnn=verifier
        )
        # X_train, y_train, target_train_label, _, _, _  = prepare_stock_tensors_split(
        #     csv_path='C:/Users/zxczx/POPQORN/vanilla_rnn/utils/A1_bin.csv',
        #     window_size=time_step,
        #     train_ratio=0.8,
        #     device=device
        # )
        # # 只取前N個樣本(到target_label是新資料)
        # X = X_train[:N]
        # y = y_train[:N]
        # target_label = target_train_label[:N]
        # print(f"Sampled {N} data points with shape: {X.shape}")
        # print(f"X: {X}")
        
        verifier.extractWeight(clear_original_model=False)

    # 設置結果保存和logger捕獲
    json_file, txt_file, log_capture, session_dir = setup_result_logging(
        args.work_dir, args.eps, args
    )
    
    logger.info(f"Starting ZeroSplit verification")
    logger.info(f"Model: {args.work_dir}")
    logger.info(f"Config: eps={args.eps}, N={args.N}, max_splits={args.max_splits}")
    logger.info(f"Results will be saved to: {session_dir}")

    logger.info(f"\n=== Zero Split Verification (merge_results={args.merge_results}) ===")
    # is_verified, unsafe_layer, top1_class, yL_out, yU_out = verifier.verify_network(
    #     X, eps, merge_results=args.merge_results
    # )
    is_verified, unsafe_layer, top1_class = verifier.verify_network_recursive(
        X, eps, max_splits=args.max_splits
    )

    # 準備結果數據
    results_data = {
        'is_verified': is_verified,
        'unsafe_layer': unsafe_layer,
        'top1_class': top1_class.tolist() if hasattr(top1_class, 'tolist') else int(top1_class),
        'split_count': verifier.split_count
    }
    
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Predicted class: {top1_class}")
    if is_verified:
        logger.info(f"Verification successful!")
        logger.info(f"Split count: {verifier.split_count}")
    else:
        logger.info(f"Verification failed")
        if unsafe_layer:
            logger.info(f"Unsafe layer: {unsafe_layer}")
            logger.info(f"Split count: {verifier.split_count}")
    
    # 保存結果（包含所有captured logs）
    save_verification_results(json_file, txt_file, log_capture.captured_logs, args, results_data, session_dir)
    
    logger.info(f"\nResults saved to session: {os.path.basename(session_dir)}")
    logger.info(f"JSON detail: {os.path.basename(json_file)}")
    logger.info(f"TXT summary: {os.path.basename(txt_file)}")

if __name__ == "__main__":
    main()