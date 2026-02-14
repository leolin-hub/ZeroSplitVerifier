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
from collections import Counter
from pathlib import Path
import time

import multiprocessing as mp
from functools import partial

import torch
from torchvision import datasets, transforms
from utils.sample_data import sample_mnist_data
from utils.sample_stock_data import prepare_stock_tensors_split
from utils.sample_seq_mnist import sample_seq_mnist_data
from utils.sample_cifar10 import sample_cifar10_data

import get_bound_for_general_activation_function as get_bound
from locate_timestep_shap import compute_shap_ranking_once, select_timestep_from_shap

class Timer:
    """輕量 context manager，累積時間到 timing_stats dict。"""
    def __init__(self, stats: dict, key: str):
        self.stats = stats
        self.key = key

    def __enter__(self):
        self._t = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._t
        entry = self.stats.setdefault(self.key, {'total': 0.0, 'count': 0})
        entry['total'] += elapsed
        entry['count'] += 1

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
    dataset_name = Path(args.work_dir).parent.name.split('_classifier')[0]
    
    # 建立時間子資料夾 - 在當前目錄(vanilla_rnn)下建立verification_results
    current_dir = os.path.dirname(os.path.abspath(__file__))  # vanilla_rnn目錄
    session_dir = os.path.join(current_dir, "verification_results", f"session_{dataset_name}_{args.mode}_{args.activation}_{args.time_step}_{args.hidden_size}_eps{eps_value}_N{args.N}_p{args.p}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 建立本次實驗的檔案名稱
    base_name = f"zerosplit__{dataset_name}_{args.mode}_{args.activation}_timestep{args.time_step}_hidden{args.hidden_size}_eps{eps_value}_N{args.N}_maxsplit{args.max_splits}"
    json_file = os.path.join(session_dir, f"{base_name}.json")
    txt_file = os.path.join(session_dir, f"{base_name}.txt")
    
    # 設置logger捕獲
    log_capture = LoggerCapture()
    
    # 添加捕獲handler到logger
    logger.add(log_capture.capture_handler, level="INFO", format="{message}")
    
    return json_file, txt_file, log_capture, session_dir

def serialize_split_tree(tree_node):
    if tree_node is None:
        return None
    
    serialized = {}
    for key, value in tree_node.items():
        if key in ['pos_bounds', 'neg_bounds', 'path']:
            # Skip torch tensors and redundant fields
            continue
        elif key in ['pos_subtree', 'neg_subtree']:
            # Recursively process subtrees
            serialized[key] = serialize_split_tree(value)
        else:
            # Other fields are copied directly
            serialized[key] = value
    
    return serialized

def save_verification_results(json_file, txt_file, captured_logs, args, results_data, session_dir, verifier):
    """保存驗證結果，重點保存logger分析資訊"""
    
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
            'max_splits': args.max_splits,
            'mode': args.mode,
        },
        'verification_results': results_data,
        
        # 改用 sample_records（新增）
        'sample_records': [
            {
                'sample_id': r['sample_id'],
                'popqorn_verified': r.get('popqorn_verified', None),
                'is_verified': r['is_verified'],
                'total_splits': r['total_splits'],
                'num_leaves': r['num_leaves'],
                'all_leaves_verified': r['all_leaves_verified'],
                'improvements': r.get('improvements'),
                'split_timesteps': r.get('split_timesteps', []),  # 新增
                'pq_failed_zs_success': r.get('pq_failed_zs_success', False),  # 新增
                'shap_values': r.get('shap_vals', None),  # 新增
                'split_times': r.get('split_times', []),
                # 'split_tree_summary': serialize_split_tree(r.get('split_tree'))
                # split_tree 和 bounds 太大，可選擇性保存
            }
            for r in getattr(verifier, 'sample_records', [])
        ] if hasattr(verifier, 'sample_records') else [],
        
        'all_logs': [log for log in captured_logs],
        'timing_stats': {
            k: {
                'total_sec': round(v['total'], 4),
                'count': v['count'],
                'avg_sec': round(v['total'] / v['count'], 6) if v['count'] > 0 else 0,
            }
            for k, v in getattr(verifier, 'timing_stats', {}).items()
        } if args.mode == 'shap' else {},
    }
    
    # 保存JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=None, ensure_ascii=False) # indent=2改None
    
    # 保存TXT摘要 - 重點整理bounds analysis
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"ZeroSplit Verification Analysis Report\n")
        f.write(f"{'='*80}\n")
        f.write(f"Session: {os.path.basename(session_dir)}\n")
        f.write(f"Timestamp: {save_data['experiment_info']['timestamp']}\n")
        f.write(f"Model: {args.work_dir}\n")
        f.write(f"Parameters: hidden_size={args.hidden_size}, time_step={args.time_step}, activation={args.activation}\n")
        f.write(f"Test config: eps={args.eps}, p={args.p}, N={args.N}, max_splits={args.max_splits}, mode={args.mode}\n\n")
        
        # Sample-level 結果
        if save_data['sample_records']:
            f.write(f"Sample-Level Results:\n")
            f.write(f"{'-'*40}\n")
            verified_count = sum(1 for r in save_data['sample_records'] if r['is_verified'])
            f.write(f"Verified samples: {verified_count}/{len(save_data['sample_records'])}\n")
            f.write(f"Average splits: {sum(r['total_splits'] for r in save_data['sample_records']) / len(save_data['sample_records']):.2f}\n")
        

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
        self.split_history = {}
        self.timing_stats = {}
        # 保存原始POPQORN的l, u
        self.original_l = None
        self.original_u = None
        # 初始化early stopping flag
        self._early_stop = False
        
    def _compute_basic_bounds(self, eps, p, v, X, N, s, n, idx_eps):
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
            # torch.set_printoptions(
            #     threshold=float('inf'),  # 顯示所有元素
            #     linewidth=200,          # 每行字符數
            #     edgeitems=10           # 邊緣顯示的元素數
            # )
            # print(f"Wax: {W_ax}")
            # print(f"Norm: {torch.norm(W_ax, p=q, dim=2)}")
                
            # 第二項: A^{<v>} W_ax x^{<v>} and Ou^{<v>} W_ax x^{<v>}
            if v == 1:
                X = X.view(N, 1, n)  # 確保維度正確           
            yU = yU + torch.matmul(W_ax, X[:, v-1, :].view(N, n, 1)).squeeze(2)  # [N, s]
            yL = yL + torch.matmul(W_ax, X[:, v-1, :].view(N, n, 1)).squeeze(2)  # [N, s]
            
            # 第三項: A^{<v>} (b_a + Delta^{<v>}) and Ou^ {<v>} (b_a + Theta^{<v>})
            yU = yU + b_aa + b_ax  # [N, s]
            yL = yL + b_aa + b_ax  # [N, s]
            
            if self.debug:
                print(f"\nCurrent Timestep {v} input impact:")
                print(f"Current Timestep {v} input lower: {yL}")
                print(f"Current Timestep {v} input upper: {yU}")
            
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
                           return_refine_preact=False, refine_preh=None, use_intersection=True):

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

            final_l = yL
            final_u = yU
            if refine_preh is not None and refine_preh_l[v] is not None and use_intersection:
                # 交集：取最嚴格的 bound
                final_l = torch.maximum(yL, refine_preh_l[v])
                final_u = torch.minimum(yU, refine_preh_u[v])

            self.l[v] = final_l
            self.u[v] = final_u

            if refine_preh is not None:
                # 逐步運用先前refineh來更新後面timestep的hidden state
                refine_preh_l[v] = final_l.clone().detach()
                refine_preh_u[v] = final_u.clone().detach()

            # 檢查是否需要在current timestep進行split
            if unsafe_layer == v and not split_done:
                if not merge_results:
                    # 不合併結果，分別計算兩個子問題
                    result = self._split_and_compute_separate(eps, p, v, Eps_idx, cross_zero, return_refine_preact)
                    
                    return result
                else:
                    # 合併結果
                    return self._split_and_merge(eps, p, v, Eps_idx, cross_zero)

            return final_l, final_u

    def _split_and_compute_separate(self, eps, p, v, Eps_idx, cross_zero=None, return_refine_preact=False):
        """分割當前層並分別計算兩個子問題"""
        # 保存原始bounds
        orig_l = self.l[v].clone().detach()
        orig_u = self.u[v].clone().detach()
        full_X = self.original_X.clone().detach()
        # print(f"full_X shape: {full_X.shape}")
        
        if self.debug:
            print(f"Splitting layer {v}, cross_zero count: {cross_zero.sum().item()}")
        
        # 結果儲存
        pos_yL, pos_yU = None, None
        neg_yL, neg_yU = None, None
        
        # 正區間: x >= 0
        pos_l = orig_l.clone().detach()
        pos_u = orig_u.clone().detach()
        pos_l[cross_zero] = 0

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
                    eps, p, k, X=full_X[:, 0:k, :], Eps_idx=Eps_idx,
                    refine_preh=(pos_l_state, pos_u_state),
                    use_intersection=False
                )
                # print(f"第 {k} timestep的pre-activation bounds: 正區間下界: {pos_yL}, 正區間上界: {pos_yU}")
            
            # 計算輸出層
            pos_yL, pos_yU = self.computeLast2sideBound(
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"正區間的final bounds: {pos_yL}, {pos_yU}")
            # print(f"正區間的bounds差異: {pos_yU - pos_yL}")
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
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"正區間的final bounds: {pos_yL}, {pos_yU}")
            # print(f"正區間的bounds差異: {pos_yU - pos_yL}")
        
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
                    eps, p, k, X=full_X[:, 0:k, :], Eps_idx=Eps_idx,
                    refine_preh=(neg_l_state, neg_u_state),
                    use_intersection=False
                ) # eps換eps_neg
                # print(f"第 {k} timestep的pre-activation bounds: 負區間下界: {neg_yL}, 負區間上界: {neg_yU}")
            
            # 計算輸出層
            neg_yL, neg_yU = self.computeLast2sideBound(
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"負區間的final bounds: {neg_yL}, {neg_yU}")
            # print(f"負區間的bounds差異: {neg_yU - neg_yL}")
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
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"負區間的final bounds: {neg_yL}, {neg_yU}")
            # print(f"負區間的bounds差異: {neg_yU - neg_yL}")
        
        # 恢復原始bounds
        # self.l[v] = orig_l
        # self.u[v] = orig_u
        
        # 返回兩個子問題的最終結果
        if return_refine_preact:
            return (pos_yL, pos_yU), (neg_yL, neg_yU), \
                   ((pos_l_state, pos_u_state)), \
                   ((neg_l_state, neg_u_state))
        else:
            return (pos_yL, pos_yU), (neg_yL, neg_yU)

    def _split_and_merge(self, eps, p, v, Eps_idx, cross_zero=None):
        """分割當前層並合併兩個子問題的結果"""
        # 獲取分別計算的結果
        (pos_yL, pos_yU), (neg_yL, neg_yU) = self._split_and_compute_separate(
            eps, p, v, Eps_idx, cross_zero=cross_zero
        )

        yL = torch.minimum(pos_yL, neg_yL)
        yU = torch.maximum(pos_yU, neg_yU)

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
            with Timer(self.timing_stats, 'getConvenientGeneralActivationBound'):
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
        # print(f"不做split的隱藏層bounds to verify: {yL}, {yU}")
        
        yL_out, yU_out = self.computeLast2sideBound(eps, p=2, v=self.time_step+1,
                                                X=X, Eps_idx=torch.arange(1,self.time_step+1))
        
        # print(f"不做split的輸出層 bounds to verify: {yL_out}, {yU_out}")
        # print(f"First time difference between yL and yU: {yU_out - yL_out}")

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
    
    def _find_max_verifiable_eps(self, X, top1_class, p, eps_range=(0.01, 0.015),
                                 precision=0.001, max_splits=None, mode=None,
                                 sample_id=0):
        """
        Unified binary search: each iteration runs POPQORN then ZeroSplit at same eps.
        Early terminates when PQ and ZS disagree (reveals EVR ordering).
        
        Returns:
            (comparison_flag, last_zs_result, last_shap_vals)
            comparison_flag: 'zs_better' | 'pq_better' | 'equal'
        """
        low, high = eps_range

        if isinstance(top1_class, torch.Tensor):
            top1_class_val = top1_class.item()
        else:
            top1_class_val = top1_class

        top1_tensor = torch.tensor([top1_class_val])
        last_result = None
        last_shap_vals = None

        # for loop
        eps_values = []
        current_eps = low
        while current_eps <= high:
            eps_values.append(current_eps)
            current_eps += precision
        
        # while high - low > precision:
        for eps in eps_values:
            # mid = (low + high) / 2 # bs

            # Phase 1: POPQORN
            pq_verified, _, _, _ = self.verify_robustness(X, eps) # from mid to eps
            
            # Save original bounds for recursive split
            self.original_l = [b.clone() if b is not None else None for b in self.l]
            self.original_u = [b.clone() if b is not None else None for b in self.u]

            # Phase 2: SHAP
            selected_neurons = None
            shap_vals = None
            if mode == 'shap':
                # eps太小會導致沒有neuron跨越0
                selected_neurons = compute_shap_ranking_once(self, X, top1_tensor, eps, p, top_k_neurons=5) # from mid to eps
                shap_vals = [[t, n, round(float(imp), 6)] for t, n, imp in selected_neurons]
                last_shap_vals = shap_vals

            # Phase 3: ZeroSplit refinement
            self._early_stop = False
            result = self._recursive_split_verify(
                X, eps, top1_tensor, p,
                split_count=0, max_splits=max_splits,
                start_timestep=1, refine_preh=None, mode=mode,
                sample_id=sample_id, path=[], selected_neurons=selected_neurons # from mid to eps
            )
            zs_verified = result['is_verified']
            last_result = result

            logger.info(f"  Sample {sample_id+1}: eps={eps:.4f}, PQ={pq_verified}, ZS={zs_verified}")

            # Early termination on disagreement
            if not pq_verified and zs_verified:
                logger.info(f"  Sample {sample_id+1}: ZS better (PQ=F, ZS=T at eps={eps:.4f})")
                return 'zs_better', result, shap_vals
            if pq_verified and not zs_verified:
                logger.info(f"  Sample {sample_id+1}: PQ better (PQ=T, ZS=F at eps={eps:.4f})")
                return 'pq_better', result, shap_vals

            # Agreement: continue binary search
            # if pq_verified and zs_verified:
            #     low = mid
            # else: # 代表eps太大
            #     high = mid

        # logger.info(f"  Sample {sample_id+1}: Converged without disagreement, equal")
        logger.info(f"  Sample {sample_id+1}: Completed all eps values without disagreement, equal")
        return 'equal', last_result, last_shap_vals

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
        
    @staticmethod
    def _process_single_sample_worker(args):
        """
        Worker for parallel processing. Must be static/picklable.
        Args: (sample_id, X_i, model_state, eps, p, max_splits, mode,
               input_size, hidden_size, output_size, time_step, activation)
        """
        if len(args) == 11:
            (sample_id, X_i, model_state, eps, p, max_splits, mode,
            input_size, hidden_size, output_size, time_step, activation) = args
            use_evr = False
            eps_range = (0.001, 1.0)
        else:
            (sample_id, X_i, model_state, eps, p, max_splits, mode,
            input_size, hidden_size, output_size, time_step, activation, use_evr, eps_range) = args
        
        shap_vals = None
        
        # Create worker verifier instance
        worker_verifier = ZeroSplitVerifier(
            input_size, hidden_size, output_size, time_step, activation, max_splits=max_splits, debug=False
        )
        worker_verifier.load_state_dict(model_state, strict=False)
        worker_verifier.extractWeight(clear_original_model=False)

        # 獲取top1預測類別（EVR和固定epsilon模式都需要）
        with torch.no_grad():
            output = worker_verifier(X_i)
            top1_i = output.argmax(dim=1)

        if use_evr:
            # EVR mode: unified PQ+ZS binary search
            logger.info(f"Sample {sample_id+1}: EVR mode - unified PQ+ZS search")

            comparison, last_result, shap_vals = worker_verifier._find_max_verifiable_eps(
                X_i, top1_i, p, eps_range=eps_range,
                max_splits=max_splits, mode=mode, sample_id=sample_id
            )
            
            logger.info(f"Sample {sample_id+1}: Result = {comparison}")

            result = {
                'sample_id': sample_id,
                'is_verified': True,
                'comparison': comparison,
                'total_splits': last_result.get('total_splits', 0) if last_result else 0,
                'split_tree': last_result.get('split_tree') if last_result else None,
                'top1_class': top1_i,
                'shap_vals': shap_vals,
            }

            return {
                'sample_id': sample_id,
                'result': result,
                'orig_bounds': None,
                'popqorn_verified': None,
                'shap_vals': shap_vals,
                'timing_stats': worker_verifier.timing_stats,
            }

        # Phase 1: POPQORN
        with Timer(worker_verifier.timing_stats, 'verify_robustness'):
            is_verified_pq, _, yL_out_i, yU_out_i = worker_verifier.verify_robustness(X_i, eps)

        orig_bounds = {
            'yL': yL_out_i.clone(),
            'yU': yU_out_i.clone(),
            'original_l': [b.clone() if b is not None else None for b in worker_verifier.l],
            'original_u': [b.clone() if b is not None else None for b in worker_verifier.u]
        }

        # Phase 2: Refinement (拿掉for連成功樣本都處理)
        # if is_verified_pq:
        #     # 只處理失敗的樣本
        #     logger.info(f"Sample {sample_id+1} already verified by POPQORN, skipping refinement")
        #     result = {
        #         'sample_id': sample_id,
        #         'is_verified': True,
        #         'total_splits': 0,
        #         'split_tree': None,
        #         'all_leaf_bounds': [([], yL_out_i.clone(), yU_out_i.clone())],
        #         'popqorn_only': True,
        #         'top1_class': top1_i,
        #         'first_unsafe_layer': None,
        #         'first_unsafe_layer_method': None,
        #         'popqorn_verified': is_verified_pq
        #     }

        #     return {
        #         'sample_id': sample_id,
        #         'result': result,
        #         'orig_bounds': orig_bounds,
        #         'popqorn_verified': is_verified_pq,
        #         'shap_vals': None
        #     }

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing SAMPLE {sample_id+1}")
        logger.info(f"{'='*80}")
        logger.info(f"Sample {sample_id+1} failed POPQORN, starting refinement")

        # Compute SHAP ranking for this sample based on mode
        if mode == 'shap':
            logger.info(f"Computing SHAP ranking for sample {sample_id+1}...")
            with Timer(worker_verifier.timing_stats, 'compute_shap_ranking_once'):
                selected_neurons = compute_shap_ranking_once(worker_verifier, X_i, top1_i, eps, p) # (timestep, neuron_index, importance)
            logger.info(f"SHAP selected {len(selected_neurons)} neurons for splitting")
            shap_vals = [[t, n, round(float(imp), 6)] for t, n, imp in selected_neurons]
        elif mode.startswith('last_'): # e.g., last_3 取最後 layer
            # Extract k from last_k
            n_last = int(mode.split('_')[1])
            total_timesteps = worker_verifier.time_step
            selected_neurons = [(t, n) for t in range(total_timesteps - n_last + 1, total_timesteps + 1) for n in range(worker_verifier.num_neurons)]
            logger.info(f"Using last {n_last} timesteps: {selected_neurons}")
            shap_vals = None
        
        # 設置 worker 的 original bounds（用於 _recursive_split_verify 內部使用）
        worker_verifier.orig_output_bounds_per_sample = [orig_bounds]
        worker_verifier.original_l = orig_bounds['original_l']
        worker_verifier.original_u = orig_bounds['original_u']

        result = worker_verifier._recursive_split_verify(
            X_i, eps, top1_i, p, split_count=0, max_splits=max_splits,
            start_timestep=1, refine_preh=None, mode=mode,
            sample_id=sample_id, path=[],
            selected_neurons=selected_neurons
        )
        result['sample_id'] = sample_id
        result['top1_class'] = top1_i
        result['popqorn_verified'] = is_verified_pq

        return {
            'sample_id': sample_id,
            'result': result,
            'orig_bounds': orig_bounds,
            'popqorn_verified': is_verified_pq,
            'shap_vals': shap_vals,
            'timing_stats': worker_verifier.timing_stats,
        }

    def verify_network_recursive(self, X, eps, p, max_splits=3, mode='critical', n_workers=None, use_evr=False, eps_range=(0.001, 1.0)):
        """整體的驗證流程，使用recursive split and verify方法"""
        self._early_stop = False # 重置全局標誌
        self.original_X = X.clone().detach()
        N = X.shape[0]

        self.orig_output_bounds_per_sample = []

        # 初始化記錄所有 split
        if not hasattr(self, 'sample_records'):
            self.sample_records = []

        # Determine number of workers
        if n_workers is None:
            n_workers = mp.cpu_count()

        logger.info(f"=== Starting Verification with {n_workers} workers ===")

        # Prepare arguments for each sample
        model_state = self.state_dict()
        worker_args = [
            (i, X[i:i+1], model_state, eps, p, max_splits, mode,
             self.input_size, self.num_neurons, self.output_size, self.time_step, self.activation, use_evr, eps_range)
            for i in range(N)
        ]

        # Parallel processing using multiprocessing Pool
        # Phase 1: 每個樣本獨立跑 POPQORN
        logger.info("=== Phase 1: POPQORN Bounds Computation ===")
        
        if n_workers > 1:
            with mp.Pool(processes=n_workers) as pool:
                worker_results = pool.map(self._process_single_sample_worker, worker_args)
        else:
            # Serial fallback (for debugging)
            worker_results = [self._process_single_sample_worker(args) for args in worker_args]

        worker_results = sorted(worker_results, key=lambda x: x['sample_id'])

        # Main Process - Collect results
        popqorn_safe_count = 0
        sample_results = []

        for worker_output in worker_results:
            sample_id = worker_output['sample_id']
            result = worker_output['result']
            orig_bounds = worker_output['orig_bounds']
            result['shap_vals'] = worker_output.get('shap_vals', None)

            # Save original bounds
            self.orig_output_bounds_per_sample.append(orig_bounds)
            sample_results.append(result)

            if worker_output['popqorn_verified']:
                popqorn_safe_count += 1

            self.record_sample_split_tree(result)

            # 聚合 worker timing
            for key, val in worker_output.get('timing_stats', {}).items():
                entry = self.timing_stats.setdefault(key, {'total': 0.0, 'count': 0})
                entry['total'] += val['total']
                entry['count'] += val['count']

        logger.info(f"\n{'='*80}")
        logger.info(f"Parallel processing completed")

        if use_evr:
            # EVR summary
            comparisons = Counter(r.get('comparison', 'unknown') for r in sample_results)
            
            logger.info(f"EVR Comparison Summary:")
            logger.info(f"  ZS better: {comparisons.get('zs_better', 0)}/{N}")
            logger.info(f"  PQ better: {comparisons.get('pq_better', 0)}/{N}")
            logger.info(f"  Equal: {comparisons.get('equal', 0)}/{N}")
            logger.info(f"{'='*80}")

            all_top1 = torch.stack([r['top1_class'] for r in sample_results])
            return True, None, all_top1, popqorn_safe_count

        logger.info(f"POPQORN Summary: {popqorn_safe_count}/{N} samples verified")
        logger.info(f"{'='*80}")

        # 檢查 early exit case
        if popqorn_safe_count == N:
            logger.info("All samples verified by POPQORN, no refinement needed")
            self.generate_final_report()
            all_top1 = torch.stack([r['top1_class'] for r in sample_results])
            return True, None, all_top1, popqorn_safe_count
        
        # 輸出最終報告
        self.generate_final_report()

        # 彙總結果
        all_verified = all(r['is_verified'] for r in sample_results)
        total_verified = sum(r['is_verified'] for r in sample_results)
        all_top1 = torch.stack([r['top1_class'] for r in sample_results])
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Samples verified: {total_verified}/{N}")
        logger.info(f"Overall verification: {'SUCCESS' if all_verified else 'FAILED'}")
        return all_verified, None, all_top1, popqorn_safe_count

    def _recursive_split_verify(self, X, eps, top1_class, p, split_count, max_splits, start_timestep=1, refine_preh=None, mode='max_violation',
                                sample_id=None, path=[], first_unsafe_layer=None, locate_method=None, selected_neurons=None,
                                split_history=None):
        """
        採用worst case merge驗證
        流程：不切計算output bounds -> 找unsafe layer -> 切完後先驗證pos/neg -> 至少一個成功則做union output bounds檢查 -> 失敗則對pos/neg區間繼續切
        """

        split_start_time = time.time()

        # 檢查全局提前終止標誌
        if self._early_stop:
            split_elapsed = time.time() - split_start_time
            self.timing_stats.setdefault('recursive_split_verify', {'total': 0.0, 'count': 0})['total'] += split_elapsed
            self.timing_stats['recursive_split_verify']['count'] += 1
            logger.info(f"  Sample {sample_id+1}: The verification has failed (split_count={split_count})")
            return {
                'sample_id': sample_id,
                'is_verified': False,
                'split_tree': None,
                'all_leaf_bounds': [],
                'total_splits': split_count,
                'termination_reason': 'early_stop_triggered',
                'first_unsafe_layer': first_unsafe_layer,
                'first_unsafe_layer_method': locate_method,
                'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                    for item in (split_history or [])],
                'split_times': [(split_count, split_elapsed, sample_id)]
            }

        self.original_X = X.clone().detach()

        Eps_idx = torch.arange(1, self.time_step + 1)

        if split_history is None:
            split_history = []

        if refine_preh is not None:
            l_state, u_state = refine_preh
            for i in range(len(l_state)):
                if l_state[i] is not None:
                    self.l[i] = l_state[i].clone()
                if u_state[i] is not None:
                    self.u[i] = u_state[i].clone()

            if start_timestep < self.time_step:
                for k in range(start_timestep + 1, self.time_step + 1):
                    with Timer(self.timing_stats, 'compute2sideBound'):
                        yL, yU = self.compute2sideBound(
                            eps, p, k, X=X[:, 0:k, :], Eps_idx=Eps_idx,
                            refine_preh=(l_state, u_state),
                            use_intersection=False
                        )

            with Timer(self.timing_stats, 'computeLast2sideBound'):
                yL_out, yU_out = self.computeLast2sideBound(
                    eps, p, v=self.time_step+1, X=X, Eps_idx=Eps_idx
                )

        else:
            with Timer(self.timing_stats, 'verify_robustness'):
                _, _, yL_out, yU_out = self.verify_robustness(X, eps)

        current_verified = self._check_region_robust(yL_out, yU_out, top1_class)

        if current_verified and split_count > 0:
            # 當前region驗證成功 -> 這是一個verified leaf node
            split_elapsed = time.time() - split_start_time
            self.timing_stats.setdefault('recursive_split_verify', {'total': 0.0, 'count': 0})['total'] += split_elapsed
            self.timing_stats['recursive_split_verify']['count'] += 1
            # split_times.append((split_count, split_elapsed, sample_id))

            logger.info(f"  Sample {sample_id+1}: Current region VERIFIED at split_count={split_count}")
            return {
                'sample_id': sample_id,
                'is_verified': True,
                'split_tree': None,
                'all_leaf_bounds': [(path, yL_out.clone(), yU_out.clone())],
                'total_splits': split_count,
                'termination_reason': 'verified_without_further_split',
                'first_unsafe_layer': first_unsafe_layer,
                'first_unsafe_layer_method': locate_method,
                'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                for item in split_history], 
                'split_times': [(split_count, split_elapsed, sample_id)], # new
            }

        # split_count=0且POPQORN成功時，繼續往下執行split
        if current_verified and split_count == 0:
            logger.info(f"  Sample {sample_id+1}: POPQORN verified at eps={eps}, but continuing to refine")

        # 終止條件判斷
        # if split_count >= max_splits:
        #     split_elapsed = time.time() - split_start_time
        #     # split_times.append((split_count, split_elapsed, sample_id))

        #     logger.info(f"  Sample {sample_id+1}: Reached max_splits={max_splits} but NOT verified, FAILED")
        #     return {
        #         'sample_id': sample_id,
        #         'is_verified': False,
        #         'split_tree': None,
        #         'all_leaf_bounds': [(path, yL_out.clone(), yU_out.clone())],
        #         'total_splits': split_count,
        #         'termination_reason': 'max_splits_reached',
        #         'first_unsafe_layer': first_unsafe_layer,
        #         'first_unsafe_layer_method': locate_method,
        #         'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
        #         for item in split_history], # new
        #         'split_times': [(split_count, split_elapsed, sample_id)], # new
        #     }
        
        if (mode == 'shap' or mode.startswith('last_')) and selected_neurons is not None: # 加入mode = last_方便測試
            with Timer(self.timing_stats, 'select_timestep_from_shap'):
                unsafe_layer, unsafe_neuron, cross_zero = select_timestep_from_shap(
                    self, selected_neurons, start_timestep, refine_preh, split_history, sample_id
                )

            if split_count == 0 and unsafe_layer is not None:
                first_unsafe_layer = [unsafe_layer, unsafe_neuron]
                locate_method = "shap"
                logger.info(f"  Sample {sample_id+1}: First unsafe (t, n) (SHAP): t={unsafe_layer}, neuron={unsafe_neuron+1}")

            if unsafe_layer is not None:
                logger.info(f"  Sample {sample_id+1}: SHAP selected t={unsafe_layer}, n={unsafe_neuron+1} (start_timestep={start_timestep})")
        else:
            assert mode == 'shap', 'Only SHAP mode is supported'

        if unsafe_layer is None:
            # 驗證失敗但找不到unsafe layer -> 無法繼續切分，這是一個unverified leaf node
            split_elapsed = time.time() - split_start_time
            self.timing_stats.setdefault('recursive_split_verify', {'total': 0.0, 'count': 0})['total'] += split_elapsed
            self.timing_stats['recursive_split_verify']['count'] += 1
            # split_times.append((split_count, split_elapsed, sample_id))

            logger.info(f"  Sample {sample_id+1}: NOT verified and no unsafe layer found at split_count={split_count}, FAILED")
            return {
                'sample_id': sample_id,
                'is_verified': False,  # RETURN FALSE
                'split_tree': None,
                'all_leaf_bounds': [(path, yL_out.clone(), yU_out.clone())],
                'total_splits': split_count,
                'termination_reason': 'no_unsafe_layer',
                'first_unsafe_layer': first_unsafe_layer,
                'first_unsafe_layer_method': locate_method,
                'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                for item in split_history], # new
                'split_times': [(split_count, split_elapsed, sample_id)], # new
            }

        logger.info(f"  Sample {sample_id+1}: Found unsafe layer {unsafe_layer}, splitting...")
        logger.info(f"  Path: {path} -> splitting layer {unsafe_layer}")

        # 在unsafe layer split為正負兩區域 (Input取到unsafe_layer)
        pos_bounds, neg_bounds, pos_preact, neg_preact = self.compute2sideBound(
            eps, p=p, v=unsafe_layer, X=X[:, 0:unsafe_layer, :],
            Eps_idx=torch.arange(1, self.time_step + 1),
            unsafe_layer=unsafe_layer, merge_results=False, cross_zero=cross_zero,
            return_refine_preact=True, refine_preh=refine_preh
        )

        (pos_yL_out, pos_yU_out) = pos_bounds
        (neg_yL_out, neg_yU_out) = neg_bounds
        (pos_bounds_state) = pos_preact
        (neg_bounds_state) = neg_preact
        # torch.set_printoptions(
        #     threshold=float('inf'),  # 顯示所有元素
        #     linewidth=200,          # 每行字符數
        #     edgeitems=10           # 邊緣顯示的元素數
        # )
        # print(f"正區間的preactivation: {pos_bounds_state}")
        # print(f"負區間的preactivation: {neg_bounds_state}")

        # ==== Process 1: 各自驗證 ====
        logger.info(f"\n=== Phase 1: Subproblem Verification ===")

        pos_verified = self._check_region_robust(pos_yL_out, pos_yU_out, top1_class)
        neg_verified = self._check_region_robust(neg_yL_out, neg_yU_out, top1_class)

        logger.info(f"  Split {split_count+1} at layer {unsafe_layer}:")
        logger.info(f"    Positive region: {'SAFE' if pos_verified else 'UNSAFE'}")
        logger.info(f"    Negative region: {'SAFE' if neg_verified else 'UNSAFE'}")
        
        # 建立目前split節點
        current_split = {
            'split_count': split_count + 1,
            'timestep': unsafe_layer,
            'path': path.copy(),
            'pos_bounds': (pos_yL_out.clone(), pos_yU_out.clone()),
            'neg_bounds': (neg_yL_out.clone(), neg_yU_out.clone()),
            'pos_verified': pos_verified,
            'neg_verified': neg_verified,
        }

        split_elapsed = time.time() - split_start_time
        self.timing_stats.setdefault('recursive_split_verify', {'total': 0.0, 'count': 0})['total'] += split_elapsed
        self.timing_stats['recursive_split_verify']['count'] += 1
        current_record = [(split_count + 1, split_elapsed, sample_id)]
        
        if mode == 'shap' and unsafe_layer is not None and unsafe_neuron is not None:
            new_split_history = split_history + [(unsafe_layer, unsafe_neuron)]
        else:
            new_split_history = split_history

        if pos_verified and neg_verified:
            # split_elapsed = time.time() - split_start_time # 思考要不要拿掉
            # split_times.append((split_count + 1, split_elapsed, sample_id))

            logger.info(f"  Sample {sample_id+1}: Both regions verified, SUCCESS at split {split_count+1}")
            current_split['both_verified'] = True
            current_split['is_leaf'] = True

            return {
                'sample_id': sample_id,
                'is_verified': True,
                'split_tree': current_split,
                'all_leaf_bounds': [
                    (path + [(unsafe_layer, unsafe_neuron, 'pos')], pos_yL_out.clone(), pos_yU_out.clone()),
                    (path + [(unsafe_layer, unsafe_neuron, 'neg')], neg_yL_out.clone(), neg_yU_out.clone())
                ],
                'total_splits': split_count + 1,
                'first_unsafe_layer': first_unsafe_layer,
                'first_unsafe_layer_method': locate_method,
                'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                for item in new_split_history], # new
                'split_times': [(split_count+1, split_elapsed, sample_id)] # new
            }
        # 左子樹切到極限但還是驗證失敗
        if (not pos_verified or not neg_verified) and split_count + 1 >= max_splits:
            self._early_stop = True # 設置全局提前終止標誌
            logger.info(f"  Sample {sample_id+1}: One or both regions FAILED as max_splits reached, early termination")
            current_split['is_leaf'] = True

            return {
                'sample_id': sample_id,
                'is_verified': False,
                'split_tree': current_split,
                'all_leaf_bounds': [
                    (path + [(unsafe_layer, unsafe_neuron, 'pos')], pos_yL_out.clone(), pos_yU_out.clone()),
                    (path + [(unsafe_layer, unsafe_neuron, 'neg')], neg_yL_out.clone(), neg_yU_out.clone())
                ],
                'total_splits': split_count + 1,
                'termination_reason': 'max_splits_reached_with_failure',
                'first_unsafe_layer': first_unsafe_layer,
                'first_unsafe_layer_method': locate_method,
                'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                        for item in new_split_history],
                'split_times': current_record
            }


        # Fix Point: Allow stay in the same layer if needed
        next_start_timestep = unsafe_layer
        # ==== Process 2: 對子區間進行遞迴切割 ====
        logger.info(f"  Sample {sample_id+1}: Need further splitting...")
        # 驗證並遞迴正區間
        if not pos_verified and not self._early_stop:
            logger.info(f"  Entering positive branch...")
            pos_result = self._recursive_split_verify(
                X, eps, top1_class, p, split_count + 1, max_splits,
                start_timestep=next_start_timestep, refine_preh=pos_bounds_state, mode=mode, # 將start_timestep從unsafe_layer + 1改next_start_timestep
                sample_id=sample_id, path=path + [(unsafe_layer, unsafe_neuron, 'pos')],
                first_unsafe_layer=first_unsafe_layer,
                locate_method=locate_method,
                selected_neurons=selected_neurons,
                split_history=new_split_history
            )
        else:
            pos_result = {
                'sample_id': sample_id,
                'is_verified': pos_verified,
                'split_tree': None,
                'all_leaf_bounds': [],
                'total_splits': split_count + 1,
                'split_times': []
            }          

        # 驗證並遞迴負區間
        if not neg_verified and not self._early_stop:
            logger.info(f"  Entering negative branch...")
            neg_result = self._recursive_split_verify(
                X, eps, top1_class, p, split_count + 1, max_splits,
                start_timestep=next_start_timestep, refine_preh=neg_bounds_state, mode=mode, # 將start_timestep從unsafe_layer + 1改next_start_timestep
                sample_id=sample_id, path=path + [(unsafe_layer, unsafe_neuron, 'neg')],
                first_unsafe_layer=first_unsafe_layer,
                locate_method=locate_method,
                selected_neurons=selected_neurons,
                split_history=new_split_history
            )
        else:
            neg_result = {
                'sample_id': sample_id,
                'is_verified': neg_verified,
                'split_tree': None,
                'all_leaf_bounds': [],
                'total_splits': split_count + 1,
                'split_times': []
            }

        # 合併子樹結果
        current_split['pos_subtree'] = pos_result.get('split_tree')
        current_split['neg_subtree'] = neg_result.get('split_tree')
        current_split['is_leaf'] = False

        # 收集所有leaf bounds
        all_leaf_bounds = []
        all_leaf_bounds.extend(pos_result.get('all_leaf_bounds', []))
        all_leaf_bounds.extend(neg_result.get('all_leaf_bounds', []))

        # 計算總split數
        total_splits = max(
            pos_result.get('total_splits', 0),
            neg_result.get('total_splits', 0)
        )

        # 整體驗證結果
        is_verified = pos_result['is_verified'] and neg_result['is_verified']

        logger.info(f"  Sample {sample_id+1}: Split {split_count+1} results - "
                    f"pos={'SUCCESS' if pos_result['is_verified'] else 'FAILED'}, "
                    f"neg={'SUCCESS' if neg_result['is_verified'] else 'FAILED'}")
        
        # 合併子樹的時間記錄
        all_split_times = current_record.copy()
        all_split_times.extend(pos_result.get('split_times', []))
        all_split_times.extend(neg_result.get('split_times', []))

        return {
            'sample_id': sample_id,
            'is_verified': is_verified,
            'split_tree': current_split,
            'all_leaf_bounds': all_leaf_bounds,
            'total_splits': total_splits,
            'first_unsafe_layer': first_unsafe_layer,
            'first_unsafe_layer_method': locate_method,
            'split_timesteps': [[item[0], item[1]] if isinstance(item, tuple) else item 
                for item in split_history],
            'split_times': all_split_times
        }
    
    def _check_region_robust(self, yL_out, yU_out, top1_class):
        """
        檢查單一子問題單一樣本的robustness
        回傳True代表驗證成功，False代表失敗需要繼續切割
        """
        if isinstance(top1_class, torch.Tensor):
            top1_class = top1_class.item()

        other_classes = [j for j in range(self.output_size) if j != top1_class]
        region_safe = all(yL_out[0, top1_class] > yU_out[0, j] for j in other_classes)

        return region_safe
    
    def record_sample_split_tree(self, sample_result):
        """記錄單個樣本的完整 split tree"""
        sample_id = sample_result['sample_id']
        all_leaf_bounds = sample_result.get('all_leaf_bounds', [])
        top1_class = sample_result.get('top1_class')
        
        # 計算 global merged bounds（所有 leaf nodes 的 worst-case union）
        if all_leaf_bounds:
            all_yL = torch.stack([bounds[1] for bounds in all_leaf_bounds])  # [num_leaves, 1, output_size]
            all_yU = torch.stack([bounds[2] for bounds in all_leaf_bounds])
            
            global_merge_yL = all_yL.min(dim=0)[0]  # [1, output_size]
            global_merge_yU = all_yU.max(dim=0)[0]
        else:
            global_merge_yL = None
            global_merge_yU = None
        
        # 檢查是否所有 leaf 都驗證成功
        if all_leaf_bounds and top1_class is not None:
            all_leaves_verified = all(
                self._check_region_robust(bounds[1], bounds[2], top1_class)
                for bounds in all_leaf_bounds
            )
        else:
            # 如果沒有 leaf bounds，檢查是否是 no_refinement 或 popqorn_only 的情況
            # 這些情況下如果 is_verified=True，則 all_leaves_verified 也應該是 True
            no_refinement = sample_result.get('no_refinement', False)
            popqorn_only = sample_result.get('popqorn_only', False)
            if (no_refinement or popqorn_only) and sample_result['is_verified']:
                all_leaves_verified = True
            else:
                all_leaves_verified = False
        
        # 計算改善率（相比 POPQORN）
        if global_merge_yL is not None and top1_class is not None:

            no_refinement = sample_result.get('no_refinement', False)

            if not no_refinement:
                # 修改：從 per-sample 的 original bounds 讀取
                original_yL = self.orig_output_bounds_per_sample[sample_id]['yL']  # [1, output_size]
                original_yU = self.orig_output_bounds_per_sample[sample_id]['yU']
                # Only include improvement if refinement was actually performed
                improvement = self.compute_bounds_improve(
                    original_yL, original_yU,
                    global_merge_yL, global_merge_yU,
                    top1_class
                )
            else:
                improvement = None # No refinement done, so no improvement
        else:
            improvement = None

        # whether this sample is changed from unverified to verified
        pq_failed_zs_success = (
            sample_result.get('popqorn_verified', True) == False and 
            sample_result['is_verified'] == True
        )

        leaf_paths = []
        if all_leaf_bounds:
            for leaf in all_leaf_bounds:
                # leaf[0] 是 path list, e.g., [(3, 'pos'), (5, 'neg')]
                raw_path = leaf[0] 
                # 提取 timestep
                timestep_seq = [step[0] for step in raw_path]
                leaf_paths.append(timestep_seq)
        
        # 找出最長的一條路徑作為代表 (或者你可以選擇存所有路徑)
        # longest_path_seq = max(leaf_paths, key=len) if leaf_paths else []
        split_timesteps_with_neurons = sample_result.get('split_timesteps', [])
        
        sample_record = {
            'sample_id': sample_id,
            'popqorn_verified': sample_result.get('popqorn_verified', None),
            'is_verified': sample_result['is_verified'],
            'split_tree': sample_result.get('split_tree'),
            'all_leaf_bounds': all_leaf_bounds,
            'num_leaves': len(all_leaf_bounds),
            'global_merged_bounds': (global_merge_yL, global_merge_yU),
            'all_leaves_verified': all_leaves_verified,
            'total_splits': sample_result['total_splits'],
            'improvements': improvement,
            'first_unsafe_layer': sample_result.get('first_unsafe_layer'),
            'first_unsafe_layer_method': sample_result.get('first_unsafe_layer_method'),
            'no_refinement': sample_result.get('no_refinement', False),
            'split_timesteps': split_timesteps_with_neurons,
            'pq_failed_zs_success': pq_failed_zs_success, # new
            'shap_vals': sample_result.get('shap_vals', None), # new
            'split_times': sample_result.get('split_times', [])
        }
        
        if not hasattr(self, 'sample_records'):
            self.sample_records = []
        
        self.sample_records.append(sample_record)

    def compute_bounds_improve(self, orig_yL, orig_yU, split_yL, split_yU, top1_class):
        """計算相對於原始output bounds的改進狀況"""
        if isinstance(top1_class, torch.Tensor):
            top1_class = top1_class.item()
        
        other_classes = [j for j in range(self.output_size) if j != top1_class]
        
        # 計算原始 worst-case gap
        orig_gaps = [orig_yL[0, top1_class] - orig_yU[0, j] for j in other_classes]
        orig_min_gap = min(orig_gaps).item()
        
        # 計算 split 後的 worst-case gap
        split_gaps = [split_yL[0, top1_class] - split_yU[0, j] for j in other_classes]
        split_min_gap = min(split_gaps).item()

        gap_improvement = split_min_gap - orig_min_gap

        # Top-1 lower bound improvement
        top1_improvement = (split_yL[0, top1_class] - orig_yL[0, top1_class]).item()

        # Other classes upper bound reduction
        other_reduction = sum(
            (orig_yU[0, j] - split_yU[0, j]).item() 
            for j in other_classes
        ) / len(other_classes)

        # 檢查是否嚴格縮小
        strict_tighten = bool(
            (split_yU <= orig_yU).all().item() and 
            (split_yL >= orig_yL).all().item()
        )

        return {
            'gap_improvement': gap_improvement,
            'top1_improvement': top1_improvement,
            'other_reduction': other_reduction,
            'original_gap': orig_min_gap,
            'split_gap': split_min_gap,
            'strict_tighten': strict_tighten
        }
    
    def _log_split_details_recursive(self, tree_node, indent=0):
        """遞歸輸出每個 split 的詳細信息"""
        if tree_node is None:
            return
        
        prefix = " " * indent
        timestep = tree_node.get('timestep')
        
        if timestep is not None:
            logger.info(f"{prefix}Split at timestep {timestep}")
            
        # 遞歸子樹
        if 'pos_subtree' in tree_node and tree_node['pos_subtree']:
            self._log_split_details_recursive(tree_node['pos_subtree'], indent + 2)
        
        if 'neg_subtree' in tree_node and tree_node['neg_subtree']:
            self._log_split_details_recursive(tree_node['neg_subtree'], indent + 2)
    
    def _log_split_tree_recursive(self, tree_node, indent=0):
        if tree_node is None:
            return
        
        prefix = "  " * indent
        split_count = tree_node.get('split_count', '?')
        layer = tree_node.get('layer', '?')
        
        logger.info(f"{prefix}Split {split_count} at Layer {layer}")
        
        if tree_node.get('is_leaf', False):
            logger.info(f"{prefix}  [LEAF] Both regions verified")
        else:
            # 顯示子樹
            if 'pos_subtree' in tree_node and tree_node['pos_subtree']:
                logger.info(f"{prefix}  ├─ Positive branch:")
                self._log_split_tree_recursive(tree_node['pos_subtree'], indent + 2)
            
            if 'neg_subtree' in tree_node and tree_node['neg_subtree']:
                logger.info(f"{prefix}  └─ Negative branch:")
                self._log_split_tree_recursive(tree_node['neg_subtree'], indent + 2)

    def generate_final_report(self):
        """生成最終的驗證報告"""
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL VERIFICATION REPORT")
        logger.info(f"{'='*80}")

        if not hasattr(self, 'sample_records') or not self.sample_records:
            logger.info("No sample records found.")
            return
        
        N = len(self.sample_records)
        verified_count = sum(1 for r in self.sample_records if r['is_verified'])

        # Differ types of results
        popqorn_only_count = sum(1 for r in self.sample_records if r.get('popqorn_only', False))
        no_refinement_count = sum(1 for r in self.sample_records if r.get('no_refinement', False))
        refined_count = N - popqorn_only_count - no_refinement_count
        
        logger.info(f"Total samples: {N}")
        logger.info(f"Verified samples: {verified_count}/{N} ({verified_count/N*100:.1f}%)")
        logger.info(f"Failed samples: {N - verified_count}/{N}")
        logger.info(f"\nRefinement breakdown:")
        logger.info(f"  POPQORN only: {popqorn_only_count}")
        logger.info(f"  No refinement (no cross-zero layer): {no_refinement_count}")
        logger.info(f"  Refined samples: {refined_count}\n")
            
        for record in self.sample_records:
            sample_id = record['sample_id']

            logger.info(f"{'='*80}")
            logger.info(f"SAMPLE {sample_id + 1}")
            logger.info(f"{'='*80}")
            logger.info(f"Verification: {'SUCCESS' if record['is_verified'] else 'FAILED'}")
            logger.info(f"Total splits performed: {record['total_splits']}")
            logger.info(f"Number of leaf nodes: {record['num_leaves']}")
            logger.info(f"All leaves verified: {record['all_leaves_verified']}")
            if record.get('first_unsafe_layer') is not None:
                logger.info(f"First unsafe layer selected: {record['first_unsafe_layer']}")
                if record.get('first_unsafe_layer_method') is not None:
                    logger.info(f"First unsafe layer method: {record['first_unsafe_layer_method']}")

            if record.get('split_timesteps'):
                logger.info(f"Split timesteps sequence: {record['split_timesteps']}")

            # 標記樣本類型
            if record.get('popqorn_only', False):
                logger.info(f"Sample type: POPQORN only")
            elif record.get('no_refinement', False):
                logger.info(f"Sample type: No refinement (no cross-zero layer)")
            else:
                logger.info(f"Sample type: Refined")

            # 顯示split_tree
            # if record['split_tree']:
            #     logger.info(f"\nSplit Tree Structure:")
            #     self._log_split_tree_recursive(record['split_tree'], indent=2)

            # 顯示improvements
            if record['improvements']:
                imp = record['improvements']
                logger.info(f"\nImprovements vs POPQORN:")
                logger.info(f"  Gap improvement: {imp['gap_improvement']:.2f}")
                logger.info(f"  Original min gap: {imp['original_gap']:.2f}")
                logger.info(f"  Split min gap: {imp['split_gap']:.2f}")
                logger.info(f"  Top-1 lower bound +: {imp['top1_improvement']:.2f}")
                logger.info(f"  Other upper bound -: {imp['other_reduction']:.2f}")
                logger.info(f"  Strict tighten: {imp['strict_tighten']}")

        
            logger.info("")  # 空行分隔各 split
        
        logger.info(f"{'='*80}\n")
        logger.info("STATISTICS SUMMARY")
        logger.info(f"{'='*80}")
        
        total_splits = sum(r['total_splits'] for r in self.sample_records)
        avg_splits = total_splits / N if N > 0 else 0
        max_splits = max(r['total_splits'] for r in self.sample_records)
        
        logger.info(f"Average splits per sample: {avg_splits:.2f}")
        logger.info(f"Maximum splits: {max_splits}")
        logger.info(f"Total splits across all samples: {total_splits}")

        # Strict tighten 統計
        strict_tighten_count = sum(
            1 for r in self.sample_records 
            if r['improvements'] and r['improvements']['strict_tighten']
        )
        logger.info(f"Samples with strict tighten: {strict_tighten_count}")
        
        # Gap improvement 統計
    
        failed_improvements = [
            r['improvements']['gap_improvement'] 
            for r in self.sample_records 
            if r['improvements'] and not r['popqorn_verified'] and not r.get('popqorn_only', False) and not r.get('no_refinement', False)
        ]
        
        if failed_improvements:
            avg_fail_gap = sum(failed_improvements) / len(failed_improvements)
            logger.info(f"\nAverage top-1 gap improvement (fail): {avg_fail_gap:.2f}")
            logger.info(f"Samples with top-1 gap improvement (fail): {sum(1 for x in failed_improvements if x > 0)}/{len(failed_improvements)}")

        # Top-1 improvement 統計

        failed_top1_imp = [
            r['improvements']['top1_improvement'] 
            for r in self.sample_records 
            if r['improvements'] and not r['popqorn_verified'] and not r.get('popqorn_only', False) and not r.get('no_refinement', False)
        ]
        
        if failed_top1_imp:
            avg_fail_top1 = sum(failed_top1_imp) / len(failed_top1_imp)
            logger.info(f"\nAverage top-1 lower bound improvement (fail, refined only): {avg_fail_top1:.2f}")
            logger.info(f"Samples with top-1 lower bound improvement (fail, refined only): {sum(1 for x in failed_top1_imp if x > 0)}/{len(failed_top1_imp)}")
        
        # Other reduction 統計
        
        failed_other_reduc = [
            r['improvements']['other_reduction'] 
            for r in self.sample_records 
            if r['improvements'] and not r['popqorn_verified'] and not r.get('popqorn_only', False) and not r.get('no_refinement', False)
        ]
        
        if failed_other_reduc:
            avg_fail_other = sum(failed_other_reduc) / len(failed_other_reduc)
            logger.info(f"\nAverage other upper bound reduction (fail, refined only): {avg_fail_other:.2f}")
            logger.info(f"Samples with other upper bound reduction (fail, refined only): {sum(1 for x in failed_other_reduc if x > 0)}/{len(failed_other_reduc)}")
        
        improved_samples = [
            r for r in self.sample_records 
            if r.get('pq_failed_zs_success', False)
        ]
        
        if improved_samples:
            logger.info(f"\n{'='*80}")
            logger.info("POPQORN FAILED → ZEROSPLIT SUCCESS SAMPLES (DETAILED)")
            logger.info(f"{'='*80}")
            logger.info(f"Total improved samples: {len(improved_samples)}/{len(failed_improvements)}")
            
            for record in improved_samples:
                sample_id = record['sample_id']
                logger.info(f"\n{'-'*80}")
                logger.info(f"Sample {sample_id + 1}:")
                logger.info(f"  Split timesteps: {record.get('split_timesteps', [])}")
                logger.info(f"  Total splits: {record['total_splits']}")
                logger.info(f"  Number of leaves: {record['num_leaves']}")
                
                # 顯示 improvement
                if record['improvements']:
                    imp = record['improvements']
                    logger.info(f"  Gap improvement: {imp['gap_improvement']:.1f} (from {imp['original_gap']:.1f} to {imp['split_gap']:.1f})")
                    logger.info(f"  Top-1 improvement: {imp['top1_improvement']:.1f}")
                    logger.info(f"  Other reduction: {imp['other_reduction']:.1f}")
                
                # 從 split_tree 中提取每個 split 的詳細信息
                if record['split_tree']:
                    logger.info(f"  Split details:")
                    self._log_split_details_recursive(record['split_tree'], indent=4)
                
                logger.info(f"{'-'*80}")

            all_split_timesteps = []
            for record in improved_samples:
                split_ts = record.get('split_timesteps', [])
                all_split_timesteps.extend([tuple(ts) if isinstance(ts, list) else ts for ts in split_ts])

            if all_split_timesteps:
                timestep_counts = Counter(all_split_timesteps)
                logger.info(f"\nTimestep split distribution (POPQORN FAILED → ZEROSPLIT SUCCESS):")
                for t in sorted(timestep_counts.keys()):
                    logger.info(f"  Timestep {t}: {timestep_counts[t]} splits")
                logger.info(f"  Most critical timesteps: {sorted(timestep_counts.items(), key=lambda x: x[1], reverse=True)[:3]}") 

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
    parser.add_argument('--mode', default='shap', type=str, choices=['shap'],
                        help='verification mode (default: shap)')
    parser.add_argument('--n-workers', default=None, type=int,
                        help='number of parallel workers (default: cpu count)')
    parser.add_argument('--use-evr', action='store_true',
                    help='use EVR (Exact Verifiable Robustness) mode instead of fixed epsilon')
    parser.add_argument('--eps-min', default=0.005, type=float,
                        help='minimum epsilon for EVR search (default: 0.005)')
    parser.add_argument('--eps-max', default=0.101, type=float,
                        help='maximum epsilon for EVR search (default: 1.0)')

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
    mode = args.mode
    
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
        # input_size = int(28*28 / args.time_step)
        input_size = int(32 * 32 * 3 / args.time_step)
        # input_size = 3
        hidden_size = args.hidden_size
        output_size = 10
        # output_size = 3
        time_step = args.time_step
        
        verifier = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, 
                                   args.activation, max_splits=args.max_splits, debug=args.debug)
        
        model_file = os.path.join(args.work_dir, args.model_name)
        verifier.load_state_dict(torch.load(model_file, map_location='cpu'))
        verifier.to(device)
        # MNIST
        # X, y, target_label = sample_mnist_data(
        #     N=N, seq_len=time_step, device=device,
        #     data_dir='../data/mnist', train=False, shuffle=True, rnn=verifier
        # )
        # MNIST sequence
        # X, y, target_label = sample_seq_mnist_data(
        #     N=N, 
        #     time_step=time_step, 
        #     device=device,
        #     data_dir='./data/mnist_seq/sequences/',
        #     train=False,
        #     rnn=verifier  # 篩選預測正確的樣本
        # )
        # CIFAR10
        X, y, target_label = sample_cifar10_data(
            N=N,
            time_step=time_step,
            device=device,
            data_dir='./data/cifar-10-batches-py/',
            train=False,
            rnn=verifier  # 篩選預測正確的樣本
        )
        
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
    is_verified, unsafe_layer, top1_class, popqorn_safe_count = verifier.verify_network_recursive(
        X, eps, p, max_splits=args.max_splits, mode=mode, n_workers=args.n_workers,
        use_evr=args.use_evr, eps_range=(args.eps_min, args.eps_max) # 不想用EVR就把這裡參數刪掉
    )

    # 準備結果數據
    results_data = {
        'is_verified': is_verified,
        'unsafe_layer': unsafe_layer,
        'top1_class': top1_class.tolist() if hasattr(top1_class, 'tolist') else int(top1_class),
        'total_samples': N,
        'split_count': sum(1 for r in verifier.sample_records if r['is_verified']) if hasattr(verifier, 'sample_records') else 0,
        'popqorn_safe_count': popqorn_safe_count,
        'zerosplit_safe_count': sum(1 for r in verifier.sample_records if r['all_leaves_verified'])
    }
    # logger.info(f"\nPQ Safe Samples: {popqorn_safe_count}/{N}")
    # logger.info(f"ZS Safe Samples: {results_data['zerosplit_safe_count']}/{N}")
    
    # 保存結果（包含所有captured logs）
    save_verification_results(json_file, txt_file, log_capture.captured_logs, args, results_data, session_dir, verifier)
    
    logger.info(f"\nResults saved to session: {os.path.basename(session_dir)}")
    logger.info(f"JSON detail: {os.path.basename(json_file)}")
    logger.info(f"TXT summary: {os.path.basename(txt_file)}")

if __name__ == "__main__":
    main()