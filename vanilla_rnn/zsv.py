import types
import torch
from bound_vanilla_rnn import RNN
import argparse
import os

import torch
from utils.sample_data import sample_mnist_data
import get_bound_for_general_activation_function as get_bound

class ZeroSplitVerifier(RNN):
    def __init__(self, input_size, hidden_size, output_size, time_step, activation, max_splits=1, debug=False):
        RNN.__init__(self, input_size, hidden_size, output_size, time_step, activation) # 1 layer
        self.max_splits = max_splits
        self.split_count = 0
        self.debug = debug

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
                           merge_results=True, split_done=False, cross_zero=None):

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
            
            self.l[v] = yL
            self.u[v] = yU

            # 檢查是否需要在current timestep進行split
            if unsafe_layer == v and not split_done:
                if not merge_results:
                    # 不合併結果，分別計算兩個子問題
                    result = self._split_and_compute_separate(eps, p, v, Eps_idx, cross_zero)

                    return result
                else:
                    # 合併結果
                    return self._split_and_merge(eps, p, v, Eps_idx, cross_zero)

            return yL, yU
        
    def _split_and_compute_separate(self, eps, p, v, Eps_idx, cross_zero=None):
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

        # print(f"X_pos shape: {X_pos.shape}")
        self.l[v] = pos_l
        self.u[v] = pos_u
        
        # 根據v是否為最後一個隱藏層選擇計算方式
        if v < self.time_step:
            # 從v+1層開始計算到最後一個隱藏層
            for k in range(v+1, self.time_step+1):
                pos_yL, pos_yU = self.compute2sideBound(
                    eps, p, k, X=full_X[:, 0:k, :], Eps_idx=Eps_idx
                ) # eps換成eps_pos
                # print(f"第 {k} timestep的pre-activation bounds: 正區間下界: {pos_yL}, 正區間上界: {pos_yU}")
            
            # 計算輸出層
            pos_yL, pos_yU = self.computeLast2sideBound(
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"正區間的final bounds: {pos_yL}, {pos_yU}")
            # print(f"正區間的bounds差異: {pos_yU - pos_yL}")
        else:

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
        
        self.l[v] = neg_l
        self.u[v] = neg_u
        
        # 根據v是否為最後一個隱藏層選擇計算方式
        if v < self.time_step:
            # 從v+1層開始計算到最後一個隱藏層
            for k in range(v+1, self.time_step+1):
                neg_yL, neg_yU = self.compute2sideBound(
                    eps, p, k, X=full_X[:, 0:k, :], Eps_idx=Eps_idx
                ) # eps換eps_neg
                # print(f"第 {k} timestep的pre-activation bounds: 負區間下界: {neg_yL}, 負區間上界: {neg_yU}")
            
            # 計算輸出層
            neg_yL, neg_yU = self.computeLast2sideBound(
                eps, p, v=self.time_step+1, X=full_X, Eps_idx=Eps_idx
            )
            # print(f"負區間的final bounds: {neg_yL}, {neg_yU}")
            # print(f"負區間的bounds差異: {neg_yU - neg_yL}")
        else:
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
            print(f"Original top1 classes: {top1_class}")

        # 2. 計算bound
        yL, yU = self.computePreactivationBounds(eps, p=2, X=X, 
                                        Eps_idx=torch.arange(1,self.time_step+1))
        # print(f"不做split的隱藏層bounds to verify: {yL}, {yU}")
        
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
        return robust, top1_class

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

    def locate_unsafe_layer(self, start_timestep=1):
        """用Sequential(原為Binary) search找到第一個出現violation的layer(需要精進)"""
        
        # 依序檢查從start_timestep開始的每一層
        for timestep in range(start_timestep, self.time_step+1):
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
            # print(f"合併後的輸出層 bounds: {yL_out}, {yU_out}")
            # print(f"合併後的bounds差異: {yU_out - yL_out}")
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
                    
            return is_verified, unsafe_layer, top1_class
        
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

def create_toy_rnn(verifier, batch_size=1, hidden_size=1, input_size=1, output_size=1):
    with torch.no_grad():
        verifier.rnn = None
        a = hidden_size
        input_size = input_size
        output_size = output_size
        
        # ===== 初始隱藏狀態 =====
        verifier.a_0 = torch.zeros(batch_size, a, dtype=torch.float32)
        
        # ===== W_ax: a × 1，全為+1 =====
        verifier.W_ax = torch.ones(a, input_size, dtype=torch.float32)
        
        # ===== W_aa: a × a，正負號各半 =====
        # verifier.W_aa = create_balanced_weight_matrix(a)
        # verifier.W_aa = torch.randn(a, a, dtype=torch.float32)
        verifier.W_aa = torch.tensor([
            [-1.3261, -1.1577, -1.3466,  2.7825],
            [ 0.4217,  2.1293,  1.7256,  0.1499],
            [-0.5402, -0.8619, -0.0342,  0.0495],
            [ 1.7684,  0.9300,  1.1032,  0.6404]
        ], dtype=torch.float32)
        # ===== W_fa: 2 × a，全為+1 =====
        verifier.W_fa = torch.ones(output_size, a, dtype=torch.float32)
        
        # ===== Bias全為0 =====
        verifier.b_ax = torch.zeros(a, dtype=torch.float32)
        verifier.b_aa = torch.zeros(a, dtype=torch.float32)
        # verifier.b_f = torch.zeros(output_size, dtype=torch.float32)
        verifier.b_f = torch.tensor([0.2, -0.925], dtype=torch.float32)
        
        # ===== 自定義forward函數 =====
        def forward(self, X):
            with torch.no_grad():
                N = X.shape[0]
                h = torch.zeros(N, self.time_step+1, self.num_neurons, device=X.device)

                if self.a_0 is not None:
                    h[:,0,:] = self.a_0
                
                # Timestep 1
                pre_h = torch.matmul(X[:,0,:], self.W_ax.t()) + self.b_ax
                if self.a_0 is not None:
                    pre_h = pre_h + torch.matmul(h[:,0,:], self.W_aa.t()) + self.b_aa
                h[:,1,:] = torch.relu(pre_h)
                
                # Timestep 2
                pre_h = (torch.matmul(X[:,1,:], self.W_ax.t()) + self.b_ax + 
                        torch.matmul(h[:,1,:], self.W_aa.t()) + self.b_aa)
                h[:,2,:] = torch.relu(pre_h)
                
                # Output
                output = torch.matmul(h[:,2,:], self.W_fa.t()) + self.b_f
                return output
        
        verifier.forward = types.MethodType(forward, verifier)
        
        print(f"\n=== Toy RNN Configuration (a={a}) ===")
        print(f"W_ax shape: {verifier.W_ax.shape} (all +1)")
        print(f"W_aa shape: {verifier.W_aa.shape} (±1 balanced)")
        print(f"W_fa shape: {verifier.W_fa.shape} (all +1)")
        print(f"\nW_aa matrix:\n{verifier.W_aa}")

def create_balanced_weight_matrix(a, pattern='checkerboard'):
    """
    創建 a×a 的權重矩陣，數值為±1，正負號各半
    
    Args:
        a: 矩陣維度
        pattern: 'checkerboard' (棋盤式), 'block' (分塊式), 'random' (隨機)
    
    Returns:
        torch.Tensor of shape (a, a)
    """
    assert a % 2 == 0, "a must be even for balanced +1/-1"
    
    if pattern == 'checkerboard':
        # 棋盤式: 交替排列
        # [+1, -1, +1, -1]
        # [-1, +1, -1, +1]
        # [+1, -1, +1, -1]
        # [-1, +1, -1, +1]
        W = torch.ones(a, a)
        for i in range(a):
            for j in range(a):
                if (i + j) % 2 == 1:
                    W[i, j] = -1
        return W
    
    elif pattern == 'block':
        # 分塊式: 每行前半+1，後半-1
        # [+1, +1, -1, -1]
        # [+1, +1, -1, -1]
        # [+1, +1, -1, -1]
        # [+1, +1, -1, -1]
        W = torch.ones(a, a)
        W[:, a//2:] = -1
        return W
    
    elif pattern == 'random':
        # 隨機式: 每行隨機分配，但保證各半
        W = torch.zeros(a, a)
        for i in range(a):
            indices = torch.randperm(a)
            W[i, indices[:a//2]] = 1
            W[i, indices[a//2:]] = -1
        return W
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
        
def main():

    parser = argparse.ArgumentParser(description='ZeroSplit Verifier for RNN robustness verification')

    parser.add_argument('--hidden-size', default=2, type=int, metavar='HS',
                        help='hidden layer size (default: 2)')
    parser.add_argument('--time-step', default=2, type=int, metavar='TS',
                        help='number of time steps (default: 2)')
    parser.add_argument('--activation', default='relu', type=str, metavar='A',
                        help='activation function: tanh or relu (default: relu)')
    parser.add_argument('--work-dir', default='../models/mnist_classifier/rnn_2_2_relu/', type=str, metavar='WD',
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
        a = 4
        batch_size = 1
        input_size = 1
        hidden_size = a
        output_size = 2
        time_step = 2

        verifier = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, 
                                   args.activation, max_splits=args.max_splits, debug=args.debug)

        create_toy_rnn(verifier, batch_size=batch_size, hidden_size=a, input_size=input_size, output_size=output_size)

        X = torch.tensor([
            [[0.03], [0.1]],
        ], dtype=torch.float32).to(device)
        # X = torch.ones((batch_size, time_step, input_size), device=device)

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
        torch.set_printoptions(
            threshold=float('inf'),  # 顯示所有元素
            linewidth=200,          # 每行字符數
            edgeitems=10           # 邊緣顯示的元素數
        )
        # print(f"{X[0]}")
        verifier.extractWeight(clear_original_model=False)
    is_verified, unsafe_layer, top1_class = verifier.verify_network(
    X, eps, merge_results=args.merge_results
    )

if __name__ == "__main__":
    main()