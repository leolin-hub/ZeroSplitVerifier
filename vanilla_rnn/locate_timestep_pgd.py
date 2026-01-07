import torch

class TimestepPGDValidator:
    """
    Revise Algo 2 from SURGEON to perform PGD-based adversarial attack to find critical timesteps.
    """

    def __init__(self, verifier, eps, p=2, x_min=-1, x_max=1, u1=10, u2=20, step_size=0.01):
        """
        Args:
            verifier: ZeroSplitVerifier instance
            eps: perturbation budget
            p: norm type (1, 2, or inf)
            u1: outer loop iterations (不同 random starts)
            u2: inner loop PGD iterations  
            step_size: gradient step size
        """
        self.verifier = verifier
        self.eps = eps
        self.p = p
        self.u1 = u1
        self.u2 = u2
        self.step_size = step_size
        self.x_min = x_min
        self.x_max = x_max

    def validate_timestep(self, X_clean, timestep, top1_class):
        """
        Validate the safety of a specific timestep using PGD attack.

        Args:
            X_clean: [N, time_step, input_size] - clean input
            timestep: which timestep to validate (1 to time_step)
            top1_class: [N] - ground truth labels
            
        Returns:
            is_unsafe: bool - True if found counterexample
            worst_violation: float - 最大 violation margin (越大越 unsafe)
        """
        N = X_clean.shape[0]
        worst_violation = 0.0
        found_counterexample = False

        # 對每個 sample 進行 validation
        for sample_idx in range(N):
            x_sample = X_clean[sample_idx:sample_idx+1] # [1, time_step, input_size]
            target = top1_class[sample_idx]

            # u1 iterations: 不同 random starts
            for _ in range(self.u1):
                # Sample v* within l_p ball around x_sample[:, timestep-1, :]
                amid = x_sample[:, timestep-1, :].clone()
                v_star = self._sample_in_ball(amid)

                # u2 iterations: PGD
                for _ in range(self.u2):
                    # Gen(v*): 構成完整輸入序列
                    x_adv = x_sample.clone()
                    x_adv[:, timestep-1, :] = v_star

                    # 用 verify_robustness 檢查 Post condition
                    is_verified, _, yL_out, yU_out = self.verifier.verify_robustness(x_adv, self.eps)

                    # 檢查 Post condition
                    if not is_verified:
                        # Found counterexample
                        found_counterexample = True
                        violation = self._compute_violation_from_bounds(yL_out, yU_out, target)
                        worst_violation = max(worst_violation, violation)

                    # === 使用 forward pass 的 gradient 引導搜索 ===
                    # 因為 verify_robustness 用 bound propagation 無 gradient
                    # 我們用 forward logit 作為 proxy 來引導方向
                    v_star_grad = self._compute_gradient_proxy(x_adv, target, amid, timestep)

                    # Update v* to maximize violation (gradient ascent)
                    with torch.no_grad():
                        v_star = v_star + self.step_size * torch.sign(v_star_grad)

                        # Clip to l_p ball around amid
                        v_star = self._project_to_ball(v_star, amid, self.eps)

            return found_counterexample, worst_violation
        
    def locate_critical_timestep(self, X_clean, top1_class, start=1, end=None):
        """
        Binary search 找到 critical timestep (第一個 unsafe 的 timestep)
        
        Args:
            X_clean: [N, time_step, input_size]
            top1_class: [N]
            start: 起始 timestep
            end: 結束 timestep (None = time_step)
            
        Returns:
            critical_timestep: int or None
            violation_scores: dict {timestep: violation_margin}
        """
        if end is None:
            end  = self.verifier.time_step

        violation_scores = {}

        # Binary search
        while start < end:
            mid = (start + end) // 2

            is_unsafe, violation = self.validate_timestep(X_clean, mid, top1_class)
            violation_scores[mid] = violation

            if is_unsafe:
                # mid is unsafe, critical timestep <= mid
                end = mid
            else:
                # mid is safe, critical timestep > mid
                start = mid + 1
        
        # Validate the final candidate
        if start <= self.verifier.time_step:
            is_unsafe, violation = self.validate_timestep(X_clean, start, top1_class)
            violation_scores[start] = violation
            if is_unsafe:
                return start, violation_scores
            
        return None, violation_scores
    
    def rank_timesteps_by_unsafety(self, X_clean, top1_class):
        """
        對所有 timesteps 計算 unsafe score 並排序
        用於決定 splitting priority
        
        Returns:
            ranked_timesteps: list of (timestep, score) sorted by score descending
        """ 
        scores = []

        for t in range(1, self.verifier.time_step + 1):
            _, violation = self.validate_timestep(X_clean, t, top1_class)
            scores.append((t, violation))

        # Sort by violation score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _sample_in_ball(self, x_center):
        """Random sample in l_p ball"""
        x_center = x_center.clone()
        noise = torch.randn_like(x_center)

        if self.p == 2:
            noise = noise / (torch.norm(noise, p=2) + 1e-8)
            radius = torch.rand(1).item() * self.eps
            noise = noise * radius
        elif self.p == float('inf'):
            noise = torch.rand_like(noise) * 2 - 1  # uniform in [-1, 1]
            noise = noise * self.eps
        elif self.p == 1:
            abs_noise = torch.abs(noise)
            abs_noise = abs_noise / (torch.sum(abs_noise) + 1e-8)
            signs = torch.sign(torch.randn_like(noise))
            noise = signs * abs_noise * torch.rand(1).item() * self.eps

        v_star = torch.clamp(x_center + noise, self.x_min, self.x_max)
            
        return v_star
    
    def _project_to_ball(self, x, x_center, eps):
        """Project x to l_p ball around x_center"""
        delta = x - x_center

        if self.p == 2:
            # l_2 projection
            norm = torch.norm(delta, p=2)
            if norm > eps:
                delta = delta / (norm + 1e-8) * eps
        elif self.p == float('inf'):
            # l_∞ projection: clip each dimension
            delta = torch.clamp(delta, -eps, eps)
        elif self.p == 1:
            # l_1 projection
            norm = torch.norm(delta, p=1)
            if norm > eps:
                # 投影到 l_1 ball 較複雜，使用近似
                delta = delta / (norm + 1e-8) * eps

        return torch.clamp(x_center + delta, self.x_min, self.x_max)
    
    def _compute_violation_from_bounds(self, yL_out, yU_out, target):
        """
        從 certified bounds 計算 Post condition violation margin
        
        Post condition: yL[target] > yU[other] for all other classes
        Violation: max_j≠target (yU[j] - yL[target])
        
        正值表示有 violation (unsafe)
        負值表示滿足 Post condition (safe)
        """
        target_lower = yL_out[0, target]  # certified lower bound of top1 class
        
        # 找所有其他 classes 的 upper bounds
        other_uppers = torch.cat([
            yU_out[0, :target],
            yU_out[0, target+1:]
        ])
        max_other_upper = torch.max(other_uppers)
        
        # Violation margin (positive = violation, negative = safe)
        violation = max_other_upper - target_lower
        return violation.item()
    
    def _compute_gradient_proxy(self, X_perturbed, target, amid, timestep):
        """
        用 forward pass 的 logits 計算 gradient，作為搜索方向的 proxy
        因為 verify_robustness 用 bound propagation 無法提供 gradient
        
        這是 Algorithm 2 Line 10 的實作:
        grad = ∂Loss/∂x_i ⊙ ∂Gen/∂v*(v*)
        
        在我們的情況，Gen(v*) 就是把 v* 放到 X[:, timestep-1, :]
        所以 ∂Gen/∂v* = Identity
        """
        # 只對 timestep 位置的 input 需要 gradient
        v = X_perturbed[:, timestep-1, :].clone().detach()
        v.requires_grad = True
        
        # Forward pass (用 clean forward，不是 certified bounds)
        X_temp = X_perturbed.clone()
        X_temp[:, timestep-1, :] = v
        output = self.verifier.forward(X_temp)  # [1, output_size]
        
        # Loss: maximize max_other_logit - target_logit
        # 這會鼓勵模型輸出錯誤
        target_logit = output[0, target]
        other_logits = torch.cat([
            output[0, :target],
            output[0, target+1:]
        ])
        max_other = torch.max(other_logits)
        
        loss = max_other - target_logit  # maximize to find adversarial
        
        # Backward
        if v.grad is not None:
            v.grad.zero_()
        loss.backward()
        
        grad = v.grad.clone().detach()
        return grad