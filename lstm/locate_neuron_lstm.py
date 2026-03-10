#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Neuron Locator for ZeroSplit

參考 vanilla_rnn/locate_timestep_shap.py，
為 LSTM 的 4-gate 結構，從所有 cross-zero 的 (timestep, gate, neuron)
中，選出 SHAP importance 最高的單一目標供 ZeroSplit 使用。

每次 split 只選一個 (t, gate, n)，切出：
  - neg sub-problem: gate_preact[n] <= 0
  - pos sub-problem: gate_preact[n] >= 0
"""

import torch
import torch.nn as nn
import numpy as np
import shap
from loguru import logger


GATE_NAMES = ['i', 'f', 'g', 'o']


class LSTMNeuronLocator:
    """
    使用 SHAP 為 LSTM ZeroSplit 選出單一最重要的
    (timestep, gate, neuron) 分割目標。
    """

    def __init__(self, verifier, background_size=20, eps=None, p=2, top_k=3):
        self.verifier = verifier
        self.background_size = background_size
        self.eps = eps
        self.p = p
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ranking(self, X, top1_class):
        """
        計算所有 cross-zero (timestep, gate, neuron) 的 SHAP importance，
        回傳由高到低排序的 list。

        Args:
            X:          (N, seq_len, input_size)
            top1_class: (N,) ground-truth labels

        Returns:
            ranked: [((t, gate, n), importance), ...]  降序排列
        """
        background = self._generate_background(X)
        candidates = []

        for t in range(1, self.verifier.seq_len + 1):
            importance_per_gate = self._compute_gate_importance(
                X, background, t, top1_class
            )
            if importance_per_gate is None:
                continue

            cross = self.verifier.detect_cross_zero(t)  # dict[gate -> BoolTensor]

            for g_idx, gate in enumerate(GATE_NAMES):
                gate_cross = cross[gate]        # (batch, hidden)
                gate_imp = importance_per_gate[g_idx]  # (hidden,)

                for n in range(self.verifier.hidden_size):
                    if gate_cross[:, n].any():
                        candidates.append(((t, gate, n), float(gate_imp[n])))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:self.top_k]
        candidates.sort(key=lambda x: x[0][0])
        return candidates

    def select_next_split(self, ranked, split_history=None):
        """
        從 ranked list 挑選下一個單一分割目標，跳過已分割過的項目。

        Args:
            ranked:        compute_ranking() 的回傳值
            split_history: set of (t, gate, n) already split

        Returns:
            (t, gate, n) or (None, None, None) if no valid target remains
        """
        if split_history is None:
            split_history = set()

        for (t, gate, n), importance in ranked:
            if (t, gate, n) in split_history:
                continue

            # 再次確認目前 bounds 下仍 cross-zero（bounds 可能因 split 而收緊）
            cross = self.verifier.detect_cross_zero(t)
            if cross[gate][:, n].any():
                logger.info(f"  Selected split target: t={t}, gate={gate}, "
                            f"n={n}, importance={importance:.4f}")
                return t, gate, n

        logger.info("  No valid split target found.")
        return None, None, None

    # ------------------------------------------------------------------
    # LSTM forward helpers
    # 參考 lstm.py My_lstm.forward() lines 271-317
    # ------------------------------------------------------------------

    def _init_states(self, N, device):
        """初始化 a0, c0，邏輯對應 My_lstm.forward() lines 284-317。"""
        v = self.verifier
        a0 = v.a0 if v.a0 is not None else torch.zeros(N, v.hidden_size, device=device)
        c0 = v.c0 if v.c0 is not None else torch.zeros(N, v.hidden_size, device=device)
        return a0, c0

    def _lstm_step(self, x_k, a_prev, c_prev):
        """
        單步 LSTM forward，對應 lstm.py My_lstm.forward() lines 273-283。
        yi = Wix x + Wia a + bi,  等價於 mat_mul(Wix, x, bi) + mat_mul(Wia, a)
        """
        v = self.verifier
        yi_k = x_k @ v.Wix.T + a_prev @ v.Wia.T + v.bi
        yf_k = x_k @ v.Wfx.T + a_prev @ v.Wfa.T + v.bf
        yg_k = x_k @ v.Wgx.T + a_prev @ v.Wga.T + v.bg
        yo_k = x_k @ v.Wox.T + a_prev @ v.Woa.T + v.bo
        c = torch.sigmoid(yf_k) * c_prev + torch.sigmoid(yi_k) * torch.tanh(yg_k)
        a = torch.sigmoid(yo_k) * torch.tanh(c)
        return a, c

    def _forward_to_gate_preact(self, X, timestep):
        """
        LSTM forward 到 timestep，回傳該 timestep 4 個 gate 的 pre-activations，
        拼接為 (N, 4 * hidden_size)，順序 [yi, yf, yg, yo]。

        對應 My_lstm.forward() lines 271-317：先跑 timestep-1 步取得 a_prev, c_prev，
        再對第 timestep 步算 gate pre-acts（不套 activation，保留原始線性值）。
        """
        v = self.verifier
        N = X.shape[0]
        device = X.device

        with torch.no_grad():
            a_prev, c_prev = self._init_states(N, device)
            for k in range(timestep - 1):
                a_prev, c_prev = self._lstm_step(X[:, k, :], a_prev, c_prev)

            x_t = X[:, timestep - 1, :]
            yi = x_t @ v.Wix.T + a_prev @ v.Wia.T + v.bi
            yf = x_t @ v.Wfx.T + a_prev @ v.Wfa.T + v.bf
            yg = x_t @ v.Wgx.T + a_prev @ v.Wga.T + v.bg
            yo = x_t @ v.Wox.T + a_prev @ v.Woa.T + v.bo

        return torch.cat([yi, yf, yg, yo], dim=1)  # (N, 4 * hidden_size)

    def _get_cell_state_before(self, X, timestep):
        """
        LSTM forward 到 timestep-1，回傳 c_{timestep-1}。
        供 _create_gate_wrapper 內的 c_t 計算使用。
        """
        v = self.verifier
        N = X.shape[0]
        device = X.device

        with torch.no_grad():
            a_prev, c_prev = self._init_states(N, device)
            for k in range(timestep - 1):
                a_prev, c_prev = self._lstm_step(X[:, k, :], a_prev, c_prev)

        return c_prev  # (N, hidden_size)

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _compute_gate_importance(self, X, background, timestep, top1_class):
        """
        計算 timestep 的 4 個 gate 各 neuron 的 SHAP importance。

        Returns:
            list of 4 np.ndarray, 每個 shape (hidden_size,)，順序 [i, f, g, o]。
            失敗時回傳 None。
        """
        v = self.verifier
        N = X.shape[0]

        c_prev_X = self._get_cell_state_before(X, timestep)
        wrapper = self._create_gate_wrapper(timestep, X, c_prev_X)

        try:
            h_bg = self._forward_to_gate_preact(background, timestep)
            h_X  = self._forward_to_gate_preact(X, timestep)

            explainer = shap.GradientExplainer(wrapper, h_bg)
            shap_values = explainer.shap_values(h_X)

            hidden = v.hidden_size
            importance_per_gate = [np.zeros(hidden) for _ in range(4)]

            if isinstance(shap_values, list):
                # shap_values: list[num_classes] of (N, 4*hidden)
                for i in range(N):
                    class_idx = int(top1_class[i].item())
                    sv = np.abs(shap_values[class_idx][i])
                    for g_idx in range(4):
                        importance_per_gate[g_idx] += sv[g_idx*hidden:(g_idx+1)*hidden]
            else:
                # shap_values: (N, 4*hidden, num_classes)
                for i in range(N):
                    class_idx = int(top1_class[i].item())
                    sv = np.abs(shap_values[i, :, class_idx])
                    for g_idx in range(4):
                        importance_per_gate[g_idx] += sv[g_idx*hidden:(g_idx+1)*hidden]

            for g_idx in range(4):
                importance_per_gate[g_idx] /= N

            return importance_per_gate

        except Exception as e:
            logger.warning(f"SHAP failed for timestep {timestep}: {e}")
            return None

    def _create_gate_wrapper(self, timestep, X, c_prev):
        """
        建立 SHAP wrapper nn.Module：
          input:  (batch, 4*hidden)  — gate pre-activations at `timestep`
          output: (batch, num_classes) — final classification logits

        固定 c_{timestep-1} 和後續 timestep 的 input X 作為 buffer，
        確保 gradient 只對 gate pre-activations 計算。
        """
        v = self.verifier
        hidden = v.hidden_size

        class GatePreactToOutput(nn.Module):
            def __init__(self_, verifier_ref, ts, hidden_dim,
                         c_prev_tensor, X_tensor):
                super().__init__()
                self_.v = verifier_ref
                self_.ts = ts
                self_.hidden = hidden_dim
                # register_buffer 讓 buffer 跟隨 .to(device)
                self_.register_buffer('c_prev', c_prev_tensor.clone())
                self_.register_buffer('X_full', X_tensor.clone())

            def forward(self_, gate_preacts):
                batch = gate_preacts.shape[0]
                h = self_.hidden

                yi = gate_preacts[:, 0*h:1*h]
                yf = gate_preacts[:, 1*h:2*h]
                yg = gate_preacts[:, 2*h:3*h]
                yo = gate_preacts[:, 3*h:4*h]

                # c_prev: expand to match current batch size
                c_p = self_.c_prev.expand(batch, -1) \
                      if self_.c_prev.shape[0] != batch else self_.c_prev

                c_t = torch.sigmoid(yf) * c_p + torch.sigmoid(yi) * torch.tanh(yg)
                a_t = torch.sigmoid(yo) * torch.tanh(c_t)

                # X_full: expand to match batch size (background > X 時)
                X_ref = self_.X_full
                if X_ref.shape[0] != batch:
                    repeat = (batch + X_ref.shape[0] - 1) // X_ref.shape[0]
                    X_ref = X_ref.repeat(repeat, 1, 1)[:batch]

                a_cur, c_cur = a_t, c_t
                vv = self_.v
                for k in range(self_.ts, vv.seq_len):
                    x_k = X_ref[:, k, :]
                    yi_k = x_k @ vv.Wix.T + a_cur @ vv.Wia.T + vv.bi
                    yf_k = x_k @ vv.Wfx.T + a_cur @ vv.Wfa.T + vv.bf
                    yg_k = x_k @ vv.Wgx.T + a_cur @ vv.Wga.T + vv.bg
                    yo_k = x_k @ vv.Wox.T + a_cur @ vv.Woa.T + vv.bo
                    c_cur = (torch.sigmoid(yf_k) * c_cur
                             + torch.sigmoid(yi_k) * torch.tanh(yg_k))
                    a_cur = torch.sigmoid(yo_k) * torch.tanh(c_cur)

                # 對應 My_lstm.get_final_output()：W (out, hidden), b (out,)
                return a_cur @ vv.W.T + vv.b

        return GatePreactToOutput(v, timestep, hidden, c_prev, X)

    def _generate_background(self, X):
        """
        與 vanilla_rnn/locate_timestep_shap.py
        TimestepSHAPLocator._generate_background() 邏輯相同，直接搬移。
        """
        N = self.background_size
        background = X.repeat(N, 1, 1)
        device = X.device

        if self.eps is None or self.eps == 0:
            return background.detach()

        if self.p == 2:
            delta = torch.randn_like(background)
            norms = delta.view(N, -1).norm(p=2, dim=1, keepdim=True)
            delta = delta / norms.view(N, 1, 1)
            r = torch.rand(N, 1, 1, device=device) ** (1.0 / (X.shape[1] * X.shape[2]))
            delta = delta * r * self.eps
        elif self.p == float('inf'):
            delta = (torch.rand_like(background) * 2 - 1) * self.eps
        elif self.p == 1:
            delta = torch.randn_like(background)
            abs_delta = torch.abs(delta)
            abs_delta = abs_delta / (abs_delta.sum(dim=(1, 2), keepdim=True) + 1e-8)
            signs = torch.sign(delta)
            r = torch.rand(N, 1, 1, device=device)
            delta = signs * abs_delta * r * self.eps
        else:
            raise ValueError(f"Unsupported p={self.p}")

        return (background + delta).detach()


# ------------------------------------------------------------------
# Module-level convenience function（供 LSTMZeroSplitVerifier 直接呼叫）
# ------------------------------------------------------------------

def select_lstm_split_target(verifier, X, top1_class, eps, p,
                              split_history=None, background_size=20):
    """
    一次性：計算 ranking 並回傳單一分割目標 (t, gate, n)。

    Returns:
        (t, gate, n) or (None, None, None)
    """
    locator = LSTMNeuronLocator(verifier, background_size=background_size,
                                eps=eps, p=p)
    ranked = locator.compute_ranking(X, top1_class)
    return locator.select_next_split(ranked, split_history=split_history)
