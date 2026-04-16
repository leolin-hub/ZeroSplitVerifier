#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My_relu_lstm: verification-side ReLU-LSTM.

Subclasses My_lstm and overrides three methods:
  forward()   — relu(yg) for g gate, identity c for output
  get_hig()   — bound relu(yg)*σ(yi) via bound_relux_sigmoidy
  get_hoc()   — bound c*σ(yo)        via bound_x_sigmoidy

All other methods (get_y, get_c, get_hfc, get_Wa_b, get_Wa_b_one_step, …)
are inherited unchanged — they operate on stored plane coefficients and
do not assume a specific activation.

Usage:
    from lstm_relu import My_relu_lstm
    lstm = My_relu_lstm(model.rnn, device, WF=W_out, bF=b_out, seq_len=T)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from loguru import logger

from lstm import My_lstm, mat_mul          # reuse helpers from lstm.py
import bound_relux_sigmoidy as relux_sigmoid
import bound_x_sigmoidy     as x_sigmoid
from get_bound_for_general_activation_function import getConvenientGeneralActivationBound


class My_relu_lstm(My_lstm):
    """
    ReLU-LSTM verifier.

    Gate equations:
        c_k = σ(yf_k)*c_{k-1}  +  σ(yi_k)*relu(yg_k)
        a_k = σ(yo_k) * c_k                              ← identity on c_k

    Bounding:
        get_hfc : c_{k-1} * σ(yf_k)   →  bound_x_sigmoidy    (inherited)
        get_hig : relu(yg_k) * σ(yi_k) →  bound_relux_sigmoidy (overridden)
        get_hoc : c_k * σ(yo_k)        →  bound_x_sigmoidy    (overridden)
    """

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x, a0=None, c0=None, use_x_seq_len=False):
        with torch.no_grad():
            if use_x_seq_len:
                seq_len = x.shape[1]
            else:
                seq_len = x.shape[1]
                if self.seq_len is not None and seq_len != self.seq_len:
                    raise Exception(
                        'seq_len mismatch: got %d, expected %d' % (seq_len, self.seq_len))

            batch = x.shape[0]
            a  = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            c  = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yi = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yf = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yg = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yo = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)

            for k in range(seq_len):
                # ---------- gate pre-activations ----------
                if k > 0:
                    prev_a = a[:, k-1, :]
                elif a0 is not None:
                    prev_a = a0
                elif self.a0 is not None:
                    prev_a = self.a0
                else:
                    prev_a = None

                if prev_a is not None:
                    yi[:, k, :] = mat_mul(self.Wix, x[:, k, :], self.bi) + mat_mul(self.Wia, prev_a)
                    yf[:, k, :] = mat_mul(self.Wfx, x[:, k, :], self.bf) + mat_mul(self.Wfa, prev_a)
                    yg[:, k, :] = mat_mul(self.Wgx, x[:, k, :], self.bg) + mat_mul(self.Wga, prev_a)
                    yo[:, k, :] = mat_mul(self.Wox, x[:, k, :], self.bo) + mat_mul(self.Woa, prev_a)
                else:
                    yi[:, k, :] = mat_mul(self.Wix, x[:, k, :], self.bi)
                    yf[:, k, :] = mat_mul(self.Wfx, x[:, k, :], self.bf)
                    yg[:, k, :] = mat_mul(self.Wgx, x[:, k, :], self.bg)
                    yo[:, k, :] = mat_mul(self.Wox, x[:, k, :], self.bo)

                # ---------- cell & hidden state ----------
                if k > 0:
                    c_prev = c[:, k-1, :]
                elif c0 is not None:
                    c_prev = c0
                elif self.c0 is not None:
                    c_prev = self.c0
                else:
                    c_prev = torch.zeros(batch, self.hidden_size, device=x.device)

                c[:, k, :] = (torch.sigmoid(yf[:, k, :]) * c_prev
                              + torch.sigmoid(yi[:, k, :]) * torch.relu(yg[:, k, :]))   # relu g gate
                a[:, k, :] = torch.sigmoid(yo[:, k, :]) * c[:, k, :]                   # identity on c

        return a, c, yi, yf, yg, yo

    # ------------------------------------------------------------------
    # get_hig: bound relu(yg_k) * σ(yi_k)
    # ------------------------------------------------------------------

    def get_hig(self, m):
        _yg_inv = (self.yg_u[m-1] - self.yg_l[m-1] < 0)
        _yi_inv = (self.yi_u[m-1] - self.yi_l[m-1] < 0)
        if _yg_inv.any():
            viol = (self.yg_l[m-1] - self.yg_u[m-1])[_yg_inv].max().item()
            logger.info(f'[relu get_hig] t={m}: yg inversion {_yg_inv.sum()} neurons, max={viol:.3e}')
            assert viol < 1e-4, f'get_hig yg large inversion {viol:.3e}'
        if _yi_inv.any():
            viol = (self.yi_l[m-1] - self.yi_u[m-1])[_yi_inv].max().item()
            logger.info(f'[relu get_hig] t={m}: yi inversion {_yi_inv.sum()} neurons, max={viol:.3e}')
            assert viol < 1e-4, f'get_hig yi large inversion {viol:.3e}'

        # relux_sigmoid.main returns (a_l, b_l, c_l, a_u, b_u, c_u)
        #   a = coeff for x = yg  (relu arg)   → beta_ig in lstm.py
        #   b = coeff for y = yi  (sigma arg)  → alpha_ig in lstm.py
        # Unpack in the same (b, a, c) swap used by lstm.py for bound_tanhx_sigmoidy:
        b_l, a_l, c_l, b_u, a_u, c_u = relux_sigmoid.main(
            self.yg_l[m-1], self.yg_u[m-1],
            self.yi_l[m-1], self.yi_u[m-1],
            use_1D_line=self.use_1D_line,
            use_constant=self.use_constant,
            print_info=self.print_info)

        self.alpha_l_ig[m-1] = a_l.detach()   # coeff for yi
        self.alpha_u_ig[m-1] = a_u.detach()
        self.beta_l_ig[m-1]  = b_l.detach()   # coeff for yg
        self.beta_u_ig[m-1]  = b_u.detach()
        self.gamma_l_ig[m-1] = c_l.detach()
        self.gamma_u_ig[m-1] = c_u.detach()
        return a_l, b_l, c_l, a_u, b_u, c_u

    # ------------------------------------------------------------------
    # get_hoc: bound c_k * σ(yo_k)   (identity output, no tanh on c)
    # ------------------------------------------------------------------

    def get_hoc(self, m):
        _c_inv  = (self.c_u[m-1]  - self.c_l[m-1]  < 0)
        _yo_inv = (self.yo_u[m-1] - self.yo_l[m-1] < 0)
        if _c_inv.any():
            viol = (self.c_l[m-1] - self.c_u[m-1])[_c_inv].max().item()
            logger.info(f'[relu get_hoc] t={m}: c inversion {_c_inv.sum()} neurons, max={viol:.3e}')
            assert viol < 1e-4, f'get_hoc c large inversion {viol:.3e}'
        if _yo_inv.any():
            viol = (self.yo_l[m-1] - self.yo_u[m-1])[_yo_inv].max().item()
            logger.info(f'[relu get_hoc] t={m}: yo inversion {_yo_inv.sum()} neurons, max={viol:.3e}')
            assert viol < 1e-4, f'get_hoc yo large inversion {viol:.3e}'

        # bound  c_k * σ(yo_k)  — same form as get_hfc (x_sigmoidy)
        # x_sigmoid.main returns (a_l, b_l, c_l, a_u, b_u, c_u)
        #   a = coeff for x = c   → beta_oc
        #   b = coeff for y = yo  → alpha_oc
        b_l, a_l, c_l, b_u, a_u, c_u = x_sigmoid.main(
            self.c_l[m-1].detach(), self.c_u[m-1].detach(),
            self.yo_l[m-1].detach(), self.yo_u[m-1].detach(),
            use_1D_line=self.use_1D_line,
            use_constant=self.use_constant,
            print_info=self.print_info)

        self.alpha_l_oc[m-1] = a_l.detach()   # coeff for yo
        self.alpha_u_oc[m-1] = a_u.detach()
        self.beta_l_oc[m-1]  = b_l.detach()   # coeff for c
        self.beta_u_oc[m-1]  = b_u.detach()
        self.gamma_l_oc[m-1] = c_l.detach()
        self.gamma_u_oc[m-1] = c_u.detach()
        return a_l, b_l, c_l, a_u, b_u, c_u
