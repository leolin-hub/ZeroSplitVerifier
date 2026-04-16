#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bound relu(x) * sigma(y) with a linear plane  a*x + b*y + c.

Three cases per neuron (vectorised):
  case1: x_plus  <= 0   →  relu(x) = 0  →  all-zero planes
  case2: x_minus >= 0   →  relu(x) = x  →  delegate to bound_x_sigmoidy
  case3: x_minus <  0 < x_plus  (zero-crossing)
         lower : zero plane          (relu(x)*σ(y) >= 0 always)
         upper : plane from bound_x_sigmoidy on [0, x_plus],
                 constant c raised so the plane stays >= 0 on [x_minus, 0]

Return order: a_l, b_l, c_l, a_u, b_u, c_u
  plane is  a*x + b*y + c  where x = relu argument, y = σ argument.
  (Same interface as bound_x_sigmoidy.main().)
"""

import sys
import os

_lstm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_lstm_dir, 'BoundTanhSigmoidy'))
sys.path.insert(0, os.path.join(_lstm_dir, '..', 'vanilla_rnn'))

import torch
import bound_x_sigmoidy as x_sigmoid
from use_1D_line_bound_2D_activation import line_bounding_2D_activation
from use_constant_bound_2D_activation import constant_bounding_2D_activation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _widen(xm, xp, ym, yp, eps=1e-3):
    """Return cloned, gap-widened copies so optimiser never sees zero-width intervals."""
    xm, xp, ym, yp = xm.clone(), xp.clone(), ym.clone(), yp.clone()
    narrow_x = (xp - xm) < eps
    xp[narrow_x] += eps
    xm[narrow_x] -= eps
    narrow_y = (yp - ym) < eps
    yp[narrow_y] += eps
    ym[narrow_y] -= eps
    return xm, xp, ym, yp


def _adjust_upper_for_neg_x(a_u, b_u, c_u, x_minus, y_minus, y_plus):
    """
    Ensure  a_u*x + b_u*y + c_u >= 0  for x in [x_minus, 0], all y in [y_l, y_u].
    Raises c_u by the worst-case deficit (no-op when already satisfied).
    """
    # worst x for upper plane: x_minus if a_u >= 0 (plane decreases), else x=0
    zeros = torch.zeros_like(x_minus)
    x_worst = torch.where(a_u >= 0, x_minus, zeros)
    # worst y for upper plane: y_minus if b_u >= 0, else y_plus
    y_worst = torch.where(b_u >= 0, y_minus, y_plus)
    min_plane = a_u * x_worst + b_u * y_worst + c_u
    return c_u - torch.clamp(min_plane, max=0)   # raise c_u if min_plane < 0


def _adjust_lower_for_neg_x(a_l, b_l, c_l, x_minus, y_minus, y_plus):
    """
    Ensure  a_l*x + b_l*y + c_l <= 0  for x in [x_minus, 0], all y in [y_l, y_u].
    Lowers c_l by the worst-case excess (no-op when a_l >= 0, which is typical).
    """
    # Only dangerous when a_l < 0: as x goes negative, a_l*x grows positive
    zeros = torch.zeros_like(x_minus)
    x_worst = torch.where(a_l <= 0, x_minus, zeros)   # a_l<0: max plane at x_minus
    y_worst = torch.where(b_l >= 0, y_plus, y_minus)  # b_l>=0: max plane at y_plus
    max_plane = a_l * x_worst + b_l * y_worst + c_l
    return c_l - torch.clamp(max_plane, min=0)         # lower c_l if max_plane > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def main(x_minus, x_plus, y_minus, y_plus,
         use_1D_line=False, use_constant=False, print_info=True):
    """
    Bound  relu(x) * σ(y)  with planes  a*x + b*y + c.

    Args:
        x_minus, x_plus : lower/upper bounds on x  (any shape, e.g. batch×hidden)
        y_minus, y_plus : lower/upper bounds on y
        use_1D_line     : use faster 1-D line bounding (looser)
        use_constant    : use corner-constant bounding (loosest, fastest)
        print_info      : pass-through to underlying optimisers

    Returns:
        a_l, b_l, c_l, a_u, b_u, c_u  — same shape as inputs
    """
    if (x_minus > x_plus).any() or (y_minus > y_plus).any():
        raise ValueError('bounds must satisfy x_minus <= x_plus and y_minus <= y_plus')

    shape  = x_minus.shape
    device = x_minus.device
    zeros  = torch.zeros(shape, device=device)

    a_l = zeros.clone(); b_l = zeros.clone(); c_l = zeros.clone()
    a_u = zeros.clone(); b_u = zeros.clone(); c_u = zeros.clone()

    # ------------------------------------------------------------------
    # Case 1: x_plus <= 0  →  relu(x) = 0  →  planes stay all-zero
    # ------------------------------------------------------------------
    # (initialised above)

    # ------------------------------------------------------------------
    # Case 2: x_minus >= 0  →  relu(x) = x  →  delegate to bound_x_sigmoidy
    # ------------------------------------------------------------------
    case2 = (x_minus >= 0)
    if case2.any():
        xm2, xp2, ym2, yp2 = _widen(
            x_minus[case2], x_plus[case2], y_minus[case2], y_plus[case2])
        al2, bl2, cl2, au2, bu2, cu2 = x_sigmoid.main(
            xm2, xp2, ym2, yp2,
            use_1D_line=use_1D_line, use_constant=use_constant,
            print_info=print_info)
        a_l[case2] = al2; b_l[case2] = bl2; c_l[case2] = cl2
        a_u[case2] = au2; b_u[case2] = bu2; c_u[case2] = cu2

    # ------------------------------------------------------------------
    # Case 3: x_minus < 0 < x_plus  →  zero-crossing
    # ------------------------------------------------------------------
    case3 = (x_minus < 0) & (x_plus > 0)
    if case3.any():
        xm3  = x_minus[case3].clone()
        xp3  = x_plus[case3].clone()
        ym3, yp3 = y_minus[case3].clone(), y_plus[case3].clone()

        # ensure x_plus is meaningfully positive
        xp3 = torch.clamp(xp3, min=1e-3)

        zeros3 = torch.zeros_like(xm3)
        xm3w, xp3w, ym3w, yp3w = _widen(zeros3, xp3, ym3, yp3)

        # --- lower bound: zero plane by default ---
        # try to get a tighter lower bound from bound_x_sigmoidy on [0, x_plus]
        if use_constant:
            al3, bl3, cl3, _, _, _ = constant_bounding_2D_activation(
                xm3w, xp3w, ym3w, yp3w, tanh=False)
        elif use_1D_line:
            al3, bl3, cl3, _, _, _ = line_bounding_2D_activation(
                xm3w, xp3w, ym3w, yp3w, tanh=False)
        else:
            al3, bl3, cl3 = x_sigmoid.main_lower(xm3w, xp3w, ym3w, yp3w,
                                                  print_info=print_info)
        # ensure lower plane <= 0 on [x_minus, 0]
        cl3 = _adjust_lower_for_neg_x(al3, bl3, cl3, xm3, ym3, yp3)

        # --- upper bound: bound_x_sigmoidy on [0, x_plus], then fix x<0 region ---
        if use_constant:
            _, _, _, au3, bu3, cu3 = constant_bounding_2D_activation(
                xm3w, xp3w, ym3w, yp3w, tanh=False)
        elif use_1D_line:
            _, _, _, au3, bu3, cu3 = line_bounding_2D_activation(
                xm3w, xp3w, ym3w, yp3w, tanh=False)
        else:
            au3, bu3, cu3 = x_sigmoid.main_upper(xm3w, xp3w, ym3w, yp3w,
                                                  print_info=print_info)
        # ensure upper plane >= 0 on [x_minus, 0]
        cu3 = _adjust_upper_for_neg_x(au3, bu3, cu3, xm3, ym3, yp3)

        a_l[case3] = al3.detach(); b_l[case3] = bl3.detach(); c_l[case3] = cl3.detach()
        a_u[case3] = au3.detach(); b_u[case3] = bu3.detach(); c_u[case3] = cu3.detach()

    return a_l, b_l, c_l, a_u, b_u, c_u


def validate(a_l, b_l, c_l, a_u, b_u, c_u,
             x_minus, x_plus, y_minus, y_plus,
             num_points=40, eps=1e-5, print_info=True):
    """
    Sample the box and fine-tune c_l / c_u for relu(x)*σ(y).
    Returns updated c_l, c_u (same shape).
    """
    orig_shape = c_l.shape
    a_l_f = a_l.view(-1); b_l_f = b_l.view(-1); c_l_f = c_l.view(-1).clone()
    a_u_f = a_u.view(-1); b_u_f = b_u.view(-1); c_u_f = c_u.view(-1).clone()
    xm_f = x_minus.view(-1); xp_f = x_plus.view(-1)
    ym_f = y_minus.view(-1); yp_f = y_plus.view(-1)

    t = torch.linspace(0, 1, num_points, device=c_l.device)
    for n in range(a_l_f.size(0)):
        xs = xm_f[n] + t * (xp_f[n] - xm_f[n])
        ys = ym_f[n] + t * (yp_f[n] - ym_f[n])
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        Z = torch.relu(X) * torch.sigmoid(Y)

        H_l = a_l_f[n]*X + b_l_f[n]*Y + c_l_f[n]
        H_u = a_u_f[n]*X + b_u_f[n]*Y + c_u_f[n]

        viol_l = (H_l - Z).max().item()   # lower plane above surface → invalid
        viol_u = (Z - H_u).max().item()   # upper plane below surface → invalid

        if print_info:
            print(f'relu*sigmoid validate[{n}]: lower_viol={viol_l:.2e} upper_viol={viol_u:.2e}')

        if viol_l > eps:
            c_l_f[n] -= viol_l * 2
        if viol_u > eps:
            c_u_f[n] -= viol_u * 2

    return c_l_f.view(orig_shape), c_u_f.view(orig_shape)
