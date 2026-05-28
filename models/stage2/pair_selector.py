"""Learnable pair selector + auxiliary alignment loss.

    s^pair_rj = σ(q̂_r^T W_p q̂_j + b_p)                   (Eq. 11)
    L_rqc     = ‖ Q̂Q̂^T / ‖Q̂‖‖Q̂^T‖ − Ψ_gt ‖_2^2          (Eq. 13)

The pair query is initialised as

    q_pair^{(s,o)} = W_q [q̂_s ‖ q̂_o] + b_q              (Eq. 12)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairSelector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_p = nn.Parameter(torch.empty(dim, dim))
        self.b_p = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W_p)
        self.W_q = nn.Linear(2 * dim, dim)

    def pairability(self, q_hat: torch.Tensor) -> torch.Tensor:
        """Return (N, N) pairability matrix s^pair."""
        proj = q_hat @ self.W_p
        scores = proj @ q_hat.t() + self.b_p
        return torch.sigmoid(scores)

    def init_pair_queries(self, q_hat: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        """``pair_idx`` is a (P, 2) long tensor of (s, o) indices."""
        s_idx, o_idx = pair_idx[:, 0], pair_idx[:, 1]
        cat = torch.cat([q_hat[s_idx], q_hat[o_idx]], dim=-1)
        return self.W_q(cat)

    @staticmethod
    def rqc_loss(q_hat: torch.Tensor, psi_gt: torch.Tensor) -> torch.Tensor:
        """Regularizer of Eq. 13."""
        norm = q_hat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Qn = q_hat / norm
        sim = Qn @ Qn.t()
        return F.mse_loss(sim, psi_gt.float())

    @staticmethod
    def pair_bce_loss(s_pair: torch.Tensor, psi_gt: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(s_pair.clamp(1e-6, 1 - 1e-6), psi_gt.float())
