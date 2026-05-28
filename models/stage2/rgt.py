"""Edge-weighted Relation Graph Transformer (RGT).

Implements Eqs. 8-10 of the paper. Compared to SPADE's RGT we add the standard
``√d`` attention scaling for stability and use the soft-graph edge weights as
*message biases* for both neighbor (P+) and non-neighbor (P-) aggregation.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedNeighborAttn(nn.Module):
    """Scaled dot-product attention with weighted neighbor mean as the key/value.

    .. math::
        g_r^+ = softmax((W_K \\bar h_r^+)^T (W_Q q_r) / √d) · W_V q_r
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, q: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """q: (N, D), h_bar: (N, D) — already aggregated neighbor mean."""
        Q = self.W_q(q)
        K = self.W_k(h_bar)
        V = self.W_v(q)
        # Token-wise: each node attends only to its own aggregated neighbor mean.
        attn = (K * Q).sum(-1, keepdim=True) * self.scale
        return torch.softmax(attn, dim=0) * V


class GCNLayer(nn.Module):
    """A single graph-convolution layer (Kipf & Welling 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        deg = w.sum(-1, keepdim=True).clamp_min(1e-6)
        agg = (w / deg) @ x
        return F.relu(self.W(agg))


class RGTBlock(nn.Module):
    """One block of edge-weighted RGT followed by a GCN refinement."""

    def __init__(self, dim: int):
        super().__init__()
        self.pos_attn = WeightedNeighborAttn(dim)
        self.neg_attn = WeightedNeighborAttn(dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.gcn = GCNLayer(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        if N == 0:
            return q
        # Positive (neighbor) and negative (non-neighbor) weights.
        w_pos = w
        w_neg = (1.0 - w).clamp_min(0.0)
        deg_pos = w_pos.sum(-1, keepdim=True).clamp_min(1e-6)
        deg_neg = w_neg.sum(-1, keepdim=True).clamp_min(1e-6)
        h_pos = (w_pos / deg_pos) @ q                # (N, D)  weighted neighbor mean
        h_neg = (w_neg / deg_neg) @ q

        g_pos = self.pos_attn(q, h_pos)
        g_neg = self.neg_attn(q, h_neg)

        merged = torch.cat([q + g_pos + g_neg, h_pos, h_neg], dim=-1)  # (N, 3D)
        q1 = self.ln1(q + self.mlp(merged))
        q2 = self.ln2(q1 + self.gcn(q1, w))
        return q2


class RGT(nn.Module):
    """Stack of RGT blocks (8 by default, paper §3.2)."""

    def __init__(self, dim: int, num_layers: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([RGTBlock(dim) for _ in range(num_layers)])

    def forward(self, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q = blk(q, w)
        return q
