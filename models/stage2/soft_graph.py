"""Soft spatial-semantic graph (Eq. 7 in the paper).

Edge weights combine geometric and semantic affinity:

    w_ij = β · σ(κ_g · g_ij) + (1 − β) · σ(κ_s · cos(h_i^v, h_j^v))
    g_ij = IoU(m_i, m_j) + exp(− ‖c_i − c_j‖_2 / τ_g)

Edges below 0.1 are pruned. ``κ_g`` and ``κ_s`` are learnable temperatures.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_iou(masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise IoU on (N, H, W) binary masks → (N, N)."""
    N = masks.shape[0]
    m = masks.float().reshape(N, -1)
    inter = m @ m.t()
    area = m.sum(-1, keepdim=True)
    union = area + area.t() - inter
    return inter / union.clamp_min(eps)


def mask_centroid(masks: torch.Tensor) -> torch.Tensor:
    """Return per-mask (cx, cy) of shape (N, 2). Falls back to image center."""
    N, H, W = masks.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=masks.device, dtype=torch.float32),
        torch.arange(W, device=masks.device, dtype=torch.float32),
        indexing="ij",
    )
    m = masks.float()
    s = m.sum(dim=(1, 2)).clamp_min(1.0)
    cx = (m * xx).sum(dim=(1, 2)) / s
    cy = (m * yy).sum(dim=(1, 2)) / s
    return torch.stack([cx, cy], dim=-1)


class SoftGraph(nn.Module):
    def __init__(self, beta: float = 0.5, tau_g: float = 64.0, prune_thresh: float = 0.1):
        super().__init__()
        self.beta = beta
        self.tau_g = tau_g
        self.prune = prune_thresh
        self.kappa_g = nn.Parameter(torch.tensor(1.0))
        self.kappa_s = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_visual: torch.Tensor,        # (N, D_v)
        masks:       torch.Tensor,        # (N, H, W) — binary
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = node_visual.shape[0]
        if N == 0:
            return node_visual.new_zeros(0, 0), node_visual.new_zeros(0, 0, dtype=torch.bool)

        c = mask_centroid(masks)                        # (N, 2)
        d = torch.cdist(c, c)                           # (N, N)
        iou = mask_iou(masks)                           # (N, N)
        g = iou + torch.exp(-d / self.tau_g)            # (N, N)

        v = F.normalize(node_visual, dim=-1)
        sem = v @ v.t()                                 # cosine

        w = (
            self.beta * torch.sigmoid(self.kappa_g * g)
            + (1 - self.beta) * torch.sigmoid(self.kappa_s * sem)
        )
        # Self-loops have weight 1 by convention
        w = w.masked_fill(torch.eye(N, dtype=torch.bool, device=w.device), 1.0)
        # Pruning mask (stored as bool; weights are not zeroed, just signalled).
        keep = w >= self.prune
        return w, keep
