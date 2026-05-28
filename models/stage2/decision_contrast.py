"""Decision-level contrast head + score fusion (§3.2.3).

Compatibility (Eq. 18):

    f_ψ(h_r, e_p) = w^T σ(W_1 h_r + W_2 e_p + b)

Discrimination score (Eq. 19):

    Δ(s,o; p, p_cf) = σ(f_ψ(h_r, e_p) − f_ψ(h_r, e_{p_cf}))

Loss (Eq. 20, symmetric BCE):

    L_cf-g = − E[ log Δ(s,o; p, p_cf) + log(1 − Δ(s,o; p_cf, p)) ]

Score fusion (Eq. 21):

    P_r^final = (P_r)^α · (P_r')^{1-α} · (Δ̄)^γ,    α = 0.34, γ = 0.2
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompatibilityMLP(nn.Module):
    def __init__(self, h_dim: int, e_dim: int, hidden: int = 512):
        super().__init__()
        self.W_h = nn.Linear(h_dim, hidden)
        self.W_e = nn.Linear(e_dim, hidden)
        self.b = nn.Parameter(torch.zeros(hidden))
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """h: (P, h_dim), e: (P, e_dim) or (1, e_dim) → (P,)."""
        if e.shape[0] == 1 and h.shape[0] != 1:
            e = e.expand(h.shape[0], -1)
        z = F.relu(self.W_h(h) + self.W_e(e) + self.b)
        return self.w(z).squeeze(-1)


class DecisionLevelContrast(nn.Module):
    def __init__(self, h_dim: int, e_dim: int):
        super().__init__()
        self.compat = CompatibilityMLP(h_dim, e_dim)

    def discrimination(
        self,
        h_r: torch.Tensor,             # (P, h_dim)
        e_p: torch.Tensor,             # (P, e_dim)
        e_pcf: torch.Tensor,           # (P, e_dim)
    ) -> torch.Tensor:
        f_p = self.compat(h_r, e_p)
        f_pcf = self.compat(h_r, e_pcf)
        return torch.sigmoid(f_p - f_pcf)

    def loss(self, h_r: torch.Tensor, e_p: torch.Tensor, e_pcf: torch.Tensor) -> torch.Tensor:
        d_pos = self.discrimination(h_r, e_p, e_pcf)
        d_neg = self.discrimination(h_r, e_pcf, e_p)
        eps = 1e-6
        return -(torch.log(d_pos.clamp_min(eps)) + torch.log((1 - d_neg).clamp_min(eps))).mean()


# ---------------------------------------------------------------------------
# Open-vocabulary classification + score fusion
# ---------------------------------------------------------------------------
def cosine_softmax(features: torch.Tensor, prompts_emb: torch.Tensor,
                   tau: torch.Tensor) -> torch.Tensor:
    f = F.normalize(features, dim=-1)
    p = F.normalize(prompts_emb, dim=-1)
    return F.softmax(tau * f @ p.t(), dim=-1)


def score_fusion(
    P_r:        torch.Tensor,    # (P, C) primary diffusion-grounded prediction
    P_r_prime:  torch.Tensor,    # (P, C) auxiliary CLIP-pooled prediction
    delta_bar:  torch.Tensor,    # (P, C) averaged discrimination over top-K cf
    *,
    alpha: float = 0.34,
    gamma: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eq. 21. Operates in log-space for numerical stability."""
    log_P = (
        alpha * torch.log(P_r.clamp_min(eps))
        + (1 - alpha) * torch.log(P_r_prime.clamp_min(eps))
        + gamma * torch.log(delta_bar.clamp_min(eps))
    )
    return torch.softmax(log_P, dim=-1)


def score_fusion_object(P_o: torch.Tensor, P_o_prime: torch.Tensor,
                        *, alpha: float = 0.34, eps: float = 1e-8) -> torch.Tensor:
    log_P = alpha * torch.log(P_o.clamp_min(eps)) + (1 - alpha) * torch.log(P_o_prime.clamp_min(eps))
    return torch.softmax(log_P, dim=-1)
