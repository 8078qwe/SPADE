"""Stage-1 calibration losses.

Implements:

    L_align = 1/|Ω| · Σ_i ‖ A*_i − A'_i ‖_1                         (Eq. 4)
    L_cf    = 1/|C| · Σ_{A_cf} max(0, d(A',A*) − d(A',A_cf) + m)    (Eq. 5)
    L_cal   = λ_a · L_align + λ_c · L_cf                            (Eq. 6)

Defaults match the paper: m = 0.15, (λ_a, λ_c) = (1.0, 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationConfig:
    margin: float = 0.15
    lambda_align: float = 1.0
    lambda_cf:    float = 0.5


def _align_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (student - teacher).abs().mean()


def _l1_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise L1 distance averaged over the spatial / token axes."""
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)))


def calibration_loss(
    student_attn: torch.Tensor,                # A'  (B, HW, L)
    teacher_attn: torch.Tensor,                # A*  (B, HW, L)
    cf_attns: Optional[List[torch.Tensor]] = None,  # list of A_cf
    cfg: CalibrationConfig = CalibrationConfig(),
) -> dict:
    losses: dict = {}
    L_align = _align_loss(student_attn, teacher_attn)
    losses["L_align"] = L_align

    if cf_attns is None or len(cf_attns) == 0:
        L_cf = student_attn.new_zeros(())
    else:
        d_pos = _l1_distance(student_attn, teacher_attn)        # (B,)
        contrib = []
        for A_cf in cf_attns:
            d_neg = _l1_distance(student_attn, A_cf)            # (B,)
            margin = F.relu(d_pos - d_neg + cfg.margin)
            contrib.append(margin)
        L_cf = torch.stack(contrib, dim=0).mean()
    losses["L_cf"] = L_cf

    L_cal = cfg.lambda_align * L_align + cfg.lambda_cf * L_cf
    losses["L_cal"] = L_cal
    return losses


# ---------------------------------------------------------------------------
# Visual MLP adapter (replaces the text-conditioning branch at inference).
# ---------------------------------------------------------------------------
class VisualConditioner(nn.Module):
    """``MLP ∘ CLIP_img(x)`` — Eq. 1 of the paper.

    The student UNet receives this as its conditioning, since at inference
    time no relation prompt is available (the very point of the calibrated
    backbone).
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 768, hidden_dim: int = 2048,
                 num_tokens: int = 77):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * num_tokens),
        )

    def forward(self, clip_image_features: torch.Tensor) -> torch.Tensor:
        """clip_image_features: (B, in_dim) → (B, num_tokens, out_dim)."""
        b = clip_image_features.shape[0]
        x = self.mlp(clip_image_features)
        return x.view(b, self.num_tokens, -1)
