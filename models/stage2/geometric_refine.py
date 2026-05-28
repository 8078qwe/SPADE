"""Geometry-conditioned pair refinement (§3.2.2 of the paper).

Two refinements correct the symmetric pair query:

    1. Union-region refinement (Eq. 14, 15) — gated residual with RoI-Aligned
       feature pooled over the union bounding box of mask_s ∪ mask_o.
    2. Role-aware geometric gating (Eq. 16, 17) — multiplies the query by a
       descriptor-conditioned gate so q_rel^{(s,o)} ≠ q_rel^{(o,s)}.

The directional descriptor is

    g^{(s,o)} = [ c_s − c_o  ‖  ‖c_s − c_o‖₂  ‖
                  IoU(m_s, m_o)  ‖  area(m_s)/area(m_o) ]
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class UnionRegionRefinement(nn.Module):
    def __init__(self, feat_dim: int, query_dim: int, output_size: int = 7):
        super().__init__()
        self.output_size = output_size
        self.flatten = nn.Flatten(1)
        in_dim = feat_dim * output_size * output_size
        self.W_u = nn.Linear(in_dim, query_dim)
        self.W_u_prime = nn.Linear(in_dim, query_dim)

    @staticmethod
    def union_bbox(box_s: torch.Tensor, box_o: torch.Tensor) -> torch.Tensor:
        x0 = torch.minimum(box_s[..., 0], box_o[..., 0])
        y0 = torch.minimum(box_s[..., 1], box_o[..., 1])
        x1 = torch.maximum(box_s[..., 2], box_o[..., 2])
        y1 = torch.maximum(box_s[..., 3], box_o[..., 3])
        return torch.stack([x0, y0, x1, y1], dim=-1)

    def forward(
        self,
        q_pair: torch.Tensor,           # (P, query_dim)
        feature_map: torch.Tensor,      # (1, C, H, W)
        boxes_s: torch.Tensor,          # (P, 4)  xyxy in feature-map coords
        boxes_o: torch.Tensor,
        spatial_scale: float = 1.0,
    ) -> torch.Tensor:
        union = self.union_bbox(boxes_s, boxes_o)                    # (P, 4)
        rois = torch.cat([torch.zeros_like(union[:, :1]), union], dim=-1)  # batch idx = 0
        u = roi_align(feature_map, rois, output_size=self.output_size,
                      spatial_scale=spatial_scale, sampling_ratio=2, aligned=True)
        u = self.flatten(u)                                          # (P, C·k·k)
        gate = torch.sigmoid(self.W_u(u))
        value = self.W_u_prime(u)
        return q_pair + gate * value


class RoleAwareGate(nn.Module):
    def __init__(self, query_dim: int, descriptor_dim: int = 5):
        super().__init__()
        self.W_g = nn.Linear(descriptor_dim, query_dim)

    @staticmethod
    def descriptor(box_s: torch.Tensor, box_o: torch.Tensor,
                   mask_iou_so: torch.Tensor) -> torch.Tensor:
        c_s = torch.stack([(box_s[..., 0] + box_s[..., 2]) / 2,
                           (box_s[..., 1] + box_s[..., 3]) / 2], dim=-1)
        c_o = torch.stack([(box_o[..., 0] + box_o[..., 2]) / 2,
                           (box_o[..., 1] + box_o[..., 3]) / 2], dim=-1)
        diff = c_s - c_o                                             # (P, 2)
        dist = diff.norm(dim=-1, keepdim=True)                       # (P, 1)
        a_s = (box_s[..., 2] - box_s[..., 0]) * (box_s[..., 3] - box_s[..., 1])
        a_o = (box_o[..., 2] - box_o[..., 0]) * (box_o[..., 3] - box_o[..., 1])
        ratio = (a_s / a_o.clamp_min(1e-6)).unsqueeze(-1)
        iou = mask_iou_so.unsqueeze(-1)
        return torch.cat([diff, dist, iou, ratio], dim=-1)            # (P, 5)

    def forward(self, q_pair: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W_g(descriptor))
        return q_pair * gate
