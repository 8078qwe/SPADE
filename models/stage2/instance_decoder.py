"""Mask2Former-style instance decoder (lightweight stub).

For brevity we expose the *interface* SPADE++ uses but back it with a thin
wrapper around a generic transformer decoder. In production one would plug
in the full Mask2Former implementation from Cheng et al. 2022.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class InstanceDecoder(nn.Module):
    """Returns (H_o, masks, class_logits) given multi-scale UNet features.

    Args:
        feat_dim:      channels of the input feature map ``F``.
        hidden_dim:    decoder hidden size.
        num_queries:   ``N``, number of object queries.
        num_classes:   open-vocabulary head returns logits of this size, but
                       in practice we score by cosine similarity to CLIP text
                       embeddings, so this dim is the CLIP embedding size.
    """

    def __init__(self, feat_dim: int = 320, hidden_dim: int = 256,
                 num_queries: int = 100, num_classes: int = 768,
                 num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.q_embed = nn.Embedding(num_queries, hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.mask_head = nn.Linear(hidden_dim, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.feat_proj_for_mask = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, F_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """F_map: (B, C, H, W). For SPADE++ we always run with B = 1."""
        B, _, H, W = F_map.shape
        f = self.proj(F_map)                              # (B, D, H, W)
        memory = f.flatten(2).transpose(1, 2)             # (B, HW, D)
        q = self.q_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        h = self.decoder(q, memory)                       # (B, N, D)
        # Mask = inner product between query and per-pixel feature.
        mask_q = self.mask_head(h)                                       # (B, N, D)
        feat_for_mask = self.feat_proj_for_mask(f).flatten(2)            # (B, D, HW)
        mask_logits = torch.einsum("bnd,bdp->bnp", mask_q, feat_for_mask)
        masks = mask_logits.reshape(B, -1, H, W)
        cls = self.cls_head(h)                                           # (B, N, C)
        return h, masks, cls
