"""Stage-2 model: ``SPADEppStage2``.

Glues together the pieces of §3.2 of the paper:

    F (frozen Stage 1) → InstanceDecoder → SoftGraph → RGT → PairSelector
                       → UnionRegionRefinement → RoleAwareGate
                       → RelationDecoder → DecisionLevelContrast → ScoreFusion
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decision_contrast import (
    DecisionLevelContrast,
    cosine_softmax,
    score_fusion,
    score_fusion_object,
)
from .geometric_refine import RoleAwareGate, UnionRegionRefinement
from .instance_decoder import InstanceDecoder
from .pair_selector import PairSelector
from .rgt import RGT
from .soft_graph import SoftGraph, mask_centroid, mask_iou


# ---------------------------------------------------------------------------
@dataclass
class Stage2Config:
    feat_dim: int = 320
    hidden_dim: int = 256
    num_queries: int = 100
    rgt_layers: int = 8
    geo_descriptor_dim: int = 5
    semantic_dim: int = 768           # CLIP text embedding size
    geometric_dim: int = 128
    alpha: float = 0.34
    gamma: float = 0.2
    lambda_rqc: float = 1.0
    lambda_mask: float = 5.0
    lambda_cfg: float = 1.0
    top_k_inf: int = 3


# ---------------------------------------------------------------------------
class GeometricFeatureMLP(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, boxes_xyxy: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        H, W = image_size
        x = boxes_xyxy.clone().float()
        x[..., 0::2] /= max(W, 1)
        x[..., 1::2] /= max(H, 1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
class SPADEppStage2(nn.Module):
    def __init__(self, cfg: Stage2Config = Stage2Config()):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim + cfg.semantic_dim + cfg.geometric_dim
        self.D = D

        self.instance_decoder = InstanceDecoder(
            feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
            num_queries=cfg.num_queries, num_classes=cfg.semantic_dim,
        )
        self.geom_mlp = GeometricFeatureMLP(cfg.geometric_dim)
        self.soft_graph = SoftGraph()
        self.rgt = RGT(D, num_layers=cfg.rgt_layers)
        self.pair_selector = PairSelector(D)
        self.union_refine = UnionRegionRefinement(cfg.feat_dim, D, output_size=7)
        self.role_gate = RoleAwareGate(D, descriptor_dim=cfg.geo_descriptor_dim)

        # Lightweight relation transformer ϕ_rel — single-layer is enough since
        # the heavy lifting happens in RGT and the geometric gates.
        rel_layer = nn.TransformerDecoderLayer(D, nhead=8, batch_first=True)
        self.rel_decoder = nn.TransformerDecoder(rel_layer, num_layers=2)
        self.rel_proj = nn.Linear(D, cfg.semantic_dim)

        self.cf_head = DecisionLevelContrast(h_dim=cfg.semantic_dim, e_dim=cfg.semantic_dim)
        self.tau = nn.Parameter(torch.tensor(2.0))

    # ------------------------------------------------------------------ stage 2
    def forward(
        self,
        F_map: torch.Tensor,                      # (1, C, H, W) calibrated UNet feature
        clip_text_obj: torch.Tensor,              # (Co, D) object-prompt text embeddings
        clip_text_rel: torch.Tensor,              # (Cr, D) predicate-prompt text embeddings
        targets: Optional[Dict] = None,           # ground-truth dict (training only)
        clip_image_pooled: Optional[torch.Tensor] = None,  # (1, C, H, W) for P'_r aux scores
        cf_indices: Optional[torch.Tensor] = None,    # (P,) hard-negative predicate idx
        image_size: Tuple[int, int] = (512, 512),
    ) -> Dict[str, torch.Tensor]:
        H_obj, masks, cls_logits = self.instance_decoder(F_map)
        H_obj = H_obj[0]            # (N, hidden)
        masks = masks[0]            # (N, H, W)
        cls_logits = cls_logits[0]  # (N, semantic_dim)

        # Object scoring (open-vocabulary, Eq. 22)
        P_o = cosine_softmax(cls_logits, clip_text_obj, self.tau)

        # Build per-node features h_i = [h_v ‖ h_s ‖ h_g]
        binary_masks = (masks.sigmoid() > 0.5).float()
        boxes = self._mask_to_box(binary_masks).clamp(min=0)
        h_v = H_obj
        h_s = (P_o @ clip_text_obj)                         # CLIP text mixture
        h_g = self.geom_mlp(boxes, image_size)
        h = torch.cat([h_v, h_s, h_g], dim=-1)              # (N, D)

        # Soft graph and edge-weighted RGT
        w, _ = self.soft_graph(h_v, binary_masks)
        q_hat = self.rgt(h, w)                              # (N, D)

        # Learnable pair selection
        s_pair = self.pair_selector.pairability(q_hat)       # (N, N)

        # During training we use ground-truth pairs; at inference we use top-K.
        if self.training and targets is not None and "pair_idx" in targets:
            pair_idx = targets["pair_idx"]
        else:
            mask = s_pair > 0.5
            pair_idx = mask.nonzero(as_tuple=False)
            if pair_idx.shape[0] == 0:
                pair_idx = torch.zeros(0, 2, dtype=torch.long, device=q_hat.device)

        if pair_idx.shape[0] == 0:
            return {"P_o": P_o, "masks": masks, "P_r": q_hat.new_zeros(0, clip_text_rel.shape[0])}

        q_pair = self.pair_selector.init_pair_queries(q_hat, pair_idx)

        # Geometric refinement
        boxes_s = boxes[pair_idx[:, 0]]
        boxes_o = boxes[pair_idx[:, 1]]
        q_pair = self.union_refine(q_pair, F_map, boxes_s, boxes_o, spatial_scale=1.0)
        m_iou = mask_iou(binary_masks)[pair_idx[:, 0], pair_idx[:, 1]]
        descr = self.role_gate.descriptor(boxes_s, boxes_o, m_iou)
        q_rel = self.role_gate(q_pair, descr)               # (P, D)

        # Relation decoder ϕ_rel
        memory = q_hat.unsqueeze(0)                        # (1, N, D)
        h_rel = self.rel_decoder(q_rel.unsqueeze(0), memory).squeeze(0)
        h_rel = self.rel_proj(h_rel)                       # (P, semantic_dim)

        # Open-vocabulary relation scoring (Eq. 23)
        P_r = cosine_softmax(h_rel, clip_text_rel, self.tau)

        # Auxiliary CLIP-pooled prediction P'_r (Eq. 25); requires the CLIP
        # image-feature map to be supplied. If not given, fall back to P_r.
        if clip_image_pooled is not None:
            H_r_prime = self._roi_pool_clip(clip_image_pooled, boxes_s, boxes_o, masks)
            P_r_prime = cosine_softmax(H_r_prime, clip_text_rel, self.tau)
        else:
            P_r_prime = P_r

        # Decision-level discrimination Δ̄
        if cf_indices is None:
            # At inference, take top-K=3 CLIP neighbors of the predicted predicate.
            _, topk = P_r.topk(self.cfg.top_k_inf, dim=-1)            # (P, K)
            delta_list = []
            preds = P_r.argmax(dim=-1)
            for k in range(self.cfg.top_k_inf):
                pcf = topk[:, k]
                e_p = clip_text_rel[preds]
                e_pcf = clip_text_rel[pcf]
                d = self.cf_head.discrimination(h_rel, e_p, e_pcf)    # (P,)
                delta_list.append(d)
            delta = torch.stack(delta_list, dim=-1).mean(-1, keepdim=True).expand_as(P_r)
        else:
            e_p = clip_text_rel[targets["pred_idx"]] if (self.training and targets is not None and "pred_idx" in targets) else clip_text_rel[P_r.argmax(-1)]
            e_pcf = clip_text_rel[cf_indices]
            d = self.cf_head.discrimination(h_rel, e_p, e_pcf)
            delta = d.unsqueeze(-1).expand_as(P_r)

        P_r_final = score_fusion(P_r, P_r_prime, delta,
                                 alpha=self.cfg.alpha, gamma=self.cfg.gamma)

        out = {
            "P_o": P_o,
            "P_r": P_r_final,
            "P_r_raw": P_r,
            "P_r_aux": P_r_prime,
            "delta": delta,
            "masks": masks,
            "boxes": boxes,
            "pair_idx": pair_idx,
            "h_rel": h_rel,
            "q_hat": q_hat,
            "s_pair": s_pair,
        }

        # ---- Losses (training only) -----------------------------------
        if self.training and targets is not None:
            losses = self._compute_losses(out, targets, clip_text_rel)
            out["losses"] = losses
        return out

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _mask_to_box(masks: torch.Tensor) -> torch.Tensor:
        """Convert (N, H, W) binary masks to (N, 4) xyxy boxes."""
        N, H, W = masks.shape
        if N == 0:
            return masks.new_zeros(0, 4)
        boxes = []
        for m in masks:
            ys, xs = torch.where(m > 0)
            if ys.numel() == 0:
                boxes.append(torch.tensor([0., 0., float(W), float(H)], device=masks.device))
            else:
                boxes.append(torch.stack([xs.min().float(), ys.min().float(),
                                          xs.max().float() + 1, ys.max().float() + 1]))
        return torch.stack(boxes, 0)

    @staticmethod
    def _roi_pool_clip(clip_feat: torch.Tensor, box_s: torch.Tensor, box_o: torch.Tensor,
                       masks: torch.Tensor) -> torch.Tensor:
        """Mask-weighted average pool of CLIP image features over the union region."""
        from torchvision.ops import roi_align
        x0 = torch.minimum(box_s[..., 0], box_o[..., 0])
        y0 = torch.minimum(box_s[..., 1], box_o[..., 1])
        x1 = torch.maximum(box_s[..., 2], box_o[..., 2])
        y1 = torch.maximum(box_s[..., 3], box_o[..., 3])
        union = torch.stack([x0, y0, x1, y1], dim=-1)
        rois = torch.cat([torch.zeros_like(union[:, :1]), union], dim=-1)
        f = roi_align(clip_feat, rois, output_size=1, spatial_scale=1.0,
                      sampling_ratio=2, aligned=True)
        return f.flatten(1)

    def _compute_losses(self, out, targets, clip_text_rel) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        losses: Dict[str, torch.Tensor] = {}
        # Classification cross-entropy on the fused score (Eq. 21).
        if "rel_label" in targets:
            losses["L_rel"] = F.cross_entropy(out["P_r"].clamp_min(1e-8).log(), targets["rel_label"])
        # Pair alignment regularizer Eq. 13.
        if "psi_gt" in targets:
            losses["L_rqc"] = cfg.lambda_rqc * self.pair_selector.rqc_loss(out["q_hat"], targets["psi_gt"])
        # Mask2Former mask loss (placeholder: BCE on coarse mask logits)
        if "mask_gt" in targets:
            mask_gt = targets["mask_gt"].float()
            losses["L_mask"] = cfg.lambda_mask * F.binary_cross_entropy_with_logits(
                out["masks"], mask_gt
            )
        # Decision-level contrast Eq. 20.
        if "rel_label" in targets and "rel_label_cf" in targets:
            e_p = clip_text_rel[targets["rel_label"]]
            e_pcf = clip_text_rel[targets["rel_label_cf"]]
            losses["L_cf_g"] = cfg.lambda_cfg * self.cf_head.loss(out["h_rel"], e_p, e_pcf)
        losses["total"] = sum(v for k, v in losses.items() if k != "total")
        return losses
