"""Bounding-box ops, mostly forwarded from torchvision but kept local for the
same-style API as CaDM-LQ/util/box_ops.py."""
from __future__ import annotations

import torch
from torchvision.ops import box_iou as _box_iou
from torchvision.ops import generalized_box_iou as _giou


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    return _box_iou(b1, b2)


def generalized_box_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    return _giou(b1, b2)
