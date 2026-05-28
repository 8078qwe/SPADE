"""HICO-DET dataset wrapper, mirroring CaDM-LQ/datasets/hico.py.

Only the surface API is reproduced here; the inner annotation parsing follows
the PPDM JSON format (``trainval_hico.json``, ``test_hico.json``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import default_transform


class HICODataset(Dataset):
    def __init__(self, ann_file: str, image_root: str, *, num_queries: int = 64,
                 image_size: int = 512):
        with open(ann_file, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, Any]] = json.load(f)
        self.image_root = image_root
        self.num_queries = num_queries
        self.transform = default_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = Image.open(os.path.join(self.image_root, rec["file_name"])).convert("RGB")
        x = self.transform(img)
        annotations = rec.get("annotations", [])  # list of per-object dicts
        hois = rec.get("hoi_annotation", [])
        boxes = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32) \
            if annotations else torch.zeros(0, 4)
        labels = torch.tensor([a["category_id"] for a in annotations], dtype=torch.long) \
            if annotations else torch.zeros(0, dtype=torch.long)
        return {
            "image": x,
            "boxes": boxes,
            "labels": labels,
            "hoi_annotations": hois,
            "file_name": rec["file_name"],
        }
