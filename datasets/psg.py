"""Generic Triple/PSG-style dataset wrapper.

Designed to be a drop-in for the PSG dataset structure used by SPADE++. The
real implementation should mirror our CaDM-LQ ``hico.py`` style; this stub
contains the minimum loader logic so the training pipeline runs end-to-end.

JSONL format expected per line::

    {
      "image_id":   123,
      "file_name":  "000000123.jpg",
      "height":     480,
      "width":      640,
      "objects":    [{"label": "person", "box": [x0,y0,x1,y1], "mask_rle": ...}, ...],
      "relations":  [[s_idx, p_label, o_idx], ...]
    }
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import default_transform


class PSGDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, *,
                 predicate_vocab: List[str], object_vocab: List[str],
                 image_size: int = 512):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        self.image_root = image_root
        self.predicate_vocab = predicate_vocab
        self.object_vocab = object_vocab
        self.p2i = {p: i for i, p in enumerate(predicate_vocab)}
        self.o2i = {o: i for i, o in enumerate(object_vocab)}
        self.transform = default_transform(image_size)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        img = Image.open(os.path.join(self.image_root, it["file_name"])).convert("RGB")
        x = self.transform(img)

        objects = it.get("objects", [])
        labels_o = torch.tensor([self.o2i.get(o["label"], 0) for o in objects], dtype=torch.long)
        boxes = torch.tensor([o["box"] for o in objects], dtype=torch.float32) if objects else torch.zeros(0, 4)
        relations = it.get("relations", [])
        rel_pairs = torch.tensor([[r[0], r[2]] for r in relations], dtype=torch.long) \
            if relations else torch.zeros(0, 2, dtype=torch.long)
        rel_labels = torch.tensor([self.p2i.get(r[1], 0) for r in relations], dtype=torch.long) \
            if relations else torch.zeros(0, dtype=torch.long)

        return {
            "image":      x,
            "image_id":   it["image_id"],
            "object_boxes":  boxes,
            "object_labels": labels_o,
            "rel_pairs":     rel_pairs,
            "rel_labels":    rel_labels,
            "image_size":    (it.get("height", self.image_size), it.get("width", self.image_size)),
        }
