"""Misc helpers."""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict]) -> Dict:
    """Stack tensors when shapes match; keep ragged ones as lists."""
    out: Dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor) and all(v.shape == vals[0].shape for v in vals):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
