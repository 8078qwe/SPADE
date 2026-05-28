"""Image transforms used by all dataset wrappers.

Stable-Diffusion expects float tensors in [-1, 1]; CLIP expects normalised
RGB. Our convention: returned tensor is in [-1, 1]; we re-normalise with the
CLIP image processor where needed, downstream of this transform.
"""
from __future__ import annotations

from typing import Callable

import torch
import torchvision.transforms as T


def default_transform(size: int = 512) -> Callable:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
