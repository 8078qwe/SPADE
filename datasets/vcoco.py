"""V-COCO dataset wrapper. Skeleton mirroring the CaDM-LQ/vcoco.py layout."""
from __future__ import annotations

from .hico import HICODataset


class VCOCODataset(HICODataset):
    """V-COCO uses an HOIA-format JSON identical in structure to HICO-DET."""
    pass
