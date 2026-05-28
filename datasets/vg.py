"""Visual Genome (open-vocab SGG) wrapper.

Same on-disk format as :mod:`datasets.psg` for simplicity; downstream code
treats them interchangeably.
"""
from .psg import PSGDataset


class VGDataset(PSGDataset):
    """Visual-Genome dataset; identical loader to PSG by design."""
    pass
