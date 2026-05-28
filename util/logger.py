"""Tiny logger that prints to stdout and (optionally) writes to a file.

Mirrors the layout of CaDM-LQ/util/logger.py but is dependency-free.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logger(name: str = "spade++", *, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
