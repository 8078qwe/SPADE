"""SPADE++ unified training/evaluation engine.

Mirrors the role of CaDM-LQ/engine.py: provides ``train_one_epoch`` and
``evaluate``. Both Stage 1 and Stage 2 use the same engine, switching on
``cfg.stage``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, optimizer, *, device, logger,
                    epoch: int, log_interval: int = 50,
                    grad_accum: int = 1, max_norm: float = 1.0,
                    scaler: torch.cuda.amp.GradScaler = None) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = _move(batch, device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            losses = model(**_unpack(batch, model))
        loss = losses["L_cal"] if "L_cal" in losses else losses.get("losses", {}).get("total", losses.get("total"))
        if loss is None:
            raise RuntimeError(f"No 'total' loss returned by model. Got keys: {list(losses.keys())}")
        if scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        for k, v in losses.items() if isinstance(losses, dict) else []:
            if torch.is_tensor(v):
                running[k] = running.get(k, 0.0) + v.item()
        if (step + 1) % log_interval == 0:
            msg = " ".join(f"{k}={running[k] / (step + 1):.4f}" for k in running)
            logger.info(f"[ep {epoch}][step {step + 1}] {msg}")
    return {k: v / max(len(loader), 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(model, loader: DataLoader, *, device, logger) -> Dict[str, float]:
    model.eval()
    metric_acc = 0.0
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        out = model(**_unpack(batch, model, training=False))
        if isinstance(out, dict) and "P_r" in out and "rel_labels" in batch:
            P = out["P_r"]
            if P.shape[0] == 0:
                continue
            preds = P.argmax(-1)
            gt = batch["rel_labels"][:preds.shape[0]]
            metric_acc += (preds == gt).float().sum().item()
            n += preds.shape[0]
    acc = metric_acc / max(n, 1)
    logger.info(f"validation rel-acc = {acc:.4f}  (over {n} pairs)")
    return {"rel_acc": acc}


def _move(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move(v, device) for v in batch)
    return batch


def _unpack(batch: Dict[str, Any], model, *, training: bool = True) -> Dict[str, Any]:
    """Adapt the dataset batch dict to the model's forward signature.

    Stage 1 expects ``latents``, ``clip_image_features``, ``factual_prompts``,
    ``counterfactual_prompts``. Stage 2 expects ``F_map`` (calibrated feature
    map), ``clip_text_obj``, ``clip_text_rel``, optional ``targets``.

    Concrete unpacking is done in the entry script ``main.py``; the engine
    only forwards what the dataloader returns.
    """
    return batch  # the dataloader is already a kwargs-shaped dict in main.py
