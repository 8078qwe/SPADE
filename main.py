"""SPADE++ command-line entry point.

This script mirrors the role of CaDM-LQ/main.py. It supports both Stage 1
(diffusion calibration) and Stage 2 (geometry-conditioned reasoning), driven
by a YAML config under ``configs/``.

Usage:

    # Stage 1
    python main.py --cfg configs/psg_stage1.yaml --stage 1

    # Stage 2
    python main.py --cfg configs/psg_stage2.yaml --stage 2 --eval-only false
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import HICODataset, PSGDataset, VCOCODataset, VGDataset
from engine import evaluate, train_one_epoch
from models import SPADEppStage1, SPADEppStage2, Stage2Config
from models.stage1 import (
    PredicateNeighbors,
    Triple,
    build_counterfactual_prompts,
    build_factual_prompt,
)
from util import collate_fn, set_seed, setup_logger


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("SPADE++ trainer/evaluator")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--eval-only", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
def build_dataset(cfg: Dict[str, Any]):
    name = cfg["dataset"]["name"]
    if name == "psg":
        return PSGDataset(
            cfg["dataset"]["ann_file"], cfg["dataset"]["image_root"],
            predicate_vocab=cfg["dataset"].get("predicate_vocab", []),
            object_vocab=cfg["dataset"].get("object_vocab", []),
            image_size=cfg["dataset"]["image_size"],
        )
    if name == "vg":
        return VGDataset(
            cfg["dataset"]["ann_file"], cfg["dataset"]["image_root"],
            predicate_vocab=cfg["dataset"].get("predicate_vocab", []),
            object_vocab=cfg["dataset"].get("object_vocab", []),
            image_size=cfg["dataset"]["image_size"],
        )
    if name == "hico":
        return HICODataset(
            cfg["dataset"]["ann_file"], cfg["dataset"]["image_root"],
            num_queries=cfg["model"].get("num_queries", 64),
            image_size=cfg["dataset"]["image_size"],
        )
    if name == "vcoco":
        return VCOCODataset(
            cfg["dataset"]["ann_file"], cfg["dataset"]["image_root"],
            num_queries=cfg["model"].get("num_queries", 64),
            image_size=cfg["dataset"]["image_size"],
        )
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
def build_stage1(cfg: Dict[str, Any], device):
    """Build the Stage 1 calibration model.

    NB: the actual Stable-Diffusion checkpoint is not bundled here; we expect
    the user to supply ``cfg.model.sd_config`` and ``cfg.model.sd_checkpoint``
    pointing at the LDM v1.4/1.5 release.
    """
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config       # type: ignore

    sd_cfg = OmegaConf.load(cfg["model"]["sd_config"])
    sd = instantiate_from_config(sd_cfg.model)
    state = torch.load(cfg["model"]["sd_checkpoint"], map_location="cpu")
    sd.load_state_dict(state["state_dict"], strict=False)

    teacher_unet = sd.model.diffusion_model
    student_unet = type(teacher_unet)()
    student_unet.load_state_dict(teacher_unet.state_dict(), strict=False)

    import clip
    clip_model, _ = clip.load(cfg["model"]["clip_model"], device=device)
    clip_image_encoder = clip_model.visual
    clip_image_encoder.output_dim = clip_model.visual.output_dim

    from transformers import CLIPTokenizer, CLIPTextModel
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    alphas_cumprod = sd.alphas_cumprod
    model = SPADEppStage1(
        teacher_unet=teacher_unet,
        student_unet=student_unet,
        clip_image_encoder=clip_image_encoder,
        clip_text_encoder=clip_text_encoder,
        clip_tokenizer=clip_tokenizer,
        alphas_cumprod=alphas_cumprod,
        lora_rank=cfg["model"]["lora_rank"],
        num_inv_steps=cfg["model"]["num_inv_steps"],
    )
    return model.to(device)


def build_stage2(cfg: Dict[str, Any]) -> SPADEppStage2:
    s2cfg = Stage2Config(
        feat_dim=cfg["model"].get("feat_dim", 320),
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        num_queries=cfg["model"].get("num_queries", 100),
        rgt_layers=cfg["model"].get("rgt_layers", 8),
        geometric_dim=cfg["model"].get("geometric_dim", 128),
        semantic_dim=cfg["model"].get("semantic_dim", 768),
        alpha=cfg["model"].get("alpha", 0.34),
        gamma=cfg["model"].get("gamma", 0.2),
        top_k_inf=cfg["model"].get("top_k_inf", 3),
        lambda_rqc=cfg["losses"].get("lambda_rqc", 1.0),
        lambda_mask=cfg["losses"].get("lambda_mask", 5.0),
        lambda_cfg=cfg["losses"].get("lambda_cfg", 1.0),
    )
    return SPADEppStage2(s2cfg)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("spade++", log_file=str(out_dir / "train.log"))
    logger.info(f"Config: {args.cfg}")

    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- Data ----
    ds = build_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=not args.eval_only,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # ---- Model ----
    if args.stage == 1:
        model = build_stage1(cfg, device)
    else:
        model = build_stage2(cfg).to(device)
        if cfg.get("stage1_ckpt") and os.path.isfile(cfg["stage1_ckpt"]):
            logger.info(f"Loading frozen Stage-1 backbone from {cfg['stage1_ckpt']}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    if args.eval_only:
        evaluate(model, loader, device=device, logger=logger)
        return

    for epoch in range(cfg["optim"]["epochs"]):
        train_one_epoch(
            model, loader, optim,
            device=device, logger=logger,
            epoch=epoch,
            log_interval=cfg["logging"]["log_interval"],
            grad_accum=cfg["optim"].get("grad_accum", 1),
            scaler=scaler,
        )
        if (epoch + 1) % cfg["logging"]["ckpt_interval"] == 0:
            ck = out_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, ck)
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg},
                       out_dir / "last.pt")
            logger.info(f"Saved checkpoint: {ck}")


if __name__ == "__main__":
    main()
