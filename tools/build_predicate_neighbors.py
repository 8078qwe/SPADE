"""Pre-compute CLIP top-K text-embedding neighbors for every predicate.

Produces ``predicate_neighbors.json`` of the form::

    {
      "on":   ["standing on", "sitting on", "lying on", ...],
      "over": ["above", "in front of", ...],
      ...
    }

Used by Stage 1 (counterfactual prompts) and Stage 2 (decision-level contrast).
"""
from __future__ import annotations

import argparse
import json
from typing import List

import torch

import clip


def main(out_path: str, predicate_file: str, top_k: int, model_name: str = "ViT-L/14"):
    with open(predicate_file, "r", encoding="utf-8") as f:
        predicates: List[str] = [ln.strip() for ln in f if ln.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    with torch.no_grad():
        toks = clip.tokenize(predicates).to(device)
        embs = model.encode_text(toks)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    sim = embs @ embs.t()
    sim.fill_diagonal_(-1)
    topk_idx = sim.topk(top_k, dim=-1).indices.cpu().tolist()
    out = {p: [predicates[j] for j in row] for p, row in zip(predicates, topk_idx)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicate_file", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--clip_model", default="ViT-L/14")
    args = ap.parse_args()
    main(args.out_path, args.predicate_file, args.top_k, args.clip_model)
