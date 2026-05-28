"""Convert pretrained DETR / Mask2Former parameters for SPADE++ Stage 2.

Mirrors CaDM-LQ/tools/convert_parameters.py: optionally drops query embeddings
and resizes them to ``num_queries`` so the Stage 2 InstanceDecoder can be
warm-started from a publicly released DETR / Mask2Former checkpoint.
"""
from __future__ import annotations

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--num_queries", type=int, default=100)
    args = ap.parse_args()

    state = torch.load(args.load_path, map_location="cpu")
    sd = state.get("model", state)

    # Resize query embeddings if present.
    for key in list(sd.keys()):
        if "query_embed" in key:
            w = sd[key]
            if w.shape[0] != args.num_queries:
                print(f"resizing {key} from {w.shape} to {args.num_queries}×{w.shape[1]}")
                new_w = torch.zeros(args.num_queries, w.shape[1])
                new_w[: min(args.num_queries, w.shape[0])] = w[: min(args.num_queries, w.shape[0])]
                sd[key] = new_w
        if "class_embed.weight" in key or "class_embed.bias" in key:
            sd.pop(key)
    torch.save({"model": sd}, args.save_path)
    print(f"Saved {args.save_path}")


if __name__ == "__main__":
    main()
