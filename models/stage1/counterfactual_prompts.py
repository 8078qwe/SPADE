"""Counterfactual prompt construction.

Implements §3.1 of the IJCV paper:

    "predicate replacement substitutes p_i with a CLIP-similar hard negative
     p_cf, yielding the prompt 's_i is p_cf o_i'.  Role swap exchanges
     subject and object, yielding 'o_i is p_i s_i'."

To sample p_cf we pre-compute the CLIP text embedding of every predicate in
the dataset's vocabulary and look up the top-K nearest neighbors. K = 5 by
default, matching Stage 2.

For efficiency, counterfactual prompts are only built for the
``n_cf = 3`` subject-object pairs with the **largest inter-centroid distance**
in the image (Eq. (3) in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLIP-neighbor lookup for predicates
# ---------------------------------------------------------------------------
class PredicateNeighbors:
    """Pre-computes top-K CLIP text-embedding neighbors of every predicate.

    Usage::

        neighbors = PredicateNeighbors(predicate_vocab, clip_text_encoder)
        cf = neighbors.sample_hard_negative('on', exclude={'on'})
    """

    def __init__(self, vocab: Sequence[str], embed_fn, *, top_k: int = 5):
        self.vocab = list(vocab)
        self.idx = {p: i for i, p in enumerate(self.vocab)}
        self.top_k = top_k
        with torch.no_grad():
            embs = embed_fn(self.vocab)
            embs = F.normalize(embs, dim=-1)
        self.embs = embs                                  # (V, D)
        sim = embs @ embs.t()                             # (V, V)
        sim.fill_diagonal_(-1.0)
        self.topk_idx = sim.topk(top_k, dim=-1).indices   # (V, K)

    def neighbors(self, predicate: str) -> List[str]:
        i = self.idx[predicate]
        return [self.vocab[j] for j in self.topk_idx[i].tolist()]

    def sample_hard_negative(self, predicate: str, exclude: Sequence[str] = ()) -> str:
        for cand in self.neighbors(predicate):
            if cand not in exclude and cand != predicate:
                return cand
        # Fall back to the closest one.
        return self.neighbors(predicate)[0]


# ---------------------------------------------------------------------------
# Counterfactual prompt construction
# ---------------------------------------------------------------------------
@dataclass
class Triple:
    s_label: str        # subject category text
    s_box:   torch.Tensor   # (4,) xyxy
    p_label: str        # predicate text
    o_label: str
    o_box:   torch.Tensor


def centroid(box: torch.Tensor) -> torch.Tensor:
    return torch.stack([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def select_distant_pairs(triples: List[Triple], n_cf: int = 3) -> List[int]:
    """Top-n_cf pairs by inter-centroid Euclidean distance, Eq. (3) in paper."""
    dists = [torch.norm(centroid(t.s_box) - centroid(t.o_box), p=2).item() for t in triples]
    order = sorted(range(len(triples)), key=lambda i: -dists[i])
    return order[:n_cf]


def build_factual_prompt(triples: List[Triple]) -> str:
    parts = [f"{t.s_label} is {t.p_label} {t.o_label}" for t in triples]
    return ", ".join(parts)


def build_counterfactual_prompts(
    triples: List[Triple],
    *,
    neighbors: PredicateNeighbors,
    n_cf: int = 3,
    role_swap: bool = True,
) -> List[Tuple[str, str]]:
    """Return list of (prompt_text, cf_type) pairs.

    cf_type ∈ {"replace", "swap"} so the loss can weight them differently.
    """
    selected = select_distant_pairs(triples, n_cf=n_cf)
    out: List[Tuple[str, str]] = []
    base = list(triples)
    for idx in selected:
        t = base[idx]
        # 1. predicate replacement
        p_cf = neighbors.sample_hard_negative(t.p_label)
        replaced = base[:]
        replaced[idx] = Triple(t.s_label, t.s_box, p_cf, t.o_label, t.o_box)
        out.append((build_factual_prompt(replaced), "replace"))
        # 2. role swap
        if role_swap:
            swapped = base[:]
            swapped[idx] = Triple(t.o_label, t.o_box, t.p_label, t.s_label, t.s_box)
            out.append((build_factual_prompt(swapped), "swap"))
    return out
