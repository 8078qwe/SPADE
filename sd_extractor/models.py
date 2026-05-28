"""SD UNet wrapper used by both Stage-1 calibration and Stage-2 feature extraction.

This file is a slimmed-down re-implementation of the wrapper used in CaDM-LQ
(``SD_Extractor/models.py``) but exposes two extra hooks SPADE++ needs:

1. A LoRA-adaptable cross-attention layer on the *student* path, so we can train
   only the adapters while keeping the pretrained SD weights frozen.
2. A multi-scale ``feature_dict`` that aggregates UNet features at scales
   1/8, 1/16 and 1/32, used by Stage 2's instance decoder.

The teacher path (frozen) is used for DDIM inversion to produce factual /
counterfactual cross-attention maps as described in the paper, Section 3.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# LoRA module (rank-r low-rank update to a linear projection)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A LoRA wrapper around an existing ``nn.Linear``.

    Eq. (LoRA) from Hu et al. 2022:  ``y = W x + (B A) x``
    where ``A ∈ R^{r×in}`` and ``B ∈ R^{out×r}`` are the only trainable params.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)  # zero init so initial output == frozen base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.base(x)
        out = out + (x @ self.A.t()) @ self.B.t() * self.scale
        return out


# ---------------------------------------------------------------------------
# Cross-attention with optional attention-controller hook (for inversion)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Cross-attention compatible with SD's UNet, with controller hook.

    The controller, when registered, receives the per-step softmax attention
    map so DDIM inversion can record factual / counterfactual maps. It does
    not modify the forward pass otherwise.
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64,
                 lora_rank: int = 0, lora_alpha: int = 32):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim_head)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        if lora_rank > 0:
            self.to_q = LoRALinear(self.to_q, rank=lora_rank, alpha=lora_alpha)
            self.to_k = LoRALinear(self.to_k, rank=lora_rank, alpha=lora_alpha)
            self.to_v = LoRALinear(self.to_v, rank=lora_rank, alpha=lora_alpha)
        self._controller_hook = None  # callable(attn, is_cross, place)
        self._place: str = "down"

    def register_controller(self, fn, place: str = "down") -> None:
        self._controller_hook = fn
        self._place = place

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.heads
        is_cross = context is not None
        ctx = context if is_cross else x
        q, k, v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            sim = sim.masked_fill(~mask[:, None, :].repeat(h, 1, 1), -float("inf"))

        attn = sim.softmax(dim=-1)
        if self._controller_hook is not None:
            head_avg = rearrange(attn, "(b h) i j -> b h i j", h=h).mean(1)
            self._controller_hook(head_avg, is_cross, self._place)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Attention-controller used for DDIM inversion
# ---------------------------------------------------------------------------
class AttentionStore:
    """Records cross-attention maps emitted by every CrossAttention layer.

    Storage is keyed by ``place`` ("down" / "mid" / "up") and timestep. The
    final-step map is the most spatially faithful (Wang et al. 2023) and is
    what we use as ``A*``.
    """

    def __init__(self) -> None:
        self.store: Dict[str, List[torch.Tensor]] = {"down": [], "mid": [], "up": []}
        self.cur_step = 0
        self.cur_layer = 0

    def reset(self) -> None:
        self.store = {"down": [], "mid": [], "up": []}
        self.cur_step = 0
        self.cur_layer = 0

    def __call__(self, attn: torch.Tensor, is_cross: bool, place: str) -> None:
        if not is_cross:
            return  # we only need cross-attention
        self.store[place].append(attn.detach())
        self.cur_layer += 1


def register_attention_controller(unet: nn.Module, controller: AttentionStore) -> int:
    """Walk the UNet, attach the controller to every CrossAttention layer."""
    count = 0

    def _attach(module: nn.Module, place: str) -> None:
        nonlocal count
        for child in module.children():
            if isinstance(child, CrossAttention):
                child.register_controller(controller, place=place)
                count += 1
            else:
                _attach(child, place)

    if hasattr(unet, "input_blocks"):
        _attach(unet.input_blocks, "down")
    if hasattr(unet, "middle_block"):
        _attach(unet.middle_block, "mid")
    if hasattr(unet, "output_blocks"):
        _attach(unet.output_blocks, "up")
    return count


# ---------------------------------------------------------------------------
# UNetWrapper exposed to the rest of the codebase
# ---------------------------------------------------------------------------
@dataclass
class UNetOutput:
    eps: torch.Tensor                      # predicted noise at the current step
    feats: Dict[str, torch.Tensor]         # multi-scale UNet features
    attn:  Optional[Dict[str, List[torch.Tensor]]] = None


class UNetWrapper(nn.Module):
    """Wraps a Stable-Diffusion UNet so callers don't depend on ldm internals.

    For Stage 1 we instantiate two copies:
      * ``teacher`` — fully frozen, used for DDIM inversion.
      * ``student`` — LoRA-adapted, the calibrated UNet we train.

    Both share the exact same architecture and pretrained weights;  only the
    LoRA adapters and the visual MLP adapter are different.
    """

    def __init__(self, unet: nn.Module, *, lora_rank: int = 0, freeze_base: bool = True):
        super().__init__()
        self.unet = unet
        self.lora_rank = lora_rank
        if freeze_base:
            for p in self.unet.parameters():
                p.requires_grad_(False)
        # Replace cross-attention modules with our hookable version.
        self._inject_lora_cross_attn()

    # -- LoRA injection ----------------------------------------------------
    def _inject_lora_cross_attn(self) -> None:
        if self.lora_rank <= 0:
            return
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ != "CrossAttention":
                continue
            # Replace the to_q / to_k / to_v projections with LoRA-adapted ones.
            for proj_name in ("to_q", "to_k", "to_v"):
                base = getattr(module, proj_name)
                if isinstance(base, nn.Linear):
                    setattr(module, proj_name, LoRALinear(base, rank=self.lora_rank))

    # -- Forward -----------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        controller: Optional[AttentionStore] = None,
    ) -> UNetOutput:
        if controller is not None:
            register_attention_controller(self.unet, controller)
        eps, feats = self._forward_collect(x_t, t, cond)
        attn = controller.store if controller is not None else None
        return UNetOutput(eps=eps, feats=feats, attn=attn)

    def _forward_collect(self, x_t, t, cond) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Hook into UNet's forward to grab multi-scale features.

        For brevity this reference implementation calls the base UNet's forward
        and returns its hidden states via standard hooks; downstream consumers
        only need ``feats['s8']``, ``feats['s16']``, ``feats['s32']``.
        """
        feats: Dict[str, torch.Tensor] = {}
        handles = []

        def _hook(name: str):
            def _fn(_m, _inp, out):
                if isinstance(out, (list, tuple)):
                    out = out[0]
                feats[name] = out
            return _fn

        # The exact module names depend on which SD UNet implementation is
        # plugged in (ldm, diffusers). The mapping below is the conventional
        # SDv1.4/v1.5 UNet from CompVis/ldm.
        if hasattr(self.unet, "output_blocks"):
            handles.append(self.unet.output_blocks[5].register_forward_hook(_hook("s8")))
            handles.append(self.unet.output_blocks[8].register_forward_hook(_hook("s16")))
            handles.append(self.unet.output_blocks[11].register_forward_hook(_hook("s32")))
        eps = self.unet(x_t, t, cond)
        for h in handles:
            h.remove()
        return eps, feats
