"""DDIM deterministic inversion (Song et al. 2020) used as a *spatial probe*.

Given a real image, DDIM with stochasticity ``η = 0`` is bijective and
produces a deterministic trajectory ``x_0 → x_T``. When run with a relation
prompt, the cross-attention maps along the trajectory localise each prompt
token to image regions. As shown in `Wang et al., 2023`, the **final-step**
map (``t = T``) is the most spatially faithful, and is what we use as the
factual / counterfactual teacher signal in Stage 1.

This module exposes a single function :func:`ddim_invert` that returns

    * ``x_T``  — the inverted latent.
    * ``A``    — the final-step cross-attention map of shape ``(HW, L)``.

The inversion is performed under ``torch.no_grad()`` because the teacher is
frozen.
"""
from __future__ import annotations

from typing import Tuple

import torch

from sd_extractor import AttentionStore, UNetWrapper


@torch.no_grad()
def ddim_invert(
    unet: UNetWrapper,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    *,
    num_steps: int = 50,
    alphas_cumprod: torch.Tensor,
    return_attention: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic DDIM inversion.

    Args:
        unet:               frozen UNet wrapper (teacher).
        x_0:                clean image latent of shape (B, C, H, W).
        cond:               prompt-conditioning tensor (B, L, D).
        num_steps:          number of inversion steps T.
        alphas_cumprod:     pre-computed cumulative alphas tensor of shape (T_train,).
        return_attention:   if True, also returns the final-step attention map.

    Returns:
        x_T:  inverted noisy latent.
        A:    final-step attention map (B, HW, L) if ``return_attention``,
              otherwise ``None``.
    """
    device = x_0.device
    B = x_0.shape[0]
    timesteps = torch.linspace(0, len(alphas_cumprod) - 1, num_steps, dtype=torch.long, device=device)

    controller = AttentionStore() if return_attention else None
    x_t = x_0.clone()

    # DDIM forward (a.k.a. inversion) update:
    #   x_{t+1} = sqrt(α_{t+1}) · x_pred_0 + sqrt(1 − α_{t+1}) · ε_θ(x_t, t)
    # with x_pred_0 = (x_t − sqrt(1 − α_t) ε) / sqrt(α_t).
    for i in range(num_steps - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        a_cur = alphas_cumprod[t_cur]
        a_next = alphas_cumprod[t_next]

        out = unet(x_t, t_cur.expand(B), cond=cond, controller=controller if i == num_steps - 2 else None)
        eps = out.eps
        x_pred_0 = (x_t - (1 - a_cur).sqrt() * eps) / a_cur.sqrt()
        x_t = a_next.sqrt() * x_pred_0 + (1 - a_next).sqrt() * eps

    if return_attention:
        # Aggregate maps from the last call: average all "down"+"up" cross-attention layers
        # at the highest spatial resolution available.
        maps = controller.store["down"] + controller.store["up"]
        if not maps:
            return x_t, None
        # Pick the highest-resolution map (largest token count) and resize others.
        max_n = max(m.shape[-2] for m in maps)
        resized = []
        for m in maps:
            if m.shape[-2] != max_n:
                # Bilinear resize on the spatial axis interpreted as sqrt-sized 2-D grid.
                b, n, l = m.shape
                s = int(round(n ** 0.5))
                m4d = m.transpose(1, 2).reshape(b, l, s, s)
                t = int(round(max_n ** 0.5))
                m4d = torch.nn.functional.interpolate(m4d, size=(t, t), mode="bilinear", align_corners=False)
                m = m4d.reshape(b, l, t * t).transpose(1, 2)
            resized.append(m)
        A = torch.stack(resized, dim=0).mean(0)  # (B, HW, L)
        return x_t, A

    return x_t, None
