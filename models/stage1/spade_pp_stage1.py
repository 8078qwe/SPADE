"""Stage-1 model: ``SPADEppStage1``.

Glues together:
  * the frozen *teacher* SD UNet (used for DDIM inversion of factual /
    counterfactual prompts);
  * the *student* LoRA-adapted UNet conditioned on a CLIP image adapter;
  * the calibration loss of Eq. 6.

Following CaDM-LQ's class layout, the model exposes a single ``forward``
that returns a loss dict; ``inference`` returns the calibrated multi-scale
feature map ``F`` consumed by Stage 2.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sd_extractor import AttentionStore, UNetWrapper
from .calibration_loss import CalibrationConfig, VisualConditioner, calibration_loss
from .ddim_inversion import ddim_invert


class SPADEppStage1(nn.Module):
    def __init__(
        self,
        teacher_unet: nn.Module,
        student_unet: nn.Module,
        *,
        clip_image_encoder: nn.Module,
        clip_text_encoder: nn.Module,
        clip_tokenizer,
        alphas_cumprod: torch.Tensor,
        lora_rank: int = 16,
        cfg: CalibrationConfig = CalibrationConfig(),
        num_inv_steps: int = 50,
    ):
        super().__init__()
        # Teacher is frozen.
        self.teacher = UNetWrapper(teacher_unet, lora_rank=0, freeze_base=True)
        # Student is LoRA-adapted.
        self.student = UNetWrapper(student_unet, lora_rank=lora_rank, freeze_base=True)

        self.clip_img = clip_image_encoder
        for p in self.clip_img.parameters():
            p.requires_grad_(False)
        self.clip_txt = clip_text_encoder
        for p in self.clip_txt.parameters():
            p.requires_grad_(False)
        self.tokenizer = clip_tokenizer

        # Visual conditioning adapter — the only part of the student input
        # path that is trainable (besides the LoRA matrices inside the UNet).
        self.cond_mlp = VisualConditioner(
            in_dim=getattr(clip_image_encoder, "output_dim", 1024),
            out_dim=768,
            num_tokens=77,
        )

        self.cfg = cfg
        self.num_inv_steps = num_inv_steps
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    # ------------------------------------------------------------------ utils
    def _encode_text(self, prompts: List[str], device) -> torch.Tensor:
        tok = self.tokenizer(prompts, padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.clip_txt(**tok).last_hidden_state
        return out  # (B, 77, D)

    # ------------------------------------------------------------------- main
    def forward(
        self,
        latents: torch.Tensor,                # (B, 4, h, w) — SD VAE latents of x_0
        clip_image_features: torch.Tensor,    # (B, D_img)
        factual_prompts: List[str],
        counterfactual_prompts: List[List[str]],   # outer list: per-image; inner list: K cf prompts
    ) -> Dict[str, torch.Tensor]:
        device = latents.device
        B = latents.shape[0]

        # ---- 1. Teacher: factual A* -----------------------------------------
        cond_fact = self._encode_text(factual_prompts, device)
        _, A_star = ddim_invert(
            self.teacher, latents, cond_fact,
            num_steps=self.num_inv_steps, alphas_cumprod=self.alphas_cumprod,
        )

        # ---- 2. Teacher: counterfactual maps --------------------------------
        cf_attns: List[torch.Tensor] = []
        for k in range(max(len(c) for c in counterfactual_prompts) if counterfactual_prompts else 0):
            cf_batch = []
            for i, lst in enumerate(counterfactual_prompts):
                cf_batch.append(lst[k] if k < len(lst) else factual_prompts[i])
            cond_cf = self._encode_text(cf_batch, device)
            _, A_cf = ddim_invert(
                self.teacher, latents, cond_cf,
                num_steps=self.num_inv_steps, alphas_cumprod=self.alphas_cumprod,
            )
            cf_attns.append(A_cf)

        # ---- 3. Student: visual-conditioned forward ------------------------
        cond_vis = self.cond_mlp(clip_image_features)
        controller = AttentionStore()
        # We only need the final-step attention, so we run a single forward
        # at t = T-1 (matching Eq. 1 — student is run for inference, not full
        # inversion).
        t = torch.full((B,), self.alphas_cumprod.numel() - 1, device=device, dtype=torch.long)
        out = self.student(latents, t, cond=cond_vis, controller=controller)
        # Aggregate student attention the same way ddim_invert does.
        maps = controller.store["down"] + controller.store["up"]
        if not maps:
            raise RuntimeError("Student UNet emitted no cross-attention; check controller registration.")
        max_n = max(m.shape[-2] for m in maps)
        resized = []
        for m in maps:
            if m.shape[-2] != max_n:
                b, n, l = m.shape
                s = int(round(n ** 0.5))
                m4d = m.transpose(1, 2).reshape(b, l, s, s)
                t_sz = int(round(max_n ** 0.5))
                m4d = torch.nn.functional.interpolate(m4d, size=(t_sz, t_sz),
                                                     mode="bilinear", align_corners=False)
                m = m4d.reshape(b, l, t_sz * t_sz).transpose(1, 2)
            resized.append(m)
        A_prime = torch.stack(resized, dim=0).mean(0)

        # ---- 4. Calibration loss ------------------------------------------
        losses = calibration_loss(A_prime, A_star, cf_attns=cf_attns, cfg=self.cfg)
        return losses

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def extract_features(
        self,
        latents: torch.Tensor,
        clip_image_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return the calibrated multi-scale feature map ``F`` for Stage 2."""
        device = latents.device
        B = latents.shape[0]
        cond = self.cond_mlp(clip_image_features)
        t = torch.full((B,), self.alphas_cumprod.numel() - 1, device=device, dtype=torch.long)
        out = self.student(latents, t, cond=cond)
        return out.feats
