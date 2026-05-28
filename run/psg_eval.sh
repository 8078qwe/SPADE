#!/usr/bin/env bash
# SPADE++ — PSG evaluation (Stage 2).
set -euo pipefail
python main.py \
  --cfg configs/psg_stage2.yaml \
  --stage 2 \
  --eval-only true \
  --device cuda
