#!/usr/bin/env bash
# SPADE++ — Stage 2 reasoning on PSG.
set -euo pipefail
python main.py \
  --cfg configs/psg_stage2.yaml \
  --stage 2 \
  --device cuda
