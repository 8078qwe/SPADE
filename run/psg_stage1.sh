#!/usr/bin/env bash
# SPADE++ — Stage 1 calibration on PSG.
set -euo pipefail
python main.py \
  --cfg configs/psg_stage1.yaml \
  --stage 1 \
  --device cuda
