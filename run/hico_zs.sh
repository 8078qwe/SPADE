#!/usr/bin/env bash
# SPADE++ — HICO-DET zero-shot (NF / UC scheme).
set -euo pipefail
python main.py \
  --cfg configs/hico_zs.yaml \
  --stage 2 \
  --device cuda
