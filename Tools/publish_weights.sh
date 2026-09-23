#!/bin/bash
# Publish the converted tiers to mlx-community. RUN ONLY AFTER OPERATOR CONFIRMATION (publishing
# is outward-facing). Each tier directory already carries LICENSE (the NetEase agreement),
# MODEL_LICENSE_zh, NOTICE and a README with the §4.1(a) derivative statement.
set -euo pipefail
W=/Volumes/Satechi/Development/mlxengine-audio/WIP/r2t2-eval/_weights
for tier in "$@"; do
  D=$W/Confucius4-R2T2-$tier
  test -f $D/LICENSE && test -f $D/NOTICE && test -f $D/README.md
  hf repo create mlx-community/Confucius4-R2T2-$tier --type model 2>/dev/null || true
  hf upload mlx-community/Confucius4-R2T2-$tier $D . --commit-message "Confucius4-R2T2 MLX $tier (qwen3-asr-mlx-swift V1)"
done
