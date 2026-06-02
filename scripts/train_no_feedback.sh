#!/bin/bash
# 纯环境 reward 基线（no-feedback 对照组）
# 用法: conda activate pantheonrl_env && bash scripts/train_no_feedback.sh
cd "$(dirname "$0")/.."
python -m src.train.trainer \
    --mode no-feedback \
    --total-timesteps 1000000 \
    --participant "${1:-P00}" \
    --seed 42 \
    --config config/default.yaml
