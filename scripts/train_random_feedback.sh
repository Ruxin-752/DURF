#!/bin/bash
# 自动随机注入反馈（验证 pipeline，不需要真人）
# 用法: conda activate pantheonrl_env && bash scripts/train_random_feedback.sh
cd "$(dirname "$0")/.."
python -m src.train.trainer \
    --mode random-feedback \
    --total-timesteps 1000000 \
    --participant "${1:-P00}" \
    --seed 42 \
    --config config/default.yaml
