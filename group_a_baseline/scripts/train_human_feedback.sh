#!/bin/bash
# 真人键盘交互训练（需要显示器）
# 用法: conda activate pantheonrl_env && bash scripts/train_human_feedback.sh P01
# 按键: Q=+1(Good) / E=-1(Bad) / C=Coach-Pause / ESC=退出
cd "$(dirname "$0")/.."
python -m src.train.trainer \
    --mode human-feedback \
    --total-timesteps 100000 \
    --participant "${1:-P00}" \
    --seed 42 \
    --config config/default.yaml
