# Group A Baseline

PPO + 人类反馈的 Overcooked-AI 训练 pipeline。人类通过键盘实时给 RL agent 注入奖励信号，验证"丰富反馈 > 标量反馈"的核心假设。

## Quick Start（5 分钟）

```bash
# 1. 激活环境
conda activate pantheonrl_env

# 2. 安装依赖（overcooked_ai 和 PantheonRL 已 editable 安装）
pip install -r requirements.txt

# 3. Smoke test（5000 steps，~30 秒）
python scripts/smoke_test.py
```

## Three Run Modes

| 模式 | 命令 | 用途 |
|------|------|------|
| `no-feedback` | `./scripts/train_no_feedback.sh` | 纯环境 reward 基线 |
| `random-feedback` | `./scripts/train_random_feedback.sh` | 自动注入，验证 pipeline 正确性 |
| `human-feedback` | `./scripts/train_human_feedback.sh` | 真人键盘交互（Q=+1 / E=-1 / C=暂停） |

## Architecture

```
keyboard_listener (subprocess)
        │ multiprocessing.Queue
        ▼
HumanFeedbackEnvWrapper   ← 核心：reward 注入层
        │
Monitor (SB3 内置)
        │
PPO.learn() + TrajectoryLoggerCallback + CheckpointCallback
```

## Project Structure

```
group_a_baseline/
├── config/default.yaml       # 所有超参，代码不写魔法数字
├── src/
│   ├── feedback/             # FeedbackEvent 协议 + 各 source
│   ├── env/                  # HumanFeedbackEnvWrapper + factory
│   ├── train/                # trainer + callbacks + logging schema
│   ├── eval/                 # checkpoint 评估器
│   └── utils/                # seed 设置 + IPC 辅助
├── tests/                    # 单元测试
├── scripts/                  # 三种模式启动脚本 + smoke test
└── docs/                     # 架构图、设计决策、已知问题、demo 脚本
```

## Known Limitations

见 [docs/known_issues.md](docs/known_issues.md)。

## References

- Carroll et al. (2019). *Overcooked-AI*
- Warnell et al. (2018). *Deep TAMER*
- Hadfield-Menell et al. (2016). *CIRL*
