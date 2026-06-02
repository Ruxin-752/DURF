# 架构概览

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        trainer.py (主进程)                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Config       │───>│  PPO (SB3)       │───>│  Callbacks    │  │
│  │  (YAML)       │    │  model.learn()   │    │  - Trajectory │  │
│  └──────────────┘    └────────┬─────────┘    │  - Checkpoint │  │
│                               │               │  - ConfigSnap │  │
│                               ▼               └───────────────┘  │
│  ┌─────────────────────────────────────────────┐                 │
│  │  Env Stack (由内到外)                        │                 │
│  │                                             │                 │
│  │  ┌─────────────────────────────────────┐   │                 │
│  │  │ Monitor (SB3, 最外层)               │   │                 │
│  │  ├─────────────────────────────────────┤   │                 │
│  │  │ GymV21CompatibilityV0 (shimmy)      │   │                 │
│  │  ├─────────────────────────────────────┤   │                 │
│  │  │ HumanFeedbackEnvWrapper (可选)       │   │                 │
│  │  ├─────────────────────────────────────┤   │                 │
│  │  │ OvercookedMultiEnv (PantheonRL)     │   │                 │
│  │  └─────────────────────────────────────┘   │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Feedback Source (子进程)                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │    │
│  │  │ null_source  │  │random_source │  │keyboard_list. │  │    │
│  │  │ (无反馈)     │  │(随机反馈)    │  │(真人按键)     │  │    │
│  │  └──────────────┘  └──────────────┘  └───────────────┘  │    │
│  │         │                 │                  │           │    │
│  │         └─────────────────┴──────────────────┘           │    │
│  │                          │ multiprocessing.Queue         │    │
│  │                          ▼                               │    │
│  │                   feedback_queue                         │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 数据流

### 训练模式 (no-feedback)
```
PPO → env.step(action) → OvercookedMultiEnv → reward (环境原生)
     → Monitor 记录 → PPO 学习
```

### 训练模式 (human-feedback)
```
PPO → env.step(action) → OvercookedMultiEnv → env_reward
     → HumanFeedbackEnvWrapper 从 queue 读取 human_reward
     → total_reward = env_reward + alpha * human_reward
     → Monitor 记录 → PPO 学习

键盘子进程:
    keyboard_listener → 监听 Q(+1) / E(-1) / C(暂停)
                      → 写入 feedback_queue
```

### 评估模式
```
evaluator.py → make_overcooked_env(no-feedback)
             → PPO.load(model_path)
             → 循环: model.predict(obs) → env.step(action)
             → 收集 metrics
```

## 文件依赖关系

```
trainer.py
  ├── config/default.yaml
  ├── src/env/factory.py
  │     ├── src/env/human_feedback_wrapper.py
  │     └── PantheonRL (overcookedgym)
  ├── src/feedback/null_source.py
  ├── src/feedback/random_source.py
  ├── src/feedback/keyboard_listener.py
  ├── src/feedback/protocol.py
  └── src/train/callbacks.py
        └── src/train/logging_schema.py

evaluator.py
  ├── src/env/factory.py
  └── PPO.load()

tests/
  ├── test_protocol.py → src/feedback/protocol.py
  ├── test_wrapper.py  → src/env/human_feedback_wrapper.py
  └── test_reproducibility.py → src/utils/repro.py
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Feedback 通信 | multiprocessing.Queue | 解耦训练循环和人类交互，避免阻塞 |
| Wrapper 顺序 | Monitor 在最外层 | SB3 issue #146: Monitor 必须在最外层 |
| 反馈注入 | 加法: total = env + alpha * human | 简单可调，alpha 控制人类影响权重 |
| 信号裁剪 | [-1, 1] | 防止异常值破坏训练稳定性 |
| 子进程 | daemon=True | 主进程退出时自动清理 |
