# 周三组会 Demo 脚本

> 目标：展示 Group A Baseline 可以成功运行，为后续 Group B 实验奠定基础。

---

## 准备工作（5 分钟）

```bash
# 1. 激活环境
conda activate pantheonrl_env

# 2. 进入项目目录
cd C:\Users\my185\Desktop\研究\durf\overcooked_ai\group_a_baseline

# 3. 运行测试确认一切正常
python -m pytest tests/ -v
```

预期输出：**16 passed**

---

## Demo 1: 单元测试（2 分钟）

展示代码质量：

```bash
python -m pytest tests/ -v --tb=short
```

**讲解要点**:
- `test_protocol.py` (8 tests): FeedbackEvent 数据协议 — 不可变、可哈希、支持 metadata
- `test_wrapper.py` (8 tests): HumanFeedbackEnvWrapper — reward 注入、alpha 缩放、信号裁剪
- `test_reproducibility.py` (4 tests): 随机种子可复现性

---

## Demo 2: 快速训练（5 分钟）

展示端到端训练 pipeline：

```bash
python scripts/test_trainer.py
```

**预期输出**:
```
============================================================
  run_id : no-feedback_P00_20260602_215636
  mode   : no-feedback
  steps  : 2,048
============================================================
Using cuda device
...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 400      |
|    ep_rew_mean     | 1.8      |
| time/              |          |
|    fps             | 163      |
|    total_timesteps | 2048     |
---------------------------------
============================================================
  训练完成！耗时 15.5s
============================================================
```

**讲解要点**:
- PPO 算法 + Overcooked 环境
- 三种 mode 通过替换子进程实现
- 训练产物：`final_model.zip`, `trajectories.jsonl`, `tb/`

---

## Demo 3: 评估模型（3 分钟）

展示训练好的模型效果：

```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.eval.evaluator import evaluate_model
metrics = evaluate_model('logs/no-feedback_P00_20260602_215636/final_model.zip', n_episodes=5)
"
```

**讲解要点**:
- 确定性策略评估
- 指标：mean_reward, success_rate, episode_length
- 可复现性：同 seed 同结果

---

## Demo 4: 代码结构总览（3 分钟）

展示项目结构：

```
group_a_baseline/
├── config/default.yaml          # 超参数配置
├── src/
│   ├── env/
│   │   ├── factory.py           # Env 构造工厂
│   │   └── human_feedback_wrapper.py  # 反馈注入层
│   ├── feedback/
│   │   ├── protocol.py          # FeedbackEvent 数据协议
│   │   ├── keyboard_listener.py # 真人键盘输入
│   │   ├── random_source.py     # 随机反馈（测试用）
│   │   └── null_source.py       # 无反馈
│   ├── train/
│   │   ├── trainer.py           # 训练主入口
│   │   ├── callbacks.py         # SB3 Callbacks
│   │   └── logging_schema.py    # 日志格式
│   ├── eval/
│   │   └── evaluator.py         # 模型评估
│   └── utils/
│       ├── repro.py             # 可复现性工具
│       └── ipc.py               # 进程通信辅助
├── tests/                       # 20 个测试
├── scripts/                     # 启动脚本
└── docs/                        # 文档
```

---

## 常见问题

### Q: 训练报错 "No module named 'src'"
**A**: 确保在 `group_a_baseline` 目录下运行，或使用 `scripts/train_*.bat`。

### Q: 训练太慢
**A**: 设置 `--total-timesteps 2048` 做快速测试。完整训练需要 1M steps（约 2 小时）。

### Q: 键盘监听不工作
**A**: 确保终端窗口在前台。`human-feedback` 模式需要显示器。

### Q: 如何查看 TensorBoard？
**A**: `tensorboard --logdir logs/` 然后打开 http://localhost:6006
