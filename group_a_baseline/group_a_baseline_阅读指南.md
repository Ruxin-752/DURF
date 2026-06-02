# Group A Baseline 阅读指南

> **项目定位**: 这是 DURF 项目中的**对照组 (Control Group)** 代码库，实现的是 **Deep TAMER** 框架——人类通过键盘实时给 RL agent 注入标量奖励信号 (+1/-1)，作为后续 Group B (LLM 自然语言反馈) 的对比基线。

---

## 一、项目全景图

```
group_a_baseline/
├── README.md              ← 项目总说明（快速启动、三种运行模式）
├── .gitignore             ← 忽略 logs/ checkpoints/ __pycache__/
├── requirements.txt       ← 依赖清单
├── config/
│   └── default.yaml       ← ★ 所有超参数集中配置（无魔法数字）
├── src/                   ← ★ 核心源代码
│   ├── __init__.py
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── protocol.py    ← ★ 反馈事件协议（唯一有实质代码的文件）
│   ├── eval/              ← (空) 预留评估模块
│   │   └── __init__.py
│   ├── train/             ← (空) 预留训练模块
│   │   └── __init__.py
│   └── utils/             ← (空) 预留工具模块
│       └── __init__.py
├── tests/                 ← (空) 预留测试目录
│   ├── __init__.py
│   └── test_protocol.py   ← (空) 预留协议测试
├── scripts/               ← (空) 预留启动脚本目录
├── docs/                  ← (空) 预留文档目录
├── checkpoints/           ← (空) 模型保存目录（被 .gitignore 忽略）
└── logs/                  ← (空) 日志目录（被 .gitignore 忽略）
```

---

## 二、文件逐项详解

### 1. `README.md` — 项目总说明书

**作用**: 项目的"大门"文件，告诉你这个项目是干什么的、怎么跑起来。

**关键内容**:
- **项目定位**: PPO + 人类反馈的 Overcooked-AI 训练 pipeline
- **三种运行模式**:
  | 模式 | 命令 | 用途 |
  |------|------|------|
  | `no-feedback` | `./scripts/train_no_feedback.sh` | 纯环境 reward 基线（无人类干预） |
  | `random-feedback` | `./scripts/train_random_feedback.sh` | 自动注入随机反馈，验证 pipeline 正确性 |
  | `human-feedback` | `./scripts/train_human_feedback.sh` | 真人键盘交互（Q=+1 / E=-1 / C=暂停） |
- **架构图**: keyboard_listener → multiprocessing.Queue → HumanFeedbackEnvWrapper → PPO
- **Quick Start**: `conda activate pantheonrl_env` → `pip install -r requirements.txt` → `python scripts/smoke_test.py`

> ⚠️ **注意**: README 中提到的 `scripts/` 目录下的 `.sh` 脚本和 `smoke_test.py` 目前尚未创建（目录为空），说明这个项目还处于**框架搭建阶段**，核心代码尚未完成。

---

### 2. `requirements.txt` — 依赖清单

**作用**: 列出运行项目需要安装的 Python 包。

**关键依赖**:
| 包名 | 版本 | 用途 |
|------|------|------|
| `gym==0.25.2` | 0.25.2 | OpenAI Gym 环境接口 |
| `stable-baselines3>=1.7.0` | ≥1.7 | SB3 的 PPO 算法实现 |
| `shimmy>=0.2.0` | ≥0.2 | Gym→PettingZoo 兼容层 |
| `numpy<2.0` | <2.0 | 数值计算 |
| `pygame>=2.1.0` | ≥2.1 | 键盘监听窗口（人类反馈 UI） |
| `pyyaml>=6.0` | ≥6.0 | 读取 YAML 配置文件 |
| `pytest>=7.0` | ≥7.0 | 单元测试 |

> ⚠️ **注意**: 依赖中**不包含** `overcooked_ai` 和 `PantheonRL`，因为它们需要以 editable 模式手动安装：
> ```bash
> pip install -e /path/to/overcooked_ai
> pip install -e /path/to/PantheonRL
> ```

---

### 3. `config/default.yaml` — 超参数配置文件

**作用**: 集中管理所有可调参数，代码中不写魔法数字。这是项目的"控制面板"。

**参数详解**:

```yaml
env:
  layout_name: cramped_room    # Overcooked 地图名称（拥挤房间）
  horizon: 400                 # 每个 episode 的最大步数

feedback:
  alpha: 1.0                   # 人类反馈奖励的缩放系数
  max_signal_per_step: 5.0     # 每步最大反馈值（防止极端反馈）
  random_source_rate_hz: 2.0   # random 模式下每秒注入反馈的次数

ppo:
  policy: MlpPolicy            # 策略网络类型（多层感知机）
  learning_rate: 3.0e-4        # PPO 学习率
  n_steps: 2048                # 每次更新前收集的步数
  batch_size: 64               # 训练批次大小
  n_epochs: 10                 # 每次更新时训练的 epoch 数
  gamma: 0.99                  # 折扣因子
  gae_lambda: 0.95             # GAE 参数
  clip_range: 0.2              # PPO clip 范围
  vf_coef: 0.5                 # 价值函数损失系数
  ent_coef: 0.01               # 熵正则化系数（鼓励探索）

train:
  total_timesteps: 100000      # 总训练步数（smoke test 默认值）
  checkpoint_freq: 10000       # 每多少步保存一次模型
  log_interval: 10             # 每多少 episode 打印一次日志

seed: 42                       # 随机种子
device: auto                   # 设备选择（auto/cuda/cpu）

logging:
  log_dir: logs                # 日志目录
  tensorboard: true            # 是否启用 TensorBoard
  trajectory_jsonl: true       # 是否保存轨迹 JSONL
```

---

### 4. `src/feedback/protocol.py` — ★ 核心代码文件

**作用**: 定义反馈事件的数据协议（Data Contract）。这是整个项目中**唯一有实质代码**的文件。

**代码详解**:

```python
from dataclasses import dataclass, field
from typing import Literal

# 事件类型：positive（正向）/ negative（负向）/ coach_pause（暂停教学）
EventType = Literal["positive", "negative", "coach_pause"]

@dataclass(frozen=True)  # frozen=True 表示不可变，保证跨进程安全
class FeedbackEvent:
    timestamp_ms: int        # 自 session 开始的单调递增毫秒时间戳
    event_type: EventType    # 事件类型
    signal_value: float      # 信号值：+1.0 / -1.0 / 0.0
    source: str              # 来源：'keyboard' / 'random' / 'null' / 'llm'
    metadata: dict | None = field(default=None, hash=False)  # 附加元数据
```

**设计意图**:
- 所有反馈来源（键盘 / 随机 / 空 / 未来 LLM）都产生同一种 `FeedbackEvent`
- `frozen=True` 保证跨进程传递时不会被篡改
- `metadata` 用 dict 留给未来 LLM source 附加 raw text，不改 protocol 本体
- 这是典型的**策略模式 (Strategy Pattern)** 设计——不同反馈源实现同一个接口

---

### 5. 空目录/文件说明

| 路径 | 状态 | 说明 |
|------|------|------|
| `src/eval/` | 空 | 预留评估模块，未来放 checkpoint 评估器 |
| `src/train/` | 空 | 预留训练模块，未来放 trainer + callbacks |
| `src/utils/` | 空 | 预留工具模块，未来放 seed 设置、IPC 辅助 |
| `tests/` | 空 | 预留测试目录 |
| `scripts/` | 空 | 预留启动脚本（README 提到的 .sh 文件尚未创建） |
| `docs/` | 空 | 预留文档目录 |
| `checkpoints/` | 空 | 模型保存目录（被 gitignore） |
| `logs/` | 空 | 日志目录（被 gitignore） |

---

## 三、架构设计解读

### 整体架构（README 中描述的蓝图）

```
keyboard_listener (子进程)          ← 监听键盘事件
        │
        │ multiprocessing.Queue     ← 跨进程通信队列
        ▼
HumanFeedbackEnvWrapper            ← 核心：在环境 reward 上叠加人类反馈
        │
Monitor (SB3 内置)                 ← 监控训练过程
        │
PPO.learn()                        ← PPO 算法训练
  + TrajectoryLoggerCallback        ← 记录轨迹
  + CheckpointCallback              ← 定期保存模型
```

### 设计模式

1. **策略模式 (Strategy Pattern)**: `FeedbackEvent` 作为统一协议，不同反馈源（keyboard/random/null/LLM）实现各自的策略
2. **数据驱动设计**: 所有超参数集中在 `config/default.yaml`，代码中不写魔法数字
3. **跨进程通信**: 键盘监听在子进程中运行，通过 `multiprocessing.Queue` 向主进程发送事件
4. **可扩展性**: `metadata` 字段为未来 LLM 反馈预留扩展点

---

## 四、与 DURF 项目的关系

根据你提供的 DURF proposal PDF，这个代码库对应的是：

```
Group A: Real-time Scalar Feedback (Control Group)
  ↓
Deep TAMER 框架：人类通过键盘实时给 RL agent 注入标量奖励信号 (+1/-1)
  ↓
作为对照组，与 Group B (LLM 自然语言反馈) 进行对比
```

**实验对比维度**:
| 维度 | Group A (本代码) | Group B (LLM 方案) |
|------|------------------|-------------------|
| 反馈方式 | 键盘按键 (Q=+1, E=-1) | 自然语言指令 |
| 奖励信号 | 标量 (+1/-1) | 结构化代码函数 |
| 信息密度 | 低（二值信号） | 高（多维语义） |
| 理论基础 | Deep TAMER | Code-as-Policy |

---

## 五、当前项目状态评估

### 已完成
- ✅ 项目结构搭建（目录、配置文件、协议定义）
- ✅ `FeedbackEvent` 数据协议设计
- ✅ 超参数集中管理 (`config/default.yaml`)
- ✅ 依赖清单 (`requirements.txt`)
- ✅ 项目文档 (`README.md`)

### 待完成（空目录）
- ❌ `src/eval/` — 评估模块
- ❌ `src/train/` — 训练模块（HumanFeedbackEnvWrapper、trainer、callbacks）
- ❌ `src/utils/` — 工具函数（seed 设置、IPC 辅助）
- ❌ `scripts/` — 启动脚本（三种模式的 .sh 文件 + smoke_test.py）
- ❌ `tests/` — 单元测试
- ❌ `docs/` — 架构图、设计决策文档

### 结论
> 这个项目目前处于**框架搭建阶段**，核心架构和协议已经设计好，但**实际的训练代码、环境包装器、键盘监听器、启动脚本等都还没有实现**。README 中描述的完整 pipeline 还是一个蓝图。

---

## 六、快速参考

### 如何运行（如果代码完整）
```bash
conda activate pantheonrl_env
pip install -r requirements.txt
# 三种模式：
python scripts/smoke_test.py          # 快速验证（5000 steps）
python scripts/train_no_feedback.sh   # 无反馈基线
python scripts/train_human_feedback.sh # 人类键盘反馈
```

### 键盘操作（human-feedback 模式）
| 按键 | 含义 | 信号值 |
|------|------|--------|
| Q | 正向反馈（做得好） | +1.0 |
| E | 负向反馈（做得差） | -1.0 |
| C | 暂停/教学模式 | 0.0 |

### 关键文件速查
| 想了解什么 | 看哪个文件 |
|-----------|-----------|
| 项目总览 | `README.md` |
| 所有参数 | `config/default.yaml` |
| 数据协议 | `src/feedback/protocol.py` |
| 依赖安装 | `requirements.txt` |
