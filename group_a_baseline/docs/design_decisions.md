# 关键设计决策

本文档记录 Group A Baseline 实现过程中的关键设计决策（D-XXX）。

---

## D-001: Feedback 使用 multiprocessing.Queue 而非 Pipe

**决策**: 使用 `multiprocessing.Queue` 在训练主进程和 feedback source 子进程之间传递 `FeedbackEvent`。

**理由**:
- Queue 天然支持多个 producer（未来可扩展多个 feedback source）
- Queue 是线程/进程安全的，无需额外锁
- Pipe 只能点对点，扩展性差
- Queue 的 `get(timeout=0)` 支持非阻塞读取，适合训练循环

**代价**: Queue 有轻微序列化开销，但对标量 feedback 可忽略。

---

## D-002: Wrapper 顺序 — Monitor 必须在最外层

**决策**: Env wrapper 顺序为 `Monitor > GymV21CompatibilityV0 > HumanFeedbackEnvWrapper > OvercookedMultiEnv`。

**理由**:
- SB3 issue #146: Monitor 必须是最外层 wrapper，否则记录的 reward 是注入前的值
- HumanFeedbackEnvWrapper 必须在 OvercookedMultiEnv 外面，才能拦截 step() 并注入 human_reward
- GymV21CompatibilityV0 在 HumanFeedbackEnvWrapper 外面，确保 Monitor 收到的是 gymnasium 格式

**验证**: 测试 `test_info_dict_has_breakdown` 确认 info 中包含 `env_reward` / `human_reward` / `total_reward`。

---

## D-003: 反馈注入策略 — 加法而非乘法

**决策**: `total_reward = env_reward + alpha * human_reward`

**理由**:
- 加法直观：alpha=0 时完全忽略人类反馈，alpha=1 时等权
- 乘法（如 `total = env * (1 + alpha * human)`）在 human_reward=0 时退化为 env_reward，但 human_reward 为负时可能导致 reward 变号
- 加法更容易调试和调参

**代价**: 需要合理设置 alpha 防止 human_reward 淹没 env_reward。

---

## D-004: 信号裁剪到 [-1, 1]

**决策**: 所有 human_reward 在注入前裁剪到 [-1, 1]。

**理由**:
- 防止键盘连击导致 reward 爆炸
- 与 Deep TAMER 的离散信号 (+1/-1) 一致
- 简化 alpha 调参（alpha 直接控制最大人类影响）

**实现**: `HumanFeedbackEnvWrapper._apply_feedback()` 中的 `np.clip(signal, -1.0, 1.0)`。

---

## D-005: FeedbackEvent 使用 frozen dataclass

**决策**: `FeedbackEvent` 使用 `@dataclass(frozen=True)`。

**理由**:
- 不可变对象天然线程安全
- 防止意外修改已发送的事件
- 可作为 dict key（hashable）
- 比 namedtuple 更灵活（支持类型注解、默认值）

**验证**: 测试 `test_frozen_immutable` 确认修改会抛 `FrozenInstanceError`。

---

## D-006: 三种 mode 通过替换子进程实现

**决策**: trainer.py 的三种 mode（no-feedback / random-feedback / human-feedback）通过启动不同的子进程实现，env wrapper 逻辑不变。

**理由**:
- 核心训练循环完全复用
- 添加新的 feedback source 只需新增一个子进程函数
- 便于测试：random-feedback 可替代真人进行 pipeline 验证

**实现**:
```python
if mode == "random-feedback":
    source_process = Process(target=run_random_source, args=(queue,))
elif mode == "human-feedback":
    source_process = Process(target=run_keyboard_listener, args=(queue,))
# no-feedback: queue=None, 不启动子进程
```

---

## D-007: 使用 shimmy 兼容 gym 0.25 → gymnasium

**决策**: 在 wrapper 栈中插入 `GymV21CompatibilityV0` 兼容层。

**理由**:
- PantheonRL 的 OvercookedMultiEnv 基于 gym 0.25（旧 API）
- SB3 2.x 的 Monitor 要求 gymnasium.Env（新 API）
- shimmy 是 Farama 官方提供的兼容库

**代价**: 增加了一层 wrapper，轻微性能开销。长期应升级 PantheonRL 到 gymnasium。

---

## D-008: 评估使用确定性策略

**决策**: `evaluator.py` 默认 `deterministic=True`。

**理由**:
- 确定性策略消除随机性，评估结果可复现
- 更真实反映模型学到的策略质量
- 便于不同 checkpoint 之间公平比较

**代价**: 可能低估模型的探索能力，但评估目的就是衡量已学到的策略。
