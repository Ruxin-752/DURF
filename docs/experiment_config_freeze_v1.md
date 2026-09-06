# 实验配置冻结 v1（freeze-v1）

> 冻结日期：2026-03-14
> 状态：**已冻结**。相对 freeze-v0 的变更见下方"相对 v0 的差异"。
> 上游基线：`docs/experiment_config_freeze_v0.md`（v0 全部条款继续生效，除非本文件显式覆盖）。

本文件只记录 freeze-v1 相对 v0 的差异。未列出的配置一律沿用 v0。

---

## 相对 v0 的差异

### 1. 协调冲突的触发面扩展（`_ai_blocks_target_line` 两步检测）

**v0**：selfish persona 的 push 触发要求人类与 AI 曼哈顿距离 == 1 且 AI 站在任务目标直线上。

**v1**：直线检测扩展为"前两步"（`ahead` 与 `ahead2`）。AI 距离目标 2 格时，selfish 人类先 `_step_toward_action` 靠近，下一步推挤，制造"AI 远离自身目标时挡路"的冲突。

**动机**：v0 数据里 `ai_adjacent_to_current_subgoal_target=false` 的协调样本仅占 13%，模型无法把"挡路"与"临近目标"解耦。

**迁移影响**：无 schema 改动，仅 sim 生成行为变化。已采集 v0 数据不追溯重标。

### 2. `ai_on_human_path` 与 `human_trying_to_pass` 语义拆分（在线 sim）

**v0**：两者在 sim 在线分支里被赋成同一个布尔（`route_blocked`），恒等共现。

**v1**（`AiRuntime.step` 内，仅 sim 在线路径）：
- `ai_on_human_path`：纯几何事实。人类当前步移动时用当步 delta 投影；人类静止时用 `_last_human_delta`（上一次移动方向）投影，AI 占据人类下一格或下下格即为真。
- `human_trying_to_pass`：主动意图。仅当人类**当步**在移动且路径被挡时为真。

离线回放（`condition_features.extract_condition_features`）无法重构人类上一帧移动方向，保持两值相等——这是文档化的限制，不影响 sim 训练数据（sim 训练样本全部来自在线路径）。

**动机**：v0 数据里两条件 100% 共现，模型对 `ai_on_human_path` 学到反直觉方向（挡路→CONTINUE +0.81）。

**迁移影响**：sim 会话的 trajectory 记录里两者可能不同（v1 新增区分），归因/训练读的是条件特征 dict，无 schema 变化。

### 3. 协调反馈极性重排（`_evaluate_coordination`）

**v0**（bug）：
```
CONTINUE + adjacent + persona!=selfish → POS_CONTINUE   # 吞掉挡路信号
CONTINUE + human_trying_to_pass      → NEG_CONTINUE   # 对 non-selfish 永远走不到
```

**v1**：
```
CONTINUE + blocking + !adjacent → NEG_CONTINUE        # 挡路且远离目标：必须让
CONTINUE + blocking + adjacent:
    cooperative/lenient → POS_CONTINUE                 # 一步到位：允许坚持
    selfish/polite      → NEG_CONTINUE                 # 依然要求让路
CONTINUE + !blocking + adjacent → POS_CONTINUE         # 不挡路且快到位：坚持正确
```

**动机**：v0 中"挡路但快到位→继续走"在 cooperative persona 被无条件判正，制造了反直觉标签；同时 A=1 桶内标签多样性靠这条 bug 才存在，重排后需显式保留 persona 分支。

### 4. 人格矩阵扩展：`polite` / `lenient`

**v0**：`cooperative | selfish` 两值，行为与极性耦合。

**v1**：4 值正交矩阵（2 行为 × 2 极性）：

| persona | 行为 | 极性 |
| --- | --- | --- |
| `cooperative` | 礼貌绕路（30% bump） | 宽容：一步到位可坚持 |
| `selfish` | 推挤（100% push） | 严格：挡路必让 |
| `polite` | 礼貌绕路 | 严格：挡路必让 |
| `lenient` | 礼貌绕路 | 极端宽容：从不批评坚持 |

**动机**：用户要求"多构造几组不同人格机制的数据进行测试"，验证模型在未见过的行为×极性组合上的泛化。

**迁移影响**：CLI `--sim-human` 接受 4 值；v0 的 `cooperative/selfish` 行为不变。

---

## 冻结时点证据（2026-03-14）

### 解耦效果（条件分布，协调样本）

| 指标 | v0 数据（31 样本） | v1 数据（30 样本） |
| --- | --- | --- |
| adjacent=True 时 P/T 共现 | 70% | 46% |
| adjacent=True & 非挡路样本占比 | 25.8% | 43.3% |
| adjacent=False 占比 | 12.9% | 20% |
| A=1 桶标签多样性 | 14Y/13C | 14Y/10C |

### 训练对比（同协议：train coop/selfish × seed7/11，test × seed13/17）

| | sim_both_v0 | sim_both_v1 |
| --- | --- | --- |
| task test acc | 0.92（25 对） | 0.92（25 对） |
| coordination test acc | 0.667（12 对） | **0.923**（13 对） |
| coordination mean/min margin | +1.61 / -4.82 | +4.60 / -2.74 |

coordination acc 显著提升（+0.256），判定为"解耦生效，实行改动"。

**残留问题（如实记录）**：v1 coordination head 的 `ai_on_human_path → CONTINUE +1.51`、`human_trying_to_pass → CONTINUE +1.50` 仍是正方向——这是 v1 数据里"挡路+坚持被表扬"（cooperative/lenient 极性）的自然结果，模型正确地学到了"人格决定极性"。要消除需在训练时区分 persona 或加 persona 条件特征，属于后续工作，不在本 freeze 内。

### 跨人格泛化（sim_both_v1 模型，4 persona 各 4 会话）

| persona | 协调样本 | acc | mean/min margin |
| --- | --- | --- | --- |
| cooperative（见过） | 10 | 1.000 | +4.58 / +3.97 |
| selfish（见过） | 20 | 0.950 | +6.94 / -2.74 |
| polite（未见） | 10 | 0.900 | +6.45 / -3.97 |
| lenient（未见） | 10 | 1.000 | +4.58 / +3.97 |

模型对未见过的行为×极性组合保持 ≥0.90，说明学到的是"条件→极性"规律而非记忆 persona。

### 测试

56 focused tests 通过（feedback_attribution / coordination / baseline_task_logic / review_replay）。

---

## 仓库状态

```text
branch: dev/human-ai-feedback
改动文件：
  durf/group_a/sim_session.py          （触发面、特征拆分、反馈极性、persona 矩阵）
  durf/feedback_attribution/condition_features.py（离线回放注释说明）
训练产物：
  outputs/hu_models/sim_both_v1/       （freeze_version: v1）
  outputs/human_ai_sessions/20260806_144949~145548（16 会话：4 persona × 4 seed）
```