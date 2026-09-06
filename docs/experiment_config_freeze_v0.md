# 实验配置冻结 v0（freeze-v0）

> 冻结日期：2026-03-14
> 状态：**已冻结**。在正式数据采集和 Hu 训练期间，以下配置不再改动。
> 变更治理：任何改动（哪怕一行阈值）必须先新建 `freeze-vN` 版本并记录迁移，训练产物必须在 `metadata.json` 里记录所用 `freeze_version`。

本文件是训练前的单一权威配置清单。代码位置只作指针，**以本文件为准**；若代码与本文件冲突，按冲突处理流程走（见文末），而不是顺手改代码。

---

## 1. 数据协议（session 原始文件）

每局 session 目录下固定三个原始文件，schema 与 `play_with_baseline.py` 真实会话一致：

| 文件 | 作用 |
| --- | --- |
| `trajectory.csv` | 每步：动作、subgoal、condition features、task/coordination 决策、事件、reward、state 快照 |
| `chat_messages.csv` | 人类自然语言反馈（`role`/`content`） |
| `session_metadata.json` | 会话元信息；sim 会话标记 `data_source: synthetic_sim_human` |

- 旧版 `feedback.csv`（J/K 标量）**不再生成**，只在旧 session 兼容读取。
- 离线转换产物：`trajectory.jsonl`、`feedback_events.jsonl`、`probe_hits.jsonl`、`hu_subgoal_preferences.jsonl`、`hu_attribution_provenance.jsonl`。

## 2. 决策层级与协调词汇

运行时决策链：

```text
task（subgoal 选择）→ recovery（低层执行修正）→ coordination（路径冲突）→ safety
```

Coordination 是**独立决策域**，只从运行时构造的选项池选择，永不混入 task 池：

```text
COORDINATION_OPTIONS = { CONTINUE_CURRENT_SUBGOAL, YIELD }
```

- `HOLD_POSITION` 已折叠进 `YIELD` 的 stay 动作；`REROUTE` 属于重规划层，不进协调词汇。
- Hu 训练样本**禁止跨域对比**（task vs coordination）；构造时即抛错。

### 让路先验（condition-sensitive，双向保留）

冲突类型由 `path_conflict_type` 判定，四种：`human_entering_ai_tile` / `contested_destination` / `ai_entering_human_tile` / `ai_blocking_human_route`。

默认先验 YIELD > CONTINUE（复现旧礼貌 wrapper）。**当 AI 距当前 subgoal 目标一步（`ai_adjacent_to_current_subgoal_target=true`）时反转先验：CONTINUE > YIELD**——靠近自己目标时坚持原任务，由人绕路。Hu 仍可翻转任一顺序。两个方向的反馈都会采集。

运行时参数（`CoordinationController`）：

```text
min_commit_steps = 1
max_option_steps = 3
yield_cooldown_steps = 2
```

## 3. Candidate Event Taxonomy（17 个，冻结）

| event_type | actor | valence | 说明 |
| --- | --- | --- | --- |
| `AI_blocked_human_path` | ai | negative_problem | 人撞 AI 挡路 |
| `AI_failed_to_yield_or_clear_path` | ai | negative_problem | AI 挡路但不直接撞人（新触发面） |
| `AI_ignored_ready_or_nearly_ready_pot` | ai | missed_opportunity | 锅煮/好时空手或拿盘却不 interact |
| `AI_failed_to_prepare_ingredient_while_waiting` | ai | missed_opportunity | 等待期间不备料 |
| `AI_missed_plate_pickup_opportunity` | ai | missed_opportunity | 锅煮/好时空手不拿盘，连续 3 帧 |
| `AI_missed_useful_counter_object` | ai | missed_opportunity | 有更优 counter 物体却不取 |
| `AI_missed_labor_division_opportunity` | ai | missed_opportunity | 角色分工错位（人已覆盖某角色） |
| `AI_pick_drop_loop` | ai | negative_problem | 同格同物反复拿放 ≥3 次 |
| `AI_held_unneeded_object_too_long` | ai | negative_problem | 持无用原料 15 步窗口累计 ≥5 帧 |
| `AI_put_object_on_unhelpful_counter` | ai | neutral_context | 放到远处 counter，极性交给人 |
| `AI_successfully_put_ingredient_into_pot` | ai | positive_progress | 成功放原料入锅（≤1 格） |
| `AI_successfully_picked_up_soup` | ai | positive_progress | 成功取汤 |
| `AI_successfully_delivered_soup` | ai | positive_progress | 成功送餐（AI 持汤 + interact） |
| `Human_successfully_delivered_soup` | human | positive_progress | 上下文，不进 Hu |
| `Team_successfully_delivered_soup` | team | positive_progress | 上下文，不进 Hu |
| `AI_successfully_yielded` | ai | （运行时） | 运行时协调事件：让路成功 |
| `AI_maintained_current_subgoal_during_conflict` | ai | （运行时） | 运行时协调事件：冲突中坚持任务 |

- 归因优先级：负面反馈先匹配 negative_problem/missed_opportunity；正面反馈先匹配 positive_progress；`actor != ai` 只作上下文。
- `AI_failed_to_follow_human_path_when_faster` 未实现，**不在冻结范围**。
- 时间可见性：离线检测不得使用反馈之后的轨迹帧。

### 事件 → Hu 标签默认映射（partial）

完整成对映射只在事件自身指明双边的场合给出：

```text
AI_blocked_human_path / AI_failed_to_yield_or_clear_path
    → preferred=[YIELD], rejected=[CONTINUE_CURRENT_SUBGOAL]
AI_successfully_yielded → preferred=[YIELD]
AI_ignored_ready_or_nearly_ready_pot → preferred=[GET_DISH, PICKUP_SOUP]
AI_missed_plate_pickup_opportunity → preferred=[GET_DISH]
AI_failed_to_prepare_ingredient_while_waiting → preferred=[GET_USEFUL_INGREDIENT], rejected=[WAIT]
AI_successfully_picked_up_soup → preferred=[PICKUP_SOUP]
AI_successfully_delivered_soup → preferred=[SERVE_SOUP]
```

部分默认是 review 建议而非自动可训练标签。`AI_failed_to_yield` 仅作为 legacy 别名保留在默认映射里，无任何检测器产出，不在事件清单中。

补充映射（冻结时点新增）：

```text
AI_maintained_current_subgoal_during_conflict → preferred=[CONTINUE_CURRENT_SUBGOAL]
```

基线归因 keyword（`sample_builder.EVENT_KEYWORDS`）冻结时点覆盖：
- `AI_blocked_human_path` / `AI_failed_to_yield_or_clear_path`：blocking / in my way / pass / step aside / move / let me through 等让路诉求词；
- `AI_successfully_yielded`：let me through / get by / squeeze past / thanks；
- `AI_maintained_current_subgoal_during_conflict`：finish what you were doing / keep going / stick with it / don't mind me / closer to your task。

## 4. Hu Subgoal 词汇（14 个，冻结）

```text
TASK_HU_SUBGOALS（12）= GET_TOMATO, PUT_TOMATO_IN_POT, GET_ONION,
  PUT_ONION_IN_POT, GET_DISH, PICKUP_SOUP, SERVE_SOUP, WAIT,
  WAIT_NEAR_POT, PUT_DOWN_OBJECT, GET_USEFUL_INGREDIENT

COORDINATION_SUBGOALS（2）= CONTINUE_CURRENT_SUBGOAL, YIELD
```

别名只做归一化（`DELIVER_SOUP→SERVE_SOUP` 等），不扩词汇。

## 5. Condition Schema（冻结）

### Hu 模型布尔条件（28 个，`MODEL_CONDITION_KEYS`）

```text
pot_empty, pot_partially_filled, pot_cooking_or_ready,
human_has_dish, ai_has_dish, human_has_tomato, human_has_onion,
human_has_soup, ai_has_tomato, ai_has_onion, ai_has_soup,
ai_empty_handed, recipe_needs_tomato, recipe_needs_onion,
human_trying_to_pass, narrow_corridor, ai_on_human_path,
useful_object_adjacent, human_waiting_near_pot,
human_holding_last_needed_ingredient, human_closer_to_dish,
human_closer_to_pot, ai_closer_to_ingredient,
useful_counter_object_available, useful_counter_object_closer_than_dispenser,
useful_counter_object_closer_to_pot_than_dispenser,
useful_counter_object_lower_task_cost_than_dispenser,
ai_adjacent_to_current_subgoal_target
```

编码：`True→1, False→-1, None→0`。

### Coordination 域专用条件（14 个，`COORDINATION_CONDITION_KEYS`）

```text
human_trying_to_pass, narrow_corridor, ai_on_human_path,
ai_adjacent_to_current_subgoal_target, ai_empty_handed,
human_has_dish, human_has_tomato, human_has_onion, human_has_soup,
ai_has_dish, ai_has_tomato, ai_has_onion, ai_has_soup,
pot_cooking_or_ready
```

### 溯源上下文（17 个，`CONDITION_CONTEXT_KEYS`，仅供 provenance/review，不进模型）

```text
needed_ingredient, human_inferred_subgoal, human_distance_to_dish,
ai_distance_to_dish, human_distance_to_pot, ai_distance_to_pot,
human_distance_to_needed_ingredient, ai_distance_to_needed_ingredient,
useful_counter_object_type, useful_counter_object_position,
useful_counter_object_distance, useful_counter_object_pot_distance,
useful_counter_total_task_distance, matching_dispenser_distance,
matching_dispenser_pot_distance, matching_dispenser_total_task_distance,
ai_current_subgoal
```

## 6. Probe 集（probe-state-v0，8 个，冻结）

| probe_name | domain | 触发条件 | preferred | rejected |
| --- | --- | --- | --- | --- |
| `pot_ready_ai_should_get_dish` | task | pot_cooking_or_ready, ai_empty_handed | GET_DISH | WAIT, GET_TOMATO, GET_ONION |
| `human_path_conflict_ai_should_yield` | coordination | human_trying_to_pass, ai_on_human_path | YIELD | CONTINUE_CURRENT_SUBGOAL |
| `ai_holding_unneeded_onion_should_put_down` | task | ai_has_onion, !recipe_needs_onion | PUT_DOWN_OBJECT | WAIT |
| `ai_holding_unneeded_tomato_should_put_down` | task | ai_has_tomato, !recipe_needs_tomato | PUT_DOWN_OBJECT | WAIT |
| `pot_needs_tomato_ai_should_get_tomato` | task | recipe_needs_tomato, ai_empty_handed, !pot_cooking_or_ready | GET_TOMATO | WAIT |
| `pot_needs_onion_ai_should_get_onion` | task | recipe_needs_onion, ai_empty_handed, !pot_cooking_or_ready | GET_ONION | WAIT |
| `human_waiting_ai_should_prepare_next` | task | human_waiting_near_pot, ai_empty_handed | GET_TOMATO, GET_ONION | WAIT |
| `useful_object_adjacent_ai_should_consider_pickup` | task | useful_object_adjacent, ai_empty_handed | GET_TOMATO, GET_ONION, GET_DISH | WAIT |

规则：
- task probe 读 `ai_subgoal_candidates`；coordination probe 读 `coordination_decision.candidates`，取 `selected` 为 chosen。
- coordination probe 命中但该步无 coordination 记录 → `evaluation_unavailable=true`，统计时忽略并单独计数。
- 同 episode 内最小间隔 `--min-gap-steps`（默认 3 帧）。

## 7. Hu 模型与训练协议（冻结）

```text
模型：HierarchicalHu，Task/Coordination 独立线性 pairwise head
分数：score(user, condition, subgoal)
    = global_subgoal_bias[subgoal]
    + user_subgoal_bias[user, subgoal]
    + Σ_k condition_weight[k, subgoal] · condition[k]
损失：-log sigmoid(score(preferred) - score(rejected))
```

训练参数（`train_subgoal_reranker.py` 默认）：

```text
--epochs 200
--learning-rate 0.05
--l2 1e-4
--validation-fraction 0.2
--seed 0
```

评估协议（**per-participant，强制**）：

```text
第一局收集训练数据 → 训练 → 同一参与者再玩一局 → --test-dataset 测第二局
```

`check_participant_protocol` 硬性检查：
- test 出现无训练数据的 user → ValueError；
- test 反馈早于该 user 最新训练反馈 → `temporal_warnings`；
- 与训练共享 `source_feedback_id` 的样本剔除（防泄漏）；
- 训练集内完全重复的 (feedback_id, pair) 去重。

验证集只用于 early stopping，不参与最终指标。

## 8. Sim 会话参数（冻结默认值）

```text
--layout ring_tomato_onion_10x6_h0_full_task
--seed 42  --horizon 800
--sim-human cooperative | selfish（默认 cooperative）
--sim-seed 0
--sim-feedback-min-gap 15
--sim-ai-stubborn-prob 0.0（每冲突概率强制 CONTINUE，用于补 CONTINUE>YIELD 场景）
--sim-stall-escape-delay 8
--hu-user-id PILOT01
```

- sim 会话 `data_source: synthetic_sim_human`，永远不与真实数据混淆。
- persona 极性决定反馈正负：cooperative 尊重 AI 一步到目标时的坚持，selfish 一律要求让路；语言由模板生成，极性不靠文本猜测。
- 反馈模板四向齐全：YIELD/CONTINUE × 正/负。

## 9. 变更治理

冻结不是“不许改”，而是“改了必须显式升级”：

1. 任何 schema/事件/条件/探针/参数/词汇改动 → 新建 `docs/experiment_config_freeze_vN.md`，注明相对 v0 的差异和迁移影响。
2. 训练产物的 `metadata.json` 必须写入 `freeze_version`。
3. 已采集数据按采集时的 freeze 版本归档，不做追溯重标（除非迁移文档明确要求且全员同意）。
4. 代码与本文件冲突时：先在 issue 里登记差异 → 决定以谁为准 → 更新对应一方，并记录。

## 10. 冻结时点的仓库状态

```text
branch: dev/human-ai-feedback
HEAD: 723426a Trim coordination vocabulary to runtime-constructible options
未提交改动：P0/P1/3/4b 批次 + sim_session.py（synthetic sim-human 会话）
  + 归因 keyword 对齐（sample_builder.py）+ AI_maintained 默认映射
  （subgoal_preferences.py）
41 focused tests 通过
sim 全链路验证（2026-03-14）：
  - selfish 300 步会话：5 条反馈，4 条归因成协调 pair（YIELD>CONTINUE）
    × 4，条件分布有区分度（ai_adjacent 3真/1假、human_trying_to_pass 3真/1假）
  - cooperative 300 步会话：正向让路/坚持反馈归因正确（单边 preferred，
    需 review 补 rejected 边才进训练集）
  - 4 pair 训练 smoke：coordination head 3 train / 1 val，
    pairwise_accuracy 1.0（样本极少，仅链路信号非实验证据）
```

### 2026-03-14 追加（freeze-v0 之后，数据管线增强，不触发 freeze-v1）

本次改动全部在**数据源/训练样本构造**侧，未触碰冻结的 schema/事件/条件/词汇/探针/训练参数，按变更治理第 2 条仍在 v0 下运行，训练产物 `sim_both_v0` 记录 `freeze_version: v0`：

- `sim_session.py`：新增 task 域反馈模板（`_TASK_NEG_READY_POT` / `_TASK_NEG_PREP_WAIT`），`SimHuman.maybe_feedback` 增加 task-event streak 检测（阈值与 `event_detectors` 对齐：ready-pot >=3、prep-wait >=4），模板文本经 `sample_builder` keyword 核对无跨事件歧义。
- `hu_dataset_builder.py`：coordination 二元域单边补全——单边 preferred/rejected 自动补另一边，使 cooperative 正向坚持反馈可进训练集。
- 8 个新 sim 会话（20260806_133421~133527）+ `outputs/hu_models/sim_both_v0`（train 4 / test 4，protocol_ok）。