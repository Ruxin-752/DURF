# Candidate Event Taxonomy

本文档说明当前 candidate event 的分类口径和检测逻辑。核心原则是：

```text
candidate event = 程序从轨迹中检测出的候选事实
attribution target = 某条人类反馈最终指向的事件
Hu training sample = 经过归因/review 后得到的 condition + preferred/rejected subgoal
```

因此，candidate event 可以比最终训练样本多。它们不是标签，只是给规则 baseline、LLM 和人工 review 看的候选证据。

## 新增字段

每条 `candidate_event` 现在都有两个分类字段：

- `actor`
  - `ai`: 事件主要由 AI 行为触发。
  - `human`: 事件主要由人类行为触发。
  - `team`: 只能确定团队结果，无法可靠归到某个单人。
  - `unknown`: 事实不足。

- `event_valence`
  - `positive_progress`: 正向任务进展，比如成功放锅、成功送餐。
  - `negative_problem`: 明显问题行为，比如反复拿放、把东西放到远离锅的位置。
  - `missed_opportunity`: 错失机会，比如锅在煮时没有准备下一步。
  - `neutral_context`: 上下文或事实不足，不应直接作为 Hu 标签。

归因时：

- 负面反馈优先匹配 `negative_problem` / `missed_opportunity`。
- 正面反馈才优先匹配 `positive_progress`。
- `actor != ai` 的事件默认只作为上下文，不直接进入 Hu 训练样本。

## 当前 Event 分类表

| event_type | actor | event_valence | 检测逻辑 | 当前判断 |
| --- | --- | --- | --- | --- |
| `AI_blocked_human_path` | `ai` | `negative_problem` | 人类尝试移动到 AI 当前/下一格，但位置没变。 | 合理，偏保守。只能检测明确“人撞 AI”的阻挡，不能覆盖“AI 站位让我绕路”。 |
| `AI_ignored_ready_or_nearly_ready_pot` | `ai` | `missed_opportunity` | pot ready，AI 空手或拿盘，连续数步没有在锅附近 interact。 | 合理，但目前只看 ready pot，对“快煮好该拿盘”覆盖不足。 |
| `AI_failed_to_prepare_ingredient_while_waiting` | `ai` | `missed_opportunity` | pot cooking，人类拿盘等待，AI 空手但没有准备原料。 | 合理，但条件偏窄：要求人类拿盘等待。 |
| `AI_missed_plate_pickup_opportunity` | `ai` | `missed_opportunity` | soup cooking/ready，AI 空手，人类没拿盘，AI 没选择盘子相关 subgoal。 | 方向合理，但容易依赖 subgoal 日志质量。后续应增加“附近有盘子/盘子位置更近”的 condition。 |
| `AI_missed_useful_counter_object` | `ai` | `missed_opportunity` | AI 从 dispenser 取物时，同类 counter 物体已经存在，且它更靠近 AI、完整任务成本更低，或更靠近锅。 | 已实现。最后一种只表示“队友已 staging”的候选偏好，不宣称客观总路径更短，必须结合反馈和 review。 |
| `AI_missed_labor_division_opportunity` | `ai` | `missed_opportunity` | 当前覆盖两个保守情形：人类已拿最后所需原料而 AI 未准备盘子；人类已覆盖盘子角色而 AI 重复选择盘子任务。 | 已实现。`opportunity_kind` 保存在 evidence，具体状态保存在 condition。 |
| `AI_pick_drop_loop` | `ai` | `negative_problem` | 短窗口内 AI 多次 interact 导致手持物反复变化，且无送餐 reward。 | 已修正重叠窗口问题。仍需注意：有些合理 staging 也可能被误报，需要 review。 |
| `AI_held_unneeded_object_too_long` | `ai` | `negative_problem` | AI 持有 recipe 已不需要的 tomato/onion 连续若干步。 | 合理，但依赖 `recipe_needs_*` condition 是否准确。 |
| `AI_put_object_on_unhelpful_counter` | `ai` | `negative_problem` | AI 把物体放到离 pot 较远的 counter。 | 已加同位置短期去重。风险是“临时让路放下”可能被误判。 |
| `AI_successfully_put_ingredient_into_pot` | `ai` | `positive_progress` | AI step 前持 tomato/onion，step 后不持有，soup ingredients 增加。 | 合理。出现多次是正常的，因为每放一个原料都会记录一次。 |
| `AI_successfully_picked_up_soup` | `ai` | `positive_progress` | AI step 前持 dish，step 后持 soup。 | 合理。 |
| `AI_successfully_delivered_soup` | `ai` | `positive_progress` | reward 增加，且 AI step 前持 soup，AI action 是 interact。 | 已修正。不会再把人类送餐算成 AI 送餐。 |
| `Human_successfully_delivered_soup` | `human` | `positive_progress` | reward 增加，且人类 step 前持 soup，人类 action 是 interact。 | 作为上下文保留，不进入 Hu 训练。 |
| `Team_successfully_delivered_soup` | `team` | `positive_progress` | reward 增加，但无法从持汤者/action 可靠判断谁送餐。 | 作为上下文保留，不进入 Hu 训练。 |

## 复盘：为什么成功放锅会很多

`AI_successfully_put_ingredient_into_pot` 是“单个动作事实”，不是“一锅汤事件”。一份 `tomato + tomato + onion` 汤至少需要三次放原料；如果一局做多份汤，这个事件自然会多次出现。

这不是 bug。真正的问题是：它以前和问题事件混在同一层候选池里。现在用 `event_valence=positive_progress` 标出后，负面反馈归因会降低它的优先级，review 窗口也可以把它当作正向进展/上下文来看。

## Condition Schema

Hu-v0 使用固定布尔条件；Review/溯源文件同时保留原始类别和距离：

- 新增 Hu 布尔条件：
  - `human_holding_last_needed_ingredient`
  - `human_closer_to_dish`
  - `human_closer_to_pot`
  - `ai_closer_to_ingredient`
  - `useful_counter_object_available`
  - `useful_counter_object_closer_than_dispenser`
  - `useful_counter_object_closer_to_pot_than_dispenser`
  - `useful_counter_object_lower_task_cost_than_dispenser`
- 新增溯源上下文：
  - `needed_ingredient`
  - `human_inferred_subgoal`
  - 双方到盘子、锅和所需原料的路径距离
  - counter 物体类型、位置、到 AI/锅的距离
  - counter 与 dispenser 的完整任务路径成本

`AI -> source -> pot` 总成本与“物体是否已经 staging 在锅边”是两个不同事实。前者可用于效率判断，后者可以承载用户协作偏好。

## 当前仍需扩展的 Event

下一步经 Review 确认后再考虑：

- `AI_failed_to_follow_human_path_when_faster`
  - 用于“你应该跟着我/从我这边走更快”。

- `AI_failed_to_yield_or_clear_path`
  - 比 `AI_blocked_human_path` 更宽，覆盖“AI 没直接撞人，但站位让人难走”。

这些不应该把条件写进 event 名。event 只描述 AI 行为；具体情境写入 `condition_features` 和 evidence。

## 时间可见性

离线处理不得使用反馈之后的轨迹。针对每条反馈，`sample_builder.py` 会在截至反馈 timestep 的近期窗口内重新运行 detector。完整 session 中某事件即使持续到未来，归因证据也只保留反馈发生时已经可见的前缀。
