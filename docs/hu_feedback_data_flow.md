# Hu 反馈归因数据流说明

> 状态说明：本文保留最初的数据流设计。当前代码已实现双头 Hu、在线
> shadow/apply、显式 Coordination 决策日志以及人工 review 覆盖。最新口径见
> [`hierarchical_hu_runtime.md`](hierarchical_hu_runtime.md)。

本文档记录当前版本的研究数据流。目标不是立刻实现 Hu 模型本体，而是先把“玩家自然语言反馈如何被保存、归因、转换为 Hu 可学习标签”这条链路打通。

## 1. 当前统一方案

我们现在采用的是“任务执行 backbone + Hu 偏好评分层”的设计：

- 任务执行 backbone 负责把游戏做完。当前可用实现主要是 subgoal executor / planner，而不是纯 PPO。
- Hu 暂时不直接改环境奖励函数，也不直接替代任务执行器。
- Hu 学习的是：在某个状态条件下，人类更喜欢哪个 subgoal，不喜欢哪个 subgoal。
- 最终决策时，可以把任务分数和 Hu 偏好分数组合：

```text
final_score(subgoal)
= task_score(subgoal)
+ lambda * Hu(user, condition_features, subgoal)
```

当前版本已经在这条数据流之后实现 Hu 模型与在线评分；本节公式仍用于
Task 决策域，Coordination 决策域使用独立的候选集合和评分头。

## 2. 核心数据流

### Step 0: 玩家游玩并产生原始日志

入口文件：

- `durf/group_a/play_with_baseline.py`

主要输入：

- layout，例如 `ring_tomato_onion_10x6_h0_full_task`
- agent / executor，例如 `RllibRingCurrentBestPurePpo30Pct` 或当前 subgoal executor
- 玩家键盘操作
- 玩家和 LLM 的对话或反馈

主要输出目录：

```text
outputs/human_ai_sessions/<session_id>/
```

主要原始文件：

- `trajectory.csv`
- `feedback.csv`
- `chat_messages.csv`
- `pause_events.csv`

其中 `trajectory.csv` 是最重要的原始轨迹文件。当前版本已经记录：

- `episode`, `episode_step`, `total_step`
- 人类动作与 AI 动作
- `ai_subgoal`: 当前 AI 正在执行的 subgoal
- `ai_condition_features_json`: 当前在线决策时使用的固定 condition features
- `ai_subgoal_candidates_json`: 当前 step 的候选 subgoal、任务分数、原因和未来 Hu 分数位置
- `ai_event`: 当前执行器暴露出的事件标签，若没有则为空
- `state_before_json`: 环境 step 前的状态事实
- `state_after_json`: 环境 step 后的状态事实
- `environment_reward`, `episode_reward`, `done`

`state_before_json` 和 `state_after_json` 中通常包含：

- AI 位置
- 人类位置
- AI 手持物
- 人类手持物
- objects
- pot states
- layout terrain

这些信息是后续事件检测和条件提取的基础。

### Step 1: CSV 转换为标准 JSONL

入口文件：

- `durf/feedback_attribution/session_converter.py`

输入：

```text
outputs/human_ai_sessions/<session_id>/trajectory.csv
outputs/human_ai_sessions/<session_id>/feedback.csv
outputs/human_ai_sessions/<session_id>/chat_messages.csv
```

输出：

```text
outputs/human_ai_sessions/<session_id>/trajectory.jsonl
outputs/human_ai_sessions/<session_id>/feedback_events.jsonl
```

`trajectory.jsonl` 每一行是一条标准化 step：

```json
{
  "record_type": "trajectory_step",
  "episode": 1,
  "episode_step": 42,
  "total_step": 42,
  "layout": "ring_tomato_onion_10x6_h0_full_task",
  "ai_action_name": "east",
  "human_action_name": "stay",
  "ai_subgoal": "GET_TOMATO",
  "ai_condition_features": {
    "pot_empty": true,
    "ai_empty_handed": true,
    "recipe_needs_tomato": true
  },
  "ai_subgoal_candidates": [
    {
      "subgoal": "GET_TOMATO",
      "task_score": 70.0,
      "hu_score": 0.0,
      "final_score": 70.0,
      "reason": "pot_needs_ingredient"
    }
  ],
  "ai_event": null,
  "state_facts": {
    "ai_pos": [3, 1],
    "human_pos": [6, 1],
    "ai_held_object": null,
    "human_held_object": null,
    "pot_states": {"empty": [[4, 0]]},
    "layout_features": {"layout_name": "...", "terrain": ["..."]}
  }
}
```

`reason` 是审计字段，只解释 task scorer 的规则来源，不作为 Hu 训练输入。Hu 训练输入以 `ai_condition_features` / `condition_features` 和 `preferred_subgoal/rejected_subgoal` 为主。

`feedback_events.jsonl` 每一行是一条玩家反馈事件：

```json
{
  "record_type": "feedback_event",
  "role": "human_language",
  "feedback_text": "你可以等我路过之后再拿洋葱",
  "total_step": 88,
  "episode": 1,
  "extra": {
    "layout": "ring_tomato_onion_10x6_h0_full_task"
  }
}
```

### Step 2: 从轨迹中提取固定条件字段

入口文件：

- `durf/feedback_attribution/condition_features.py`

作用：

把复杂的 `state_facts` 压缩成 Hu 可以稳定使用的一组布尔条件字段。

当前固定字段包括：

```text
pot_empty
pot_partially_filled
pot_cooking_or_ready
human_has_dish
ai_has_dish
human_has_tomato
human_has_onion
ai_has_tomato
ai_has_onion
ai_empty_handed
recipe_needs_tomato
recipe_needs_onion
human_trying_to_pass
narrow_corridor
ai_on_human_path
useful_object_adjacent
human_waiting_near_pot
```

设计原则：

- condition 是“当时状态是否满足某些条件”。
- condition 不负责表达玩家喜欢什么。
- condition 应该尽量保持固定 schema，避免每条样本随意长出不同字段。
- 如果某些字段无法从旧日志中恢复，则用 `null`，而不是硬猜。

### Step 3: 程序检测候选事件

入口文件：

- `durf/feedback_attribution/event_detectors.py`
- `durf/feedback_attribution/generate_candidate_events.py`

输入：

```text
trajectory.jsonl
```

输出：

```text
candidate_events.jsonl
```

candidate event 的定义：

candidate event 是程序从轨迹中检测到的“可能被玩家反馈指向的行为事实”。它不是 Hu 的最终训练标签，而是给 LLM / 研究者做归因时看的候选证据。

当前已有事件包括：

- `AI_blocked_human_path`: AI 挡住人类路径。
- `AI_ignored_ready_or_nearly_ready_pot`: 锅快好或已好，但 AI 没有响应。
- `AI_failed_to_prepare_ingredient_while_waiting`: 等待期间 AI 没有准备有用食材。

一条 candidate event 大致长这样：

```json
{
  "record_type": "candidate_event",
  "event_type": "AI_failed_to_prepare_ingredient_while_waiting",
  "start_timestep": 51,
  "end_timestep": 65,
  "related_subgoal": "WAIT",
  "condition_features": {
    "pot_cooking_or_ready": true,
    "human_has_dish": true,
    "ai_empty_handed": true,
    "human_waiting_near_pot": true
  },
  "evidence": {
    "duration_steps": 15,
    "ai_actions": ["stay", "stay", "..."],
    "human_actions": ["interact", "east", "..."],
    "reason": "Pot was cooking while human held a dish, but AI stayed empty-handed..."
  }
}
```

这里的 `related_subgoal` 是“这个事件发生时 AI 原本在执行什么”。它不是最终偏好结论。

### Step 4: 反馈归因

入口文件：

- `durf/feedback_attribution/sample_builder.py`
- `durf/feedback_attribution/llm_attributor.py`

输入：

```text
feedback_events.jsonl
trajectory.jsonl
candidate_events.jsonl
```

输出：

```text
attribution_preview.jsonl
llm_attribution_audit.jsonl  # 仅 use-llm 时生成
```

这一层回答的问题是：

```text
这条玩家反馈，最可能指向哪一段时间、哪一个候选事件、哪些 subgoal 偏好？
```

非 LLM baseline：

- 用时间窗口和简单规则把反馈对齐到附近事件。
- 如果无法可靠归因，则标记 `needs_clarification = true`。

LLM attribution：

- 输入自然语言反馈、近期轨迹摘要、candidate events。
- LLM 不直接改奖励函数。
- LLM 只负责语义相关、程序规则解释不清的部分：
  - 玩家说的“别堵我”指向哪个 candidate event？
  - 玩家说的“你等我的时候可以先备菜”更像哪个事件？
  - 如果现有 event schema 无法覆盖，它可以提出 `proposed_schema_update`。

归因结果示例：

```json
{
  "record_type": "attribution_result",
  "feedback_event_id": "chat_messages.csv:1:...",
  "target_event": "AI_failed_to_prepare_ingredient_while_waiting",
  "target_time_window": [51, 65],
  "condition_features": {
    "pot_cooking_or_ready": true,
    "human_has_dish": true,
    "ai_empty_handed": true
  },
  "preferred_subgoals": ["GET_USEFUL_INGREDIENT"],
  "rejected_subgoals": ["WAIT"],
  "needs_clarification": false,
  "confidence": 0.74
}
```

### Step 5: 生成 Hu 溯源版和训练版数据

入口文件：

- `durf/feedback_attribution/hu_dataset_builder.py`

输入：

```text
trajectory.jsonl
feedback_events.jsonl
attribution_preview.jsonl
```

输出：

```text
hu_attribution_provenance.jsonl
hu_subgoal_preferences.jsonl
schema_updates.jsonl
hu_dataset_summary.json
```

#### 5.1 Hu 溯源版

文件：

```text
hu_attribution_provenance.jsonl
```

用途：

- 给研究者看完整证据链。
- 用于检查 LLM 是否误解反馈。
- 用于之后人工审核、修正、写论文方法部分。

包含：

- 原始反馈文本
- 反馈所属用户
- 目标事件
- 目标时间窗口
- event evidence
- condition features
- preferred / rejected subgoals
- 是否需要追问
- LLM 建议新增的 schema 字段

#### 5.2 Hu 训练版

文件：

```text
hu_subgoal_preferences.jsonl
```

用途：

- 之后喂给 Hu 模型。
- 格式更干净，只保留 pairwise subgoal preference。

样本格式：

```json
{
  "record_type": "hu_pairwise_subgoal_preference",
  "sample_id": "PILOT01:feedback_001:GET_USEFUL_INGREDIENT>WAIT",
  "user_id": "PILOT01",
  "layout": "ring_tomato_onion_10x6_h0_full_task",
  "condition_features": {
    "pot_cooking_or_ready": true,
    "human_has_dish": true,
    "ai_empty_handed": true
  },
  "preferred_subgoal": "GET_USEFUL_INGREDIENT",
  "rejected_subgoal": "WAIT",
  "source_event": "AI_failed_to_prepare_ingredient_while_waiting",
  "source_feedback_id": "feedback_001"
}
```

这条训练样本表达的是：

```text
在这些条件下，这个用户更偏好 GET_USEFUL_INGREDIENT，而不是 WAIT。
```

注意：

- `event` 是溯源证据，不是 Hu v0 的核心输入。
- `condition_features` 是 Hu v0 的核心输入。
- `preferred_subgoal/rejected_subgoal` 是 Hu v0 的监督标签。
- `polarity`, `preference_key`, `direction`, `strength` 暂时不进入 Hu v0 训练版，因为它们会把标签设计复杂化。

## 3. 一键运行命令

非 LLM 版本：

```powershell
cd "C:\Users\my185\Desktop\研究\durf\DURF"

python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 30
```

LLM 版本：

```powershell
cd "C:\Users\my185\Desktop\研究\durf\DURF"

$env:DEEPSEEK_API_KEY="你的 key"

python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 30 `
  --use-llm
```

运行后应看到类似输出：

```text
trajectory.jsonl records: 207
feedback_events.jsonl records: 0
candidate_events.jsonl records: 2
attribution_preview.jsonl records: 0
hu_attribution_provenance.jsonl records: 0
hu_subgoal_preferences.jsonl records: 0
schema_updates.jsonl records: 0
```

如果某局没有玩家反馈，Hu 样本为 0 是正常的。

## 4. 每个文件的作用

### `durf/group_a/play_with_baseline.py`

Pygame 人机协作入口。它负责启动 Overcooked 游戏、接入当前 AI、记录轨迹、暂停、玩家对话和游戏状态。

当前数据流依赖它记录：

- `trajectory.csv`
- `state_before_json`
- `state_after_json`
- `ai_subgoal`
- `ai_event`

### `durf/feedback_attribution/schemas.py`

统一所有 JSONL 的记录格式。它回答：

```text
trajectory_step、feedback_event、candidate_event、attribution_result 应该长什么样？
```

本次更新中，它新增支持：

- trajectory 中保存 `ai_subgoal` 和 `ai_event`
- candidate event 中保存 `related_subgoal` 和 `condition_features`
- attribution result 中保存 `condition_features`, `preferred_subgoals`, `rejected_subgoals`

### `durf/feedback_attribution/io_utils.py`

轻量读写工具。负责：

- 读 CSV
- 写 JSONL
- 读 JSONL
- 转换 int / float / bool / json 字段

### `durf/feedback_attribution/session_converter.py`

把原始 session 日志转换为标准 JSONL。它是数据流第一步。

它回答：

```text
游戏跑出来的 CSV 如何变成后续程序都能读的统一格式？
```

### `durf/feedback_attribution/condition_features.py`

条件特征提取器。它回答：

```text
在某个 timestep，哪些与人类偏好有关的状态条件成立？
```

例如：

- 人类是否拿着盘子？
- AI 是否空手？
- 锅是否正在煮或已经好了？
- 人类是否可能想通过某条路？
- AI 是否正在挡路？

### `durf/feedback_attribution/event_detectors.py`

事件识别规则库。它回答：

```text
这段轨迹中发生了哪些可能被玩家评价的行为事件？
```

它目前是程序规则，不是 LLM。这样做的原因是：

- 程序更擅长处理位置、物体、锅状态等客观事实。
- LLM 更适合处理“玩家这句话到底指哪个事件”的语义问题。
- 两者分工后，可解释性更强。

### `durf/feedback_attribution/generate_candidate_events.py`

命令行包装器。它负责读取 `trajectory.jsonl`，调用 `event_detectors.py`，写出 `candidate_events.jsonl`。

### `durf/feedback_attribution/sample_builder.py`

非 LLM attribution baseline。它回答：

```text
如果不用 LLM，只根据时间窗口和简单规则，这条反馈可能归因到哪里？
```

它也用于 LLM 失败时的 fallback。

### `durf/feedback_attribution/llm_attributor.py`

DeepSeek / LLM 归因器。它回答：

```text
玩家的自然语言到底在说哪个事件？更偏好哪个 subgoal？反对哪个 subgoal？
```

LLM 的职责边界：

- 可以做语义指代。
- 可以从候选事件中选择目标。
- 可以输出 preferred/rejected subgoals。
- 可以提出新的 event schema 建议。
- 不直接改环境 reward。
- 不直接生成最终策略。

### `durf/feedback_attribution/subgoal_preferences.py`

subgoal 偏好映射表。它回答：

```text
如果已经知道 target_event，默认可以转成什么 preferred/rejected subgoal？
```

例如：

```text
AI_failed_to_prepare_ingredient_while_waiting
-> preferred: GET_USEFUL_INGREDIENT
-> rejected: WAIT
```

LLM 输出优先；如果 LLM 没给出明确 subgoal，就用这里的默认映射。

### `durf/feedback_attribution/hu_dataset_builder.py`

Hu 数据集构建器。它把归因结果拆成两份：

- `hu_attribution_provenance.jsonl`: 研究者审核用，保留完整证据链。
- `hu_subgoal_preferences.jsonl`: Hu 训练用，保留最小 pairwise 标签。

### `durf/feedback_attribution/demo_offline_attribution.py`

当前一键离线管线。它串起：

```text
CSV logs
-> trajectory.jsonl / feedback_events.jsonl
-> candidate_events.jsonl
-> attribution_preview.jsonl
-> hu_attribution_provenance.jsonl
-> hu_subgoal_preferences.jsonl
```

这是目前最推荐先跑的入口。

### `durf/feedback_attribution/feedback_type_router.py`

早期反馈类型分类器。它可以初步区分 scalar / language / unclear 等反馈类型。当前主线仍保留它，但重点已经转移到 subgoal preference 数据流。

## 5. 当前已验证情况

已执行语法检查：

```powershell
python -m compileall durf\feedback_attribution
```

已用旧 session 验证旧日志兼容：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\20260624_221940 `
  --user-id PILOT_SMOKE `
  --lookback-steps 30
```

结果：

- 能生成 `trajectory.jsonl`
- 能生成 `feedback_events.jsonl`
- 能生成 `candidate_events.jsonl`
- 能生成 `attribution_preview.jsonl`
- 能生成 `hu_attribution_provenance.jsonl`
- 没有可靠 target event 时，不会硬生成 Hu 训练样本
- 旧日志缺 state facts 时，会写入 `schema_updates.jsonl`，提示需要更完整状态字段

已用当前 H0 session 验证新日志兼容：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\20260714_225025 `
  --user-id PILOT_SMOKE `
  --lookback-steps 30
```

结果：

- `trajectory.jsonl` 成功保留 `ai_subgoal`
- candidate events 成功带上 `condition_features`
- 因为该 session 没有玩家反馈，所以 Hu 样本为 0，符合预期

## 6. 当前进度与下一步

已用 `20260722_211635` 的 12 条真实语言反馈完成一次 schema-v2 回放：

- 修正 soup 嵌套结构读取，`recipe_needs_*` 不再长期误报。
- 修正 `human_trying_to_pass` 的实时误报。
- 增加相对距离、最后原料、staged counter 物体和劳动分工条件。
- 新增 `AI_missed_useful_counter_object`。
- 新增 `AI_missed_labor_division_opportunity`。
- 归因候选严格截止到反馈 timestep，不使用未来轨迹。
- 规则 baseline 生成 12 条 attribution、7 条保守 Hu pairwise 样本，其余模糊反馈进入 Review。

接下来优先级：

1. 在 Review 窗口审核这 12 条反馈，特别是 `yes`、`good move`、`good pick` 和路径建议。
2. 使用同一 schema 重新运行 DeepSeek 归因，并与人工决定比较。
3. 让 `review_decisions.jsonl` 成为 Hu 训练集的优先数据源。
4. 在 probe states 上训练并检查 Hu-v0 的 subgoal 排序变化。
5. 经 Review 证据支持后，再决定是否新增“跟随更快路径”和“主动让路”事件。
