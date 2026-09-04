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

**“当时”指决策前的状态（2026-09-03 修正）。** 轨迹每一步有两份快照：`state_before_json`
（AI 做决策时看到的）和 `state_after_json`（动作执行后）。`extract_condition_features(step)`
读的是 `state_facts`，而 `session_converter` 把它设成了 **after** 快照——这是"发生了什么"
的视角，适合事件检测；但 Hu 标签说的是"在 X 条件下应选 A 而非 B"，X 必须是做出 B 那个
决策时的状态，也就是 **before** 快照，和运行时 `ai_condition_features_json` 记录的完全一致。
之前二者错开一步，在状态切换点直接自相矛盾：4554 条仿真样本里 2666 条（59%）是
`GET_DISH/PICKUP_SOUP > PUT_DOWN_OBJECT` 且条件写着 `ai_empty_handed=True`——空手时根本没有
PUT_DOWN_OBJECT 这个候选。现在所有进入标签的条件都走 `decision_condition_features(step)`：
优先取运行时记录的 `ai_condition_features`（Hu 打分时实际看到的那份），缺的字段再从
`extra.state_before` 重算；旧 session 没有 before 快照的，退回 after 并用
`condition_state_source="state_after_fallback"` 标出，可过滤。事件检测器内部判断"发生了
什么"仍然用 after 快照，不受影响。

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

## 6. 标签质量审计（2026-09-03）

对 4 人格 × 20 seed 仿真数据的 4554 条 pairwise 样本和冻结的 `outputs/hu_general` 做了一次
逐条审计，发现三个定义层面的问题，均已修正：

1. **条件错位一步**（见 Step 2）。修法：`decision_condition_features`。
2. **两个任务域事件检测器系统性误报。** `AI_ignored_ready_or_nearly_ready_pot` 把"锅在煮、
   AI 正在为下一轮取番茄"当成"无视锅"（抽样 20 个事件 20 个都是这种），仿真人类模板随之
   说出"the pot is ready, go get a dish"这句事实上不成立的话；
   `AI_failed_to_prepare_ingredient_while_waiting` 把"正在走向出料口"当成"没在备料"，还会在
   运行时根本没有 GET_TOMATO/GET_ONION 候选（下一轮原料已全部在台面上）时要求 AI 去取料。
   由此产生的标签让冻结 `hu_general` 的任务头学到 GET_TOMATO −0.19、GET_ONION −0.16、
   PUT_DOWN_OBJECT −0.21、GET_DISH/PICKUP_SOUP +0.24——即"别备料、别放下"，纯属检测器伪影。
   修法：`event_detectors.ai_ignoring_pot`（锅 ready 时只要 AI 没在做 GET_DISH/PICKUP_SOUP/
   WAIT_NEAR_POT 就算无视；锅只是 cooking 时只有空手 WAIT 且人类没拿盘子才算）、
   `prep_option_available`（运行时候选集里没有取料项就不算错过机会）；`SimHuman` 的
   streak 逻辑同步镜像，并把 AI 的 subgoal 和候选集传给它。
3. **标签词表 ≠ 运行时词表。** `GET_USEFUL_INGREDIENT` 有 201 条 preferred 标签、bias +0.18，
   但候选生成器只产出 GET_TOMATO/GET_ONION，运行时对未知名字 `except KeyError: hu_score=0`
   静默吞掉，这条偏好永远到不了决策。修法：它从 `TASK_HU_SUBGOALS` 移除、降为归因层名字
   （`ATTRIBUTION_ONLY_TASK_SUBGOALS`），`hu_dataset_builder` 在成对之前用该决策步真实的
   候选集把它解析成 GET_TOMATO/GET_ONION（`resolve_useful_ingredient`），解析不了的记入
   provenance 的 `unresolved_subgoals` 并丢弃；运行时遇到模型打不了分的候选会在 stderr 警告
   一次（`warn_unknown_runtime_subgoal`）。

**修正后重跑同一批 80 个仿真 session（`outputs/hu_general_v2_regen/`，未覆盖冻结模型）的结果
必须如实记录：任务域样本从 3143 条降到 0 条。** 400 条仿真反馈全部是协调域（让路/坚持），
没有一条任务域反馈——因为 H0 底座根本不犯仿真模板所抱怨的那两种错（锅好了它一定去拿盘子；
空手等待的 560 个抽样步里 100% 是"下一轮原料已经全在台面上、无料可取"）。这意味着：
冻结 `hu_general` 的任务头**整体是伪影**，不能再作为任务域先验使用；仿真人格目前只能为
协调域提供先验（重训后 YIELD/CONTINUE 的 global bias 从 ±0.49 回到 ±0.02，条件权重仍在）。
任务域的偏好学习要么等真人数据，要么给仿真人格设计真正的"偏好式"任务反馈（例如在
GET_DISH 与 GET_TOMATO 都可行时说"盘子我来拿"），而不是"纠错式"反馈。

## 7. 仿真人格的任务域偏好反馈（2026-09-03，接 §6）

§6 的结论是"纠错式"模板在这个底座上产不出任务域标签。补的办法不是放宽检测器，而是让
仿真参与者说**偏好**而不是**纠错**：在两个候选都可行、任务分差 ≤10 的决策点上表达"这件事
你做还是我做"。任务分差 ≤10 正是 ε-约束规则允许偏好说了算的那条带（§3.3），所以这类
陈述在定义上不是在指出错误。

实现（`durf/group_a/sim_session.py`）：

- `_preference_opportunity(candidates)` 只认两种机会，都要求前两名候选分差 ≤10：
  `division_of_labour`（锅在煮、AI 空手：GET_DISH 60 vs GET_TOMATO/GET_ONION 50）和
  `hold_plate`（无锅在煮、手持盘子：PUT_DOWN_OBJECT 20 vs WAIT_NEAR_POT 10）。
  手持多余原料时 PUT_DOWN_OBJECT 领先 50 分，不算机会——那是任务退步，不是口味。
- `PERSONA_TASK_PREFERENCE` 给每个人格加了第二条、与协调性格独立的特质：
  cooperative/polite = `prep_first`（"盘子我来拿，你去备下一份"），selfish/lenient =
  `dish_first`（"你去拿盘子，原料我管"）。`--sim-task-preference` 可以单独覆盖。
  `prep_first` 与 H0 默认相反（H0 选 GET_DISH），`dish_first` 与 H0 一致，天然构成
  处理组/对照组。
- 每个 episode 每种机会最多说 2 次（`PREFERENCE_MAX_PER_EPISODE`），不是每步都刷。
- 措辞由 `infer_explicit_preference_from_text` 解析成成对标签；模板与解析器由
  `testing/feedback_attribution_test.py::test_every_template_parses_to_the_pair_it_is_meant_to_express`
  钉在一起，改一边不改另一边会红。
- 归因侧配套改了一处：事件选择本质是关键词匹配，而一句"你拿 X、我拿 Y"是明确的两侧
  陈述。因此当事件给出的配对**不完整**或**属于另一个决策域**时，以句子为准，被顶掉的事件
  记进 `preference_overridden_event` 供审计，条件取反馈发生的那一步。两者同域且事件配对
  完整时仍以事件为准（事件知道 AI 当时具体在做 GET_TOMATO 还是 GET_ONION，比句子里泛指的
  "ingredient"更具体）。

重跑 80 个 session（`outputs/hu_general_v3_pref/`）的结果：

| | 数量 |
|---|---|
| pairwise 样本总数 | 560 |
| 协调域 | 400（其中不变量 160 进 Hu_general） |
| 任务域 | 160（GET_TOMATO>GET_DISH 80 / GET_DISH>GET_TOMATO 80） |

任务域 160 条全部是**人格冲突**样本（audit 的 `conflict_details` 里明确记着四个人格的方向），
因此按协议不进 Hu_general，而是个体差异信号。**Hu_general 的任务头仍然是空的**，这次是
正确的空：仿真里不存在"所有人都同意"的任务偏好。`hold_plate` 那种可能是不变量的机会在
128000 步里一次都没出现（这个布局下 AI 从不会在没有锅在煮时手持盘子），所以没有支撑。

### 7.1 任务域行为确实被偏好改变了（第一次）

按人格各训一个 HierarchicalHu（`outputs/hu_general_v3_pref/personas/<persona>/`，用 seed
0–15 训、20–23 留出），再用留出 seed 跑仿真：

| 人格 | 特质 | 运行 | 选 GET_DISH | 选备料 | preference_override | 每局分数 |
|---|---|---|---|---|---|---|
| cooperative | prep_first | H0（容忍带 0） | 18 | 0 | 0 | 60 |
| cooperative | prep_first | 自己的模型，容忍带 10 | 0 | 26 | 26 | 20 |
| polite | prep_first | H0 | 18 | 0 | 0 | 60 |
| polite | prep_first | 自己的模型，容忍带 10 | 0 | 26 | 26 | 20 |
| selfish | dish_first | 自己的模型，容忍带 10 | 18 | 0 | 0 | 60 |
| lenient | dish_first | 自己的模型，容忍带 10 | 18 | 0 | 0 | 60 |

翻转点精确落在容忍带 = 分差：容忍带 9 完全不变，10 全翻。交叉对照（cooperative 的对局用
selfish 的模型跑）也不变，说明变化来自偏好本身而不是仿真随机数漂移。三个留出 seed 全部
复现。

**但那个 60→20 不是偏好的代价，是一个被顺带暴露出来的死锁。** 逐帧看 cooperative/seed20
的轨迹：翻转之后局面走到"AI 手持盘子、锅已好、要去锅边"，而人类在 1 格宽的上走廊里
来回，两人相隔 2 格互相镜像地 east/west 摆动，692 步没有任何进展。协调层从未介入
（`coordination_decision` 全程为空）——因为两人从不真的争夺同一格，`path_conflict_type`
不触发；`stalled_escape` 也不管——它只处理"空手 + WAIT + 无动作"。这是机制里一个既有的
鲁棒性缺口，不是这次改动引入的，只是偏好把对局推进了这个状态。**在修掉它之前，任何
容忍带标定量到的都是这个死锁，而不是偏好的真实代价。**

### 7.2 已知的两个局限（写清楚，别在论文里当成人类证据）

1. **这些标签验证的是机制，不是人类偏好。** 模板是我们自己写的，人格特质是我们自己
   指派的，然后 Hu 学到了我们指派的东西——这条闭环只能证明"从一句话到行为改变"的管道
   通了，不能证明任何关于真人偏好的事实。
2. **仿真人格说的和做的不一致。** 说"盘子我来拿"的人格，它自己的动作仍然由同一个规则
   教师（`rule_teacher_candidates(state, mp, 1)`）决定，并不会真的去拿盘子。真人说了会做，
   所以真人实验里honor这条偏好的代价大概率比仿真里小。要修就得让 `SimHuman.choose_action`
   也按自己的特质在同样的 ≤10 分差带内选边。

## 8. 当前进度与下一步

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
