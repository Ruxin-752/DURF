# Day1 Pilot 数据流说明：从游戏日志到 LLM 归因

更新时间：2026-07-02  
当前固定入口：

```powershell
python -m durf.group_a.play_with_baseline `
  --agent RllibRingHalfTaskStableTopLeft `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left
```

Day1 的目标不是训练 `H_u`，也不是正式评估 LLM 准确率，而是确认 pilot 数据入口是否稳定：

1. 人类玩家能和固定 PPO agent 玩一局。
2. 游戏过程能稳定记录轨迹、聊天、暂停。
3. 原始 CSV 能转换成结构化 JSONL。
4. 程序能从轨迹中检测 candidate events。
5. 规则归因和 LLM 归因都能生成可审计结果。
6. 这些结果能暴露 Day2 需要扩展的 schema。

Day1 有效 session：

```text
outputs/human_ai_sessions/20260701_210024
outputs/human_ai_sessions/20260701_214629
```

其中 `20260701_214629` 的数据概况：

```text
trajectory.csv               402 steps
chat_messages.csv             11 user feedback messages
trajectory.jsonl             402 records
feedback_events.jsonl         11 records
candidate_events.jsonl        17 records
attribution_preview.jsonl     11 records
llm_attribution_audit.jsonl   11 records
```

## 1. 总体数据流

完整流程如下：

```text
Pygame 游戏界面
  |
  |  durf/group_a/play_with_baseline.py
  v
原始 session CSV
  - trajectory.csv
  - chat_messages.csv
  - pause_events.csv
  |
  |  durf/feedback_attribution/session_converter.py
  v
结构化 JSONL
  - trajectory.jsonl
  - feedback_events.jsonl
  |
  |  durf/feedback_attribution/event_detectors.py
  v
候选事件
  - candidate_events.jsonl
  |
  |  durf/feedback_attribution/sample_builder.py
  |  或 durf/feedback_attribution/llm_attributor.py
  v
反馈归因结果
  - attribution_preview.jsonl
  - llm_attribution_audit.jsonl
```

一句话理解：

```text
trajectory 记录“游戏发生了什么”
feedback_events 记录“玩家说了什么”
candidate_events 记录“程序从轨迹里看见了哪些可能被评价的事件”
attribution_preview 记录“这句话被归因到哪个事件/是否需要澄清”
llm_attribution_audit 记录“LLM 是怎么收到输入、怎么输出判断的”
```

## 2. 原始日志文件

原始日志由 `durf/group_a/play_with_baseline.py` 在游戏过程中直接生成，位置为：

```text
outputs/human_ai_sessions/<session_id>/
```

### 2.1 `trajectory.csv`

来源：

```text
play_with_baseline.py
```

生成时机：

```text
每个 environment timestep 写入一行
```

它回答的问题：

```text
这一帧之前厨房是什么状态？
人类做了什么？
AI 做了什么？
动作执行后厨房变成什么状态？
环境奖励是多少？
```

核心字段：

| 字段 | 含义 |
|---|---|
| `timestamp_utc` | 这一行日志写入的 UTC 时间 |
| `episode` | 当前第几局 episode |
| `episode_step` | 当前 episode 内第几步 |
| `total_step` | 当前 session 的总步数 |
| `layout` | 当前地图 |
| `ai_action` / `ai_action_name` | AI 动作编号和动作名 |
| `human_action` / `human_action_name` | 人类动作编号和动作名 |
| `environment_reward` | 当前一步环境奖励 |
| `episode_reward` | 当前 episode 累积奖励 |
| `predict_ms` | AI 预测动作耗时 |
| `environment_step_ms` | 环境 step 耗时 |
| `done` | episode 是否结束 |
| `state_before_json` | 动作执行前的厨房状态 |
| `state_after_json` | 动作执行后的厨房状态 |

`state_before_json` 和 `state_after_json` 是后续事件检测的关键。它们用于比较：

```text
动作前：AI 拿着 onion，锅里有 2 个 tomato
动作：AI interact
动作后：AI 手空了，锅开始 cooking
```

因此我们可以推断：

```text
AI 成功把 onion 放进锅里
```

### 2.2 `chat_messages.csv`

来源：

```text
play_with_baseline.py 的 Chat 窗口
```

生成时机：

```text
玩家发送一句话时写一行；DeepSeek 回复时也写一行
```

它回答的问题：

```text
玩家在游戏的哪个 timestep 说了什么？
DeepSeek 又回复了什么？
```

核心字段：

| 字段 | 含义 |
|---|---|
| `timestamp_utc` | 聊天发生时间 |
| `episode` | 聊天发生在第几局 |
| `episode_step` | 聊天发生在 episode 内第几步 |
| `total_step` | 聊天发生在 session 总第几步 |
| `layout` | 当前地图 |
| `role` | `user` 或 `assistant` |
| `content` | 聊天内容 |

Day1 中典型玩家反馈：

```text
you did great in putting tomato into the pot!
don't wander around that corner, go get ingredients
don't stuck me
when ingredients are already being cooked and I didn't get a plate, you should go and get it
you should get some ingredient when passing by
when I "push" you, you should go forward
good! you know how to deliver food!
```

注意：`chat_messages.csv` 同时包含玩家和 assistant。后续 `feedback_events.jsonl` 只保留玩家反馈。

### 2.3 `pause_events.csv`

来源：

```text
play_with_baseline.py
```

生成时机：

```text
玩家暂停/恢复游戏时写入
```

它回答的问题：

```text
玩家什么时候暂停？
玩家是否在暂停状态下输入反馈？
暂停是否影响反馈延迟？
```

核心字段：

| 字段 | 含义 |
|---|---|
| `timestamp_utc` | 暂停/恢复时间 |
| `event` | `paused` 或 `resumed` |
| `episode` | 所属 episode |
| `episode_step` | episode 内 step |
| `total_step` | session 总 step |
| `layout` | 当前地图 |
| `pygame_ticks` | Pygame 内部时间 |

Day1 暂时只用它做完整性检查。正式实验中，它可以用于分析：

```text
玩家是否需要暂停才能给出准确反馈
暂停是否降低操作负担
暂停反馈是否比实时反馈更准确
```

## 3. 转换后的 JSONL 文件

转换入口：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id>
```

或者只做 CSV -> JSONL：

```powershell
python -m durf.feedback_attribution.session_converter `
  --session outputs\human_ai_sessions\<session_id>
```

### 3.1 `trajectory.jsonl`

来源：

```text
trajectory.csv
```

生成脚本：

```text
durf/feedback_attribution/session_converter.py
```

它回答的问题：

```text
把每一步 trajectory 从 CSV 的字符串格式转成程序能读取的 JSON 对象。
```

一行概念示例：

```json
{
  "record_type": "trajectory_step",
  "episode": 1,
  "episode_step": 22,
  "total_step": 22,
  "layout": "ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left",
  "ai_action_name": "east",
  "human_action_name": "stay",
  "environment_reward": 0.0,
  "episode_reward": 0.0,
  "done": false,
  "state_before": {
    "players": "...",
    "objects": "...",
    "pot_states": "..."
  },
  "state_after": {
    "players": "...",
    "objects": "...",
    "pot_states": "..."
  }
}
```

为什么需要它：

```text
CSV 方便人用 Excel 看；
JSONL 方便程序逐行读取复杂结构。
```

后续谁用它：

```text
event_detectors.py 用它检测 candidate events
llm_attributor.py 用它构造 recent_trajectory 摘要
```

### 3.2 `feedback_events.jsonl`

来源：

```text
chat_messages.csv 中 role=user 的行
```

生成脚本：

```text
durf/feedback_attribution/session_converter.py
```

它回答的问题：

```text
玩家在什么时候说了哪条反馈？
```

一行真实结构示例：

```json
{
  "episode": 1,
  "episode_step": 80,
  "total_step": 80,
  "feedback_text": "don't wander around that corner, go get ingredients",
  "feedback_value": null,
  "record_type": "feedback_event",
  "role": "human_language",
  "source": "chat_messages.csv",
  "timestamp_utc": "2026-07-01T13:49:14.006+00:00",
  "extra": {
    "layout": "ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left"
  }
}
```

注意：

```text
feedback_events.jsonl 还没有做归因。
它只知道“玩家说了什么”，不知道“玩家在评价哪个 AI 行为”。
```

## 4. 候选事件文件

### 4.1 `candidate_events.jsonl`

来源：

```text
trajectory.jsonl
```

生成脚本：

```text
durf/feedback_attribution/event_detectors.py
```

它回答的问题：

```text
在玩家反馈附近，程序从轨迹中检测到了哪些可能被评价的 AI 行为事件？
```

当前 Day1 主要检测到两类：

```text
AI_blocked_human_path
AI_failed_to_prepare_ingredient_while_waiting
```

Day1 session `20260701_214629` 中检测结果：

```text
AI_blocked_human_path: 15
AI_failed_to_prepare_ingredient_while_waiting: 2
```

一行概念示例：

```json
{
  "record_type": "candidate_event",
  "event_type": "AI_blocked_human_path",
  "start_timestep": 126,
  "end_timestep": 126,
  "confidence": 0.65,
  "severity": 0.75,
  "evidence": {
    "ai_before": [6, 4],
    "ai_after": [6, 4],
    "human_before": [5, 4],
    "human_after": [5, 4],
    "human_attempted_target": [6, 4],
    "human_action": "east",
    "target_was_ai": true,
    "narrow_corridor": true
  },
  "missing_required_facts": []
}
```

重要理解：

```text
candidate event 不是玩家反馈。
candidate event 是程序从轨迹里检测出的“可能值得评价的行为”。
```

程序不是自然知道“什么叫堵路”。我们提前定义了事件 schema 和检测规则，例如：

```text
如果 human 尝试移动到的位置正好被 AI 占着，
且该位置属于狭窄通道，
则生成 AI_blocked_human_path。
```

Day1 暴露的问题：

```text
真实玩家反馈远多于当前两类 candidate event。
例如：
- 成功放食材
- 成功送餐
- 路过食材不拿
- 在角落游走
- 面对墙反复调整方向
- 被玩家“推”时不移动
- 锅在煮而玩家没盘子时 AI 应去拿盘
```

这些是 Day2 扩展 schema 的依据。

### 4.2 Event / Condition / Preference 固定范式

为了防止后续代码越改越散，Day1 之后先把三个核心概念固定下来：

```text
event       = 程序从轨迹里检测到的事实事件：刚才发生了什么？
condition   = 这个评价成立时的环境前提：当时处在什么局面？
preference  = 从玩家反馈中总结出的偏好方向：用户希望以后怎么改变？
```

三者的关系不是并列乱塞字段，而是：

```text
condition + event + human feedback
        -> preference label
        -> H_u event-level training sample
```

也就是说，第一版 `H_u` 不直接学习每一步底层动作，而是学习事件级偏好：

```text
在某种 condition 下，某个 event 对这个用户来说是好还是坏。
```

例如：

```json
{
  "event_type": "AI_blocked_human_path",
  "condition": {
    "human_trying_to_pass": true,
    "narrow_corridor": true,
    "ai_on_human_path": true
  },
  "preference": {
    "key": "path_coordination",
    "direction": "avoid_blocking_human_path",
    "polarity": "negative"
  }
}
```

#### 4.2.1 当前已经固定的 event

当前代码里真正由程序检测的 event 只有 3 类。它们定义在：

```text
durf/feedback_attribution/event_detectors.py
```

| event_type | 事实含义 | 程序检测规则 | 当前用途 |
|---|---|---|---|
| `AI_blocked_human_path` | AI 挡住人类路径 | 人类尝试向某方向移动，但位置没变；人类目标格正好被 AI 占据 | 路径协作、堵路反馈 |
| `AI_ignored_ready_or_nearly_ready_pot` | 锅 ready 后 AI 没及时处理 | 存在 ready pot；AI 有能力处理锅；连续若干步没有 `interact` | 出餐时机、锅处理反馈 |
| `AI_failed_to_prepare_ingredient_while_waiting` | 等锅时 AI 没有备菜 | 锅在 cooking；人类拿着盘子等待；AI 空手且连续若干步没有拿食材 | 等待期间任务推进反馈 |

注意：

```text
event 是事实层，不判断玩家是否喜欢。
event 只回答“轨迹里有没有发生这类可评价行为”。
```

例如 `AI_blocked_human_path` 的证据字段包括：

```json
{
  "human_action": "east",
  "human_before": [3, 2],
  "human_after": [3, 2],
  "human_attempted_target": [4, 2],
  "ai_before": [4, 2],
  "ai_after": [4, 2],
  "target_was_ai": true,
  "narrow_corridor": true,
  "walkable_neighbor_count": 2
}
```

#### 4.2.2 当前 preference 的定义方式

当前规则版 preference 是从 `event_type + polarity` 推出来的轻量标签，定义在：

```text
durf/feedback_attribution/sample_builder.py
```

当前映射为：

| event_type | negative / unknown 时的 preference |
|---|---|
| `AI_blocked_human_path` | `avoid_blocking_human_path` |
| `AI_ignored_ready_or_nearly_ready_pot` | `handle_ready_pot_when_possible` |
| `AI_failed_to_prepare_ingredient_while_waiting` | `prepare_ingredients_during_cooking_wait` |

这里的 `preference` 不是另一个事实事件，而是学习目标的名字。

可以把它理解为“压缩包名”：

```text
event_type 说明刚才发生了什么；
polarity 说明玩家觉得好还是坏；
preference 说明这条反馈应该归入哪个可复用的偏好主题/方向。
```

例如：

```text
event_type = AI_blocked_human_path
polarity = negative
preference = avoid_blocking_human_path
```

表示：

```text
这次发生了 AI 堵路，玩家负面评价它，因此推断出用户偏好是“避免堵路”。
```

后续建议不要让 `preference_key` 无限增殖成自由文本。第一版更稳的做法是把它收敛成少量可解释主题，例如：

```text
path_coordination
task_progress_when_waiting
pot_and_plate_timing
ingredient_preparation
avoid_redundant_motion
human_workspace_respect
```

其中更具体的行为方向可以放在 `direction` 字段中，例如：

```json
{
  "key": "path_coordination",
  "direction": "avoid_blocking_human_path"
}
```

这样做的原因是：`preference_key` 太细会导致训练样本被切得太碎，`H_u` 很难稳定学习。

#### 4.2.3 condition 暂定设计

当前代码还没有正式独立的 `condition` schema。现在的近似版本是：

```text
key_conditions_from_event(event)
```

也就是从 event 的 `evidence` 中抽取关键字段，写入 `attribution_result.key_conditions`。

例如 `AI_blocked_human_path` 当前会抽出：

```json
{
  "human_attempted_target": [4, 2],
  "ai_before": [4, 2],
  "ai_after": [4, 2],
  "narrow_corridor": true
}
```

这还不是理想的 condition，因为它更像 evidence 摘要，而不是可训练的布尔/类别特征。下一版建议把 condition 拆成三层：

```json
{
  "language_condition": "when pot is cooking and human has no plate",
  "matched_state_facts": {
    "pot_is_cooking": true,
    "human_has_plate": false,
    "ai_has_plate": false
  },
  "event_condition_features": {
    "ai_idle_or_wandering": true,
    "ai_near_plate_dispenser": true
  },
  "unmatched_conditions": []
}
```

其中：

| 字段 | 含义 | 来源 |
|---|---|---|
| `language_condition` | 玩家语言里说出的条件 | LLM 从自然语言提取 |
| `matched_state_facts` | 程序能在轨迹中验证的状态事实 | `trajectory.jsonl` / `state_facts` |
| `event_condition_features` | 从事件证据中整理出的可训练特征 | `candidate_event.evidence` |
| `unmatched_conditions` | 玩家说了但当前日志无法验证的条件 | LLM + 程序校验 |

第一版可优先支持以下 condition features：

```text
human_trying_to_pass
ai_on_human_path
narrow_corridor
pot_is_cooking
pot_ready_or_nearly_ready
human_has_plate
ai_has_plate
ai_has_free_hand
ai_near_pot
ai_near_plate_dispenser
ai_near_ingredient_dispenser
ai_idle_or_wandering
human_waiting_with_dish
```

这些字段应尽量来自程序事实，而不是让 LLM 凭空判断。LLM 的职责是解析语言指代，例如：

```text
"when I don't have a plate" -> human_has_plate = false
"don't block me" -> path_coordination / AI_blocked_human_path
```

程序的职责是验证这些条件是否真的能在轨迹中找到证据。

#### 4.2.4 未来 `H_u` 训练样本范式

Day1 还没有生成正式 `H_u` 数据集。下一步建议从当前文件生成：

```text
trajectory.jsonl
feedback_events.jsonl
candidate_events.jsonl
attribution_preview.jsonl
llm_attribution_audit.jsonl
        -> hu_event_dataset.jsonl
```

`hu_event_dataset.jsonl` 的目标范式：

```json
{
  "record_type": "hu_event_preference_sample",
  "sample_id": "20260701_214629:feedback:80",
  "user_id": "PILOT01",
  "layout": "ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left",
  "feedback": {
    "text": "don't stuck me",
    "type": "descriptive",
    "total_step": 126
  },
  "event": {
    "type": "AI_blocked_human_path",
    "time_window": [126, 126],
    "evidence": {
      "human_attempted_target": [6, 4],
      "ai_before": [6, 4],
      "ai_after": [6, 4],
      "narrow_corridor": true
    }
  },
  "condition": {
    "language_condition": null,
    "matched_state_facts": {
      "human_trying_to_pass": true,
      "ai_on_human_path": true,
      "narrow_corridor": true
    },
    "unmatched_conditions": []
  },
  "preference": {
    "key": "path_coordination",
    "direction": "avoid_blocking_human_path",
    "polarity": "negative",
    "score": -1
  },
  "attribution": {
    "method": "llm",
    "confidence": 0.8,
    "needs_clarification": false
  }
}
```

训练时不建议把所有原始字段都直接喂给 `H_u`。建议分层处理：

| 层级 | 是否保存 | 是否直接喂给 `H_u` |
|---|---|---|
| 原始 trajectory/state JSON | 保存，用于审计和复查 | 不直接喂 |
| candidate event evidence | 保存，用于解释 event 为什么成立 | 只抽关键特征 |
| condition features | 保存并规范化 | 直接喂 |
| preference key/direction | 保存为受控标签 | 直接喂，作为类别特征 |
| feedback 原文 | 保存，用于审计/LLM 重跑 | 第一版不直接喂 |

因此第一版 `H_u` 的核心输入应压缩为：

```json
{
  "event_type": "AI_blocked_human_path",
  "preference_key": "path_coordination",
  "preference_direction": "avoid_blocking_human_path",
  "condition_features": {
    "human_trying_to_pass": true,
    "ai_on_human_path": true,
    "narrow_corridor": true
  },
  "event_features": {
    "duration_steps": 1,
    "severity": 0.8,
    "confidence": 0.8
  },
  "user_id": "PILOT01",
  "label": -1
}
```

这条范式先固定一个原则：

```text
完整信息要保存，训练输入要压缩。
event 负责事实，condition 负责适用场景，preference 负责学习目标。
```

## 5. 规则归因与 LLM 归因结果

### 5.1 `attribution_preview.jsonl`

来源：

```text
feedback_events.jsonl
candidate_events.jsonl
trajectory.jsonl
```

生成脚本：

```text
durf/feedback_attribution/sample_builder.py
```

如果加了 `--use-llm`，则会由：

```text
durf/feedback_attribution/llm_attributor.py
```

进一步覆盖/更新为 LLM semantic attribution 结果。

它回答的问题：

```text
这条玩家反馈最可能对应哪个 candidate event？
如果不能归因，是否需要澄清？
LLM 是否建议新增 schema？
```

运行规则版：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id>
```

运行 LLM 版：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --use-llm
```

一行真实 LLM attribution 示例：

```json
{
  "record_type": "attribution_result",
  "feedback_event_id": "chat_messages.csv:22:2026-07-01T13:47:10.342+00:00",
  "feedback_type": "evaluative",
  "polarity": "positive",
  "target_event": null,
  "target_time_window": [0, 0],
  "confidence": 0.0,
  "needs_clarification": false,
  "clarification_question": null,
  "preference": null,
  "key_conditions": {},
  "proposed_schema_update": [
    "add_event_type: put_ingredient_into_pot"
  ],
  "notes": "LLM semantic attribution. Rationale: Feedback praises putting tomato into pot but no candidate event captures that action."
}
```

这个例子说明：

```text
玩家说：you did great in putting tomato into the pot!
现有 candidate events 没有“成功放食材”事件。
LLM 没有硬归因到 blocking，而是提出新增 schema。
```

这是 Day1 的成功点。

### 5.2 `llm_attribution_audit.jsonl`

来源：

```text
llm_attributor.py
```

只有运行 `--use-llm` 时才生成。

它回答的问题：

```text
LLM 到底看到了什么输入？
LLM 原始输出是什么？
JSON parse 后的结果是什么？
这次调用是否成功？
```

一行结构：

```json
{
  "feedback_event_id": "chat_messages.csv:80:2026-07-01T13:49:14.006+00:00",
  "messages": [
    {
      "role": "system",
      "content": "You are an attribution module..."
    },
    {
      "role": "user",
      "content": "{... feedback + candidate_events + recent_trajectory ...}"
    }
  ],
  "raw_response": "{... LLM 原始 JSON 字符串 ...}",
  "parsed_response": {
    "target_event": null,
    "needs_clarification": true,
    "clarification_question": "Which corner are you referring to, and what specific ingredient should the AI go get?",
    "confidence": 0.2,
    "feedback_type": "imperative",
    "polarity": "negative",
    "preference": "go_get_ingredients",
    "key_conditions": {},
    "proposed_schema_update": [],
    "rationale": "No candidate events match the feedback about wandering and fetching ingredients.",
    "target_time_window": [0, 0]
  },
  "status": "ok",
  "used_candidate_event_count": 0
}
```

注意：

```text
llm_attribution_audit.jsonl 的结果不在顶层字段。
真正的 LLM 输出在 parsed_response 里面。
```

它和 `attribution_preview.jsonl` 的区别：

| 文件 | 用途 |
|---|---|
| `attribution_preview.jsonl` | 给后续训练/人工检查看的简洁归因结果 |
| `llm_attribution_audit.jsonl` | 给研究者审计 LLM 输入输出过程 |

## 6. 当前文件之间的串联关系

以一句反馈为例：

```text
玩家反馈：
"don't wander around that corner, go get ingredients"
```

它首先出现在：

```text
chat_messages.csv
```

然后被转换为：

```text
feedback_events.jsonl
```

其中保留：

```json
{
  "feedback_text": "don't wander around that corner, go get ingredients",
  "total_step": 80
}
```

系统根据 `total_step=80` 回看近期 trajectory：

```text
trajectory.jsonl 中 total_step 55 到 80 附近的状态和动作
```

程序检测 candidate events：

```text
candidate_events.jsonl
```

如果附近没有匹配事件，规则版会输出：

```json
{
  "target_event": null,
  "needs_clarification": true,
  "notes": "No candidate event found near this feedback."
}
```

LLM 版会进一步判断：

```json
{
  "target_event": null,
  "needs_clarification": true,
  "preference": "go_get_ingredients",
  "rationale": "No candidate events match the feedback about wandering and fetching ingredients."
}
```

所以这条反馈在 Day1 的结论是：

```text
这不是 attribution 失败，而是 schema 覆盖不足。
Day2 应新增 AI_wandered_in_low_value_corner / AI_failed_to_fetch_available_ingredient 等事件。
```

## 7. Day1 LLM 归因初步结果

### 7.1 `20260701_214629`

```text
LLM calls: 11
status=ok: 11

feedback_type:
- imperative: 6
- evaluative: 3
- unknown: 2

polarity:
- negative: 7
- positive: 3
- neutral: 1

needs_clarification:
- true: 6
- false: 5

target_event:
- None: 8
- AI_blocked_human_path: 3

schema update count: 2
```

LLM 建议的 schema update：

```text
add_event_type: put_ingredient_into_pot
AI_facing_wall_idle
```

### 7.2 `20260701_210024`

```text
LLM calls: 20
status=ok: 20

feedback_type:
- imperative: 7
- unknown: 5
- descriptive: 4
- evaluative: 4

polarity:
- neutral: 7
- negative: 5
- unknown: 4
- positive: 4

needs_clarification:
- true: 12
- false: 8

target_event:
- None: 16
- AI_blocked_human_path: 4

schema update count: 1
```

LLM 建议的 schema update：

```text
ingredient_corner_zone
```

## 8. Day1 结论

Day1 证明了：

```text
1. 固定地图和固定 agent 可以稳定运行。
2. 玩家自然语言反馈可以被记录到 chat_messages.csv。
3. 轨迹可以被完整记录到 trajectory.csv。
4. 原始日志可以转换成 trajectory.jsonl 和 feedback_events.jsonl。
5. 现有规则可以生成 candidate_events.jsonl。
6. 规则归因和 LLM 归因都可以生成 attribution_preview.jsonl。
7. LLM 调用过程可以被 llm_attribution_audit.jsonl 审计。
```

Day1 也暴露了：

```text
1. 当前 candidate event schema 太窄。
2. 真实玩家反馈包含大量成功事件、条件化偏好、机会捕捉和路径低效问题。
3. LLM 能发现部分 schema 缺口，但有时仍会把不完全匹配的反馈归到 AI_blocked_human_path。
4. Day2 需要扩展 candidate event schema，并建立人工标注表。
```

## 9. Day2 的输入

Day2 不需要重新设计数据流，而是在 Day1 数据上做：

```text
1. 人工标注 feedback_events.jsonl。
2. 对比 attribution_preview.jsonl 与人工 gold label。
3. 根据无法覆盖的反馈扩展 candidate event schema。
4. 增加正向事件：
   - AI_successfully_put_ingredient_into_pot
   - AI_successfully_delivered_soup
   - AI_successfully_rerouted_when_blocked
5. 增加负向事件：
   - AI_wandered_in_low_value_corner
   - AI_passed_ingredient_without_pickup
   - AI_failed_to_get_plate_when_pot_cooking
   - AI_facing_wall_idle
   - AI_failed_to_move_when_human_pushes
   - AI_failed_to_fetch_complementary_ingredient
```

Day1 的产物就是 Day2 的材料。
