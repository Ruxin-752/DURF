# 归因 Review 窗口与数据接口

这个模块的目标不是替代模型，而是把每局游戏结束后的“人工审核”工程化。它用于检查 LLM 归因、候选事件、条件字段和 Hu 训练标签是否靠谱，避免把草率 schema 或语义误解直接喂给 Hu。

## 什么时候用

每跑完一局并完成离线归因后使用：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --use-llm
```

然后打开 review 窗口：

```powershell
python -m durf.feedback_attribution.review_session `
  --session outputs\human_ai_sessions\<session_id>
```

如果只想生成机器可读的 review item，不打开窗口：

```powershell
python -m durf.feedback_attribution.review_session `
  --session outputs\human_ai_sessions\<session_id> `
  --export-only
```

## 输入文件

Review 窗口会读取同一个 session 文件夹下的这些文件：

- `trajectory.jsonl`
  - 每一步游戏状态和动作。
  - 用来显示反馈发生前的一小段轨迹。
  - 里面包括 AI/人类位置、手持物、锅状态、地图对象、AI 当前 subgoal、候选 subgoal 分数等。

- `feedback_events.jsonl`
  - 玩家输入的自然语言反馈。
  - 每条反馈有 `feedback_event_id`、文本、发生 timestep、episode 信息。

- `candidate_events.jsonl`
  - 程序从轨迹里检测出的候选事件。
  - 它回答“附近有哪些客观发生过、可能被评价的 AI 行为”。

- `attribution_preview.jsonl`
  - 规则 baseline 或 LLM 给出的初步归因结果。
  - 它回答“这句话最可能指向哪个事件、哪段时间、哪些条件、哪些 subgoal 偏好”。

- `hu_attribution_provenance.jsonl`
  - Hu 标签的溯源版。
  - 它保留 LLM/规则如何得到训练标签的证据，适合人工复核。

- `schema_updates.jsonl`
  - LLM 认为现有 event schema 不够时提出的新 schema 或拆解建议。
  - 现在这些内容默认都是 `needs_human_review`，不会自动进入事件库。

- `probe_hits.jsonl`
  - 程序在轨迹中检测到的 probe state。
  - 用来判断某个偏好以后能不能在类似状态下被重新触发。

## 输出文件

Review 模块会写出两个文件：

- `review_items.jsonl`
  - 自动生成的“待审核任务包”。
  - 每条对应一条玩家反馈。
  - 内容包括反馈文本、自动归因、附近候选事件、近期轨迹、schema review、probe hit 和默认审核结论。

- `review_decisions.jsonl`
  - 人工审核后的最终决策。
  - 这个文件才是后续构建高质量 Hu 数据时应该优先读取的来源。

## 每条 review 要确认什么

1. 时间窗口是否正确
   - 反馈到底是在评价最近几步，还是更早的一段行为。

2. 事件归因是否正确
   - LLM/规则选中的 `approved_event` 是否真的是玩家在说的行为。

3. event 和 condition 是否混在一起
   - event 应该描述 AI 做了什么。
   - condition 应该描述这个行为发生时的状态事实。
   - 例如 `AI_should_get_dish_when_human_has_last_ingredient` 不适合作为 event；更合理的是：
     - event: `AI_missed_plate_pickup_opportunity`
     - condition: `human_has_last_ingredient=True`

4. condition 是否正确
   - 检查 `approved_condition_overrides`。
   - 如果程序提取的条件不够或错误，可以手动补充或修正。

5. subgoal preference 是否正确
   - 检查 `preferred_subgoals` 和 `rejected_subgoals`。
   - 例如“我已经放最后一个原料了，你去拿盘子”应该更偏向：
     - preferred: `GET_DISH`, `PICKUP_READY_SOUP`
     - rejected: `GET_ONION`, `WAIT`

6. schema update 如何处理
   - `schema_action` 可以是：
     - `none`
     - `accept_new_event`
     - `revise_new_event`
     - `map_to_existing`
     - `add_condition_only`
     - `reject`

7. 是否进入 Hu 训练
   - `use_for_hu_training=True` 表示这条审核结果可以作为 Hu 训练样本。
   - 模糊反馈、语义不清、证据不足的反馈可以保留为 `provenance_only`，暂时不训练。

## review_decisions.jsonl 字段

每条人工决策大致长这样：

```json
{
  "record_type": "review_decision",
  "feedback_event_id": "fb-000042",
  "feedback_total_step": 120,
  "feedback_text": "我已经放洋葱了，你去拿盘子",
  "decision": "revise",
  "approved_time_window": [108, 120],
  "approved_event": "AI_missed_plate_pickup_opportunity",
  "approved_condition_overrides": {
    "human_has_last_ingredient": true,
    "soup_cooking_or_ready": true,
    "ai_empty_handed": true
  },
  "approved_preference": {
    "preferred_subgoals": ["GET_DISH", "PICKUP_READY_SOUP"],
    "rejected_subgoals": ["GET_ONION", "WAIT"]
  },
  "use_for_hu_training": true,
  "schema_action": "map_to_existing",
  "approved_schema_update": null,
  "review_notes": "LLM 原本把条件写进 event name，这里改成 event + condition。"
}
```

## 这个设计和研究主线的关系

Review 窗口保留了两层数据：

- 溯源版：解释这条反馈为什么被归因到某个事件和条件，用于 debug、汇报和人工审核。
- 训练版：只保留 Hu 真正需要学习的结构化结论，也就是 condition 下 preferred/rejected subgoal 的偏好关系。

这样做的好处是：我们既不让 LLM 黑箱输出直接污染训练数据，也不把人工 review 伪装成模型能力。正式实验时，review 可以用于 pilot 数据清洗和 gold-label 评估；在线系统仍然需要单独报告自动归因的准确率。
