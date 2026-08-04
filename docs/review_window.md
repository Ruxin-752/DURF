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

Review 窗口的“近期轨迹”页提供“在地图上回放当前评价时段”按钮。地图回放直接使用
`trajectory.jsonl` 中已经记录的状态，不重新运行 agent，也不修改自动归因结果。播放器会显示：

- AI（蓝色）和人类（绿色）的位置、朝向与手持物；
- 锅、原料、盘子、出餐口和台面物体；
- 当前 timestep、双方动作、AI subgoal；
- 当前帧命中的候选事件；
- 自动或人工填写的归因时间段（红框）以及反馈发生时刻。

播放器支持播放/暂停、前后单步、时间轴拖动和三档速度。默认额外显示归因窗口前后各
3 步，帮助判断事件发生前因与结果。

- `feedback_events.jsonl`
  - 玩家输入的自然语言反馈。
  - 每条反馈有 `feedback_event_id`、文本、发生 timestep、episode 信息。

- `candidate_events.jsonl`
  - 程序从轨迹里检测出的候选事件。
  - 它回答“附近有哪些客观发生过、可能被评价的 AI 行为”。
  - 现在每条事件还包含 `actor` 和 `event_valence`：
    - `actor`: `ai` / `human` / `team` / `unknown`
    - `event_valence`: `positive_progress` / `negative_problem` / `missed_opportunity` / `neutral_context`
  - 负面反馈一般优先审核 `negative_problem` 和 `missed_opportunity`；正面反馈才优先看 `positive_progress`。
  - 完整文件记录全局事件，但归因输入会按反馈 timestep 截断，不能查看反馈之后的轨迹。

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

新版窗口默认打开“审核摘要”页，不再要求审核者直接阅读整块 JSON。每条反馈只需确认四件事：

1. 时间：反馈评价的是哪一段近期轨迹。
2. 行为：反馈指向哪个 AI event。
3. 条件：这个评价在什么状态条件下成立。
4. 偏好：Hu 应提高和降低哪些 subgoal。

中间证据区分为四个标签页：

- `审核摘要`
  - 默认页面。
  - 集中显示玩家原话、自动建议、关键条件、subgoal 偏好和需要特别注意的问题。
- `候选事件`
  - 用表格比较反馈前检测到的候选 event。
  - 选中候选后可以查看规则证据和关键条件。
  - 双击候选，或点击“采用选中候选”，会自动填写右侧 event 和时间窗口，并把 decision 改为 `revise`。
- `近期轨迹`
  - 每个 timestep 只显示 AI/人类动作、AI subgoal、双方手持物和锅状态。
  - 用于检查行为顺序，不再重复展示每一步的完整 condition 字典。
- `完整 JSON`
  - 保留全部原始证据，仅在摘要和表格不足以判断时查看。

左侧状态会明确区分：

- `AUTO: ...`：程序生成的默认建议，尚未人工保存。
- `REVIEWED: ...`：已经写入 `review_decisions.jsonl` 的人工结论。

注意：“上一条”和“下一条”不会保存当前修改。审核后应使用“保存”或“保存并下一条”。

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
   - 右侧不再要求手工输入名称，而是从全部合法 `HU_SUBGOALS` 中多选。
   - `preferred_subgoals` 表示：在当前 condition 下，更希望 AI 选择什么。
   - `rejected_subgoals` 表示：在当前 condition 下，更不希望 AI 选择什么。
   - 这是条件化的相对排序，不是永久提高或降低某个行为的全局奖励。
   - 窗口会显示归因时间段里真实出现过的运行时候选，但仍允许从完整 Hu subgoal 集合中选择。
   - 同一个 subgoal 不能同时出现在两侧。
   - 例如“我已经放最后一个原料了，你去拿盘子”应该更偏向：
     - preferred: `GET_DISH`, `PICKUP_SOUP`
     - rejected: `GET_ONION`, `WAIT`

自动默认值采用保守规则：

- LLM 如果只明确给出 preferred 或 rejected 一侧，程序不会擅自补齐另一侧。
- event 只有在明确表达替代行为时才提供默认 preferred。
- 对负面事件，只有轨迹中实际观测到的 subgoal 与 preferred 不同时，才可能作为默认 rejected。
- `AI_pick_drop_loop`、错误放置位置、冗余取物等无法由现有 subgoal 可靠表达的事件默认不生成完整 pair，必须人工确认或暂不训练。
- 正向事件通常只能说明“喜欢什么”，不能自动证明“相对于哪个行为更喜欢”，因此默认可能只填写 preferred。

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
   - 保存时会检查：event、时间窗、preferred/rejected 是否完整，两侧是否冲突，以及 decision 是否允许训练。

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
    "human_holding_last_needed_ingredient": true,
    "pot_cooking_or_ready": true,
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

Pilot 阶段可以用 Review 修正 schema 和训练标签；正式实验前必须冻结 detector 与 condition schema。正式测试数据不能边看边改规则，否则不同参与者的处理标准不一致。
