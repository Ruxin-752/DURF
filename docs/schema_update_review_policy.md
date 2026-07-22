# Schema Update Review Policy

本文档说明 LLM 提出的 `proposed_schema_update` 如何进入人工审核，而不是被直接加入事件库。

## 核心边界

我们的 schema 分三层：

```text
event = AI 的行为模式
condition = 当时世界状态事实
preference = 人希望 subgoal 排序如何改变
```

这三层不能混在一个字段里。

## Event 应该是什么

event 只描述 AI 做了什么，或者错过了什么行为机会。

推荐：

```text
AI_missed_plate_pickup_opportunity
AI_missed_useful_counter_object
AI_missed_labor_division_opportunity
AI_blocked_human_path
AI_pick_drop_loop
```

不推荐：

```text
AI_should_get_dish_when_human_has_last_ingredient
AI_should_use_table_tomato_near_pot
AI_should_get_ingredient_when_human_closer_to_dish
```

这些不推荐命名的问题是：把状态条件塞进了 event 名字。

## Condition 应该是什么

condition 是从轨迹和 state facts 中提取的客观事实。

例如：

```json
{
  "human_has_last_needed_ingredient": true,
  "dish_available": true,
  "ai_empty_handed": true,
  "counter_has_useful_tomato_near_pot": true,
  "human_closer_to_dish": true,
  "ai_closer_to_ingredient": true
}
```

condition 不表达“人喜欢什么”，只表达“当时发生了什么/状态是什么”。

## Preference 应该是什么

preference 是 Hu 的训练目标，通常是 pairwise subgoal label：

```text
preferred_subgoal > rejected_subgoal
```

例如：

```json
{
  "preferred_subgoal": "GET_DISH",
  "rejected_subgoal": "GET_TOMATO"
}
```

或者：

```json
{
  "preferred_subgoal": "GET_USEFUL_INGREDIENT",
  "rejected_subgoal": "WAIT"
}
```

## LLM Schema Update 如何处理

LLM 可以提出 schema update，但它的输出只是原始建议，不会被直接加入 event detector。

处理流程：

```text
LLM raw proposal
-> schema_updates.jsonl
-> human review
-> 拆分为 candidate_event + condition keys + subgoal preference mapping
-> 再决定是否写入 event_detectors.py / condition_features.py / subgoal_preferences.py
```

例如 LLM 提出：

```text
AI_should_get_dish_when_human_has_last_ingredient
```

人工审核时应拆成：

```json
{
  "candidate_event": "AI_missed_plate_pickup_opportunity",
  "new_condition_keys": [
    "human_has_last_needed_ingredient",
    "dish_available"
  ],
  "preferred_subgoals": ["GET_DISH"],
  "rejected_subgoals": ["GET_TOMATO", "PUT_TOMATO_IN_POT", "WAIT"]
}
```

## 当前代码行为

`durf/feedback_attribution/hu_dataset_builder.py` 会把 LLM 的 `proposed_schema_update` 写成审核记录：

```text
schema_updates.jsonl
```

每条记录包含：

```text
raw_schema_update
raw_name
status = needs_human_review
suggested_decomposition.event_hint
suggested_decomposition.condition_hint
suggested_decomposition.condition_bundled_in_name
preferred_subgoals
rejected_subgoals
condition_features_at_feedback
```

注意：

```text
schema_updates.jsonl 不是训练数据。
schema_updates.jsonl 也不是最终事件库。
它只是人工审核队列。
```

## 为什么这样设计

如果允许 LLM 直接生成带条件的 event，事件库会膨胀成很多一次性名字：

```text
AI_should_get_dish_when_pot_ready
AI_should_get_dish_when_human_has_last_ingredient
AI_should_get_dish_when_ai_closer_to_dish
```

这样 Hu 学不到条件化偏好，只会记住事件名字。

保持三层拆分后，Hu 学到的是：

```text
在某些 condition 下，某些 subgoal 更符合用户偏好。
```

这才和我们的研究主线一致。
