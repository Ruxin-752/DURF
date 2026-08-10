# Probe State Detector 说明

这个模块用于回答一个评估问题：Hu 训练或应用之后，AI 是否在相似情境下改变了 subgoal 排序？

它不是训练标签生成器，也不依赖用户恰好在那个时刻反馈。用户反馈可以帮助我们发现某类问题，但一旦某类问题被固化为 probe state，程序会在每个 timestep 自动扫描 condition features，命中后写入 `probe_hits.jsonl`。

## 核心思想

一次反馈通常只能说明某个时刻发生了问题：

```text
“汤快好了，你应该去拿盘子”
```

我们不会把它只绑定到某一帧，而是总结成一个可复现的状态模式：

```json
{
  "probe_name": "pot_ready_ai_should_get_dish",
  "required_conditions": {
    "pot_cooking_or_ready": true,
    "ai_empty_handed": true
  },
  "preferred_subgoals": ["GET_DISH", "PICKUP_SOUP"],
  "rejected_subgoals": ["WAIT", "GET_TOMATO", "GET_ONION"]
}
```

之后每次运行游戏或回放轨迹时，只要 condition 再次满足这个模式，就会记录一次 probe hit。同一个 probe 在同一个 episode 内默认至少间隔 3 帧（`--min-gap-steps`）才再次记录，避免高频刷屏。这样我们可以比较：

```text
Hu 前：这个 probe 下 subgoal 排序是什么？
Hu 后：这个 probe 下 subgoal 排序有没有变化？
```

## 代码入口

主要文件：

- `durf/feedback_attribution/probe_state_detector.py`

离线单独运行：

```powershell
python -m durf.feedback_attribution.probe_state_detector `
  --session outputs\human_ai_sessions\<session_id>
```

如果已经有 `trajectory.jsonl`，可以避免重新转换 CSV：

```powershell
python -m durf.feedback_attribution.probe_state_detector `
  --session outputs\human_ai_sessions\<session_id> `
  --no-convert
```

完整离线流水线也会自动生成 probe hits：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 50
```

## 输入

输入文件：

```text
outputs/human_ai_sessions/<session_id>/trajectory.jsonl
```

每个 step 至少需要：

```json
{
  "episode": 1,
  "episode_step": 85,
  "total_step": 85,
  "layout": "ring_tomato_onion_10x6_h0_full_task",
  "ai_subgoal": "GET_DISH",
  "ai_condition_features": {
    "pot_cooking_or_ready": true,
    "ai_empty_handed": true
  },
  "ai_subgoal_candidates": [
    {
      "subgoal": "GET_DISH",
      "task_score": 90.0,
      "hu_score": 0.8,
      "final_score": 90.8,
      "reason": "soup_ready_get_dish"
    },
    {
      "subgoal": "WAIT",
      "task_score": 0.0,
      "hu_score": -0.4,
      "final_score": -0.4,
      "reason": "fallback_wait"
    }
  ]
}
```

如果 `ai_condition_features` 不完整，程序会尝试从 `state_facts` 和 `state_before_json` 重新提取 condition。

## 输出

输出文件：

```text
outputs/human_ai_sessions/<session_id>/probe_hits.jsonl
```

每一行是一条自动检测到的 probe：

```json
{
  "record_type": "probe_hit",
  "probe_schema_version": "probe-state-v0",
  "probe_name": "pot_ready_ai_should_get_dish",
  "episode": 1,
  "episode_step": 85,
  "total_step": 85,
  "layout": "ring_tomato_onion_10x6_h0_full_task",
  "required_conditions": {
    "pot_cooking_or_ready": true,
    "ai_empty_handed": true
  },
  "condition_features": {
    "pot_cooking_or_ready": true,
    "ai_empty_handed": true,
    "human_has_dish": false
  },
  "preferred_subgoals": ["GET_DISH", "PICKUP_SOUP"],
  "rejected_subgoals": ["WAIT", "GET_TOMATO", "GET_ONION"],
  "chosen_subgoal": "GET_DISH",
  "candidate_ranking": [
    {
      "rank": 1,
      "subgoal": "GET_DISH",
      "task_score": 90.0,
      "hu_score": 0.8,
      "final_score": 90.8,
      "reason": "soup_ready_get_dish"
    }
  ],
  "evaluation": {
    "top_candidate": "GET_DISH",
    "preferred_available": true,
    "best_preferred_rank": 1,
    "best_rejected_rank": 2,
    "chosen_is_preferred": true,
    "chosen_is_rejected": false
  }
}
```

## 当前默认 Probe

当前 `probe-state-v0` 包含 8 类默认 probe：

| probe_name | 检测条件 | 期望偏好的 subgoal | 用途 |
|---|---|---|---|
| `pot_ready_ai_should_get_dish` | `pot_cooking_or_ready=true`, `ai_empty_handed=true` | `GET_DISH`, `PICKUP_SOUP` | 检测锅快好/已好时是否优先处理盘子和汤 |
| `human_path_conflict_ai_should_yield` | `human_trying_to_pass=true`, `ai_on_human_path=true` | `YIELD` | 检测通道冲突时是否让路 |
| `ai_holding_unneeded_onion_should_put_down` | `ai_has_onion=true`, `recipe_needs_onion=false` | `PUT_DOWN_OBJECT` | 检测拿着不需要洋葱时是否放下 |
| `ai_holding_unneeded_tomato_should_put_down` | `ai_has_tomato=true`, `recipe_needs_tomato=false` | `PUT_DOWN_OBJECT` | 检测拿着不需要番茄时是否放下 |
| `pot_needs_tomato_ai_should_get_tomato` | `recipe_needs_tomato=true`, `ai_empty_handed=true`, `pot_cooking_or_ready=false` | `GET_TOMATO`, `PUT_TOMATO_IN_POT` | 任务能力 sanity check |
| `pot_needs_onion_ai_should_get_onion` | `recipe_needs_onion=true`, `ai_empty_handed=true`, `pot_cooking_or_ready=false` | `GET_ONION`, `PUT_ONION_IN_POT` | 任务能力 sanity check |
| `human_waiting_ai_should_prepare_next` | `human_waiting_near_pot=true`, `ai_empty_handed=true` | `GET_USEFUL_INGREDIENT`, `GET_TOMATO`, `GET_ONION` | 检测等待时是否准备下一轮 |
| `useful_object_adjacent_ai_should_consider_pickup` | `useful_object_adjacent=true`, `ai_empty_handed=true` | `GET_USEFUL_INGREDIENT`, `GET_TOMATO`, `GET_ONION`, `GET_DISH` | 检测是否利用附近可用物体 |

## 和 event / condition / Hu 的关系

三者不要混在一起：

```text
condition = 当前世界事实
event = 过去轨迹里发生的候选行为事实
probe = 未来评估时可自动检测的状态模式
```

Hu 训练仍然使用：

```text
feedback -> attribution -> preferred_subgoal > rejected_subgoal
```

Probe 评估使用：

```text
trajectory step -> condition features -> probe hit -> candidate ranking
```

因此 probe 不是“靠用户反馈抓取”的，而是“由程序在轨迹中自动识别”的。

## 候选池按 domain 读取

每个 probe 声明自己的 `domain`（`task` 或 `coordination`），决定从哪份候选池读取评价依据：

- **task probe** 读取 `ai_subgoal_candidates`（task head 的 subgoal 池），`chosen_subgoal` 取该步执行的 task subgoal；
- **coordination probe** 读取 `coordination_decision.candidates`（协调选项池），`chosen_subgoal` 取 `coordination_decision.selected`。协调候选在构造时就绑定到具体低层动作，永远不会出现在 task 池中，因此不能从 `ai_subgoal_candidates` 读取——否则协调 probe 的 preferred/rejected 全部落空，评价静默空转。

当 coordination probe 命中但该步没有 `coordination_decision` 记录（例如旧版运行时产物）时，hit 会输出 `evaluation.evaluation_unavailable=true`，而不是对一个空池做无意义的评价。统计 Hu 前后差异时，应忽略 `evaluation_unavailable=true` 的 hit，并单独报告其数量，作为数据覆盖不足的信号。

## 如何比较 Hu 前后

最直接的评估方式：

1. 用 Hu shadow/apply 之前的 session 生成 `probe_hits.jsonl`。
2. 用同样地图、相似任务、同一批 probe，在 Hu 后 session 再生成 `probe_hits.jsonl`。
3. 对每个 `probe_name` 统计：
   - `best_preferred_rank` 是否下降到更靠前；
   - `chosen_is_preferred` 比例是否上升；
   - `chosen_is_rejected` 比例是否下降；
   - 任务分数是否没有明显降低。

这样可以证明 Hu 不只是解释了某条反馈，而是在相似状态下系统性改变了 subgoal preference。
