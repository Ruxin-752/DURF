# Task / Coordination 分层 Hu 工程说明

本文描述当前已经落到代码中的决策结构、数据流、日志字段和运行方式。

## 1. 为什么拆成两层

当前系统不再把 `GET_TOMATO` 和 `YIELD` 放进同一个候选集合比较。

- Task 决策回答“AI 现在要做什么”。
- Coordination 决策回答“遇到人类协作冲突时，当前任务要怎么继续执行”。
- Safety 只阻止非法动作。
- Recovery 只处理已经进入的无效任务状态，例如手里拿着当前不需要的原料。

运行顺序是：

```text
游戏状态
-> Task candidates + task_score
-> Hu_task preference_score
-> 选择 task subgoal
-> Recovery 检查
-> 检测即时路径冲突
-> Coordination candidates + coordination prior
-> Hu_coord preference_score
-> 选择 coordination option
-> Safety 检查
-> 执行低层动作
```

Task 和 Coordination 是上下游关系，但 Hu 内部是两个互不混合的评分头：

```text
Hu_task(condition, task_subgoal, user)
Hu_coord(condition, coordination_option, user)
```

因此“去拿番茄”和“暂时让路”不会形成没有意义的 pairwise 标签。

## 2. 当前候选空间

Task 候选由任务 planner 根据可行性生成，主要包括：

```text
GET_TOMATO
PUT_TOMATO_IN_POT
GET_ONION
PUT_ONION_IN_POT
GET_DISH
PICKUP_SOUP
SERVE_SOUP
GET_USEFUL_INGREDIENT
PUT_DOWN_OBJECT
WAIT_NEAR_POT
WAIT
```

⚠️ **这是词表，不是每一步都同时可选**。手里拿着东西时，候选生成器目前仍然是一条带
`return` 的规则级联，绝大多数持物状态只会产出上表里的**一个**候选（详见
`docs/mechanism_blueprint_v1.md` §1.3 的实测数据）。放开这些分支、让它们改成 `append`
是 `mechanism_blueprint_v1.md` 提出、尚未落地的工作，不要把这张词表读成"每步都有全部
候选参与竞争"。

当前运行时已经启用的 Coordination 候选是：

```text
CONTINUE_CURRENT_SUBGOAL
YIELD
```

`HOLD_POSITION` 与 `YIELD` 在动作上重合（都是停在原地），已从词表移除。`REROUTE`
**不是**新的协调选项——Hu 的协调词表现在和以前一样，永远只有这两个。REROUTE 已经实现，
但是作为 `YIELD` 选中之后、决定"具体怎么让"的执行层规则（见 §4.1），不是 Hu 要挑选的
第三个选项。

## 3. 基础分数与 Hu 分数

Task 层和 Coordination 层用的是两条**不同**的混合规则，不要套用同一个公式。

**Task 层：ε-约束分层满足式（不是加法）**。`task_score` 和 `hu_score` 单位不同、不可通约，
从来就不该相加——`task_score` 是研究者标定的任务价值判断，量级是几十到一百；`hu_score`
来自成对 logistic 排序模型，只有序信息可靠，量级不固定。实际规则是：

```text
best_task = 可行候选里最高的 task_score
acceptable = { c | c.task_score >= best_task - hu_task_tolerance }   # 容忍带内的候选
selected = acceptable 里 hu_score 最高的那个                          # Hu 只在容忍带内挑
selected.final_score = selected.task_score                           # 不做加法
```

`hu_task_tolerance`（原来叫 `hu_lambda`，已改名并改了语义）是"愿意为学到的偏好放弃多少
任务分"，单位是**任务点数**，不是权重系数。`hu_task_tolerance = 0` 时容忍带只剩任务最优
一个候选，行为与迁移前的任务底座完全一致。实现见
`durf/baseline/collect_rule_teacher_dataset.py::choose_task_candidate`。

**Coordination 层：仍然是加法**。这一层还没有做同样的重构：

```text
coord_final_score
= coordination_prior
+ hu_coordination_lambda * Hu_coord(user, condition, option)
```

`coordination_prior` 是 0/1 量级，`hu_coordination_lambda` 依然是权重系数而不是任务点数，
和 task 层的 `hu_task_tolerance` 不是同一种东西，不要混用参数含义。这一层是否也要改成
满足式，是待评审的开放问题。

当 `--hu-apply` 未开启时，Hu 只进行 shadow scoring：记录分数，但不改变选择。

当 Hu 没有 coordination head，或者 `--hu-coordination-lambda 0` 时，系统使用初始协调 prior。当前 prior 在检测到直接路径冲突时优先 `YIELD`，以保持旧版本行为。

## 4. Coordination option 的生命周期

`YIELD` 不是永久覆盖 task。

```text
检测到冲突
-> 创建 coordination decision_id
-> 短时间保持所选 option，避免每一步左右摇摆
-> option 到期
-> 如果冲突消失，恢复原 task
-> 如果冲突持续，YIELD 进入短 cooldown
-> 重新比较候选，避免无限后退
```

当前默认参数：

```text
min_commit_steps = 1
max_option_steps = 3
yield_cooldown_steps = 2
```

### 4.1 `YIELD` 选中之后：WAIT / BACK_OFF / REROUTE

`YIELD` 曾经等价于"原地停住"（`STAY`）。这混淆了两件事：**要不要让路**（Hu 该学的
偏好）和**让路的具体动作**（该不该往后退一步、该不该干脆换条不经过人的路）。现在这两
件事分开了：

- Hu 仍然只在 `CONTINUE_CURRENT_SUBGOAL` / `YIELD` 之间选，词表没有变化，训练数据、
  pairwise 样本、`condition_delta` 维度都不需要重新设计——这正是当初决定"先只拆
  执行层，不拆 Hu 词表"的原因（数据量在 coordination 维度已经吃紧，见
  `docs/hierarchical_hu_runtime.md` 历史讨论）。
- `YIELD` 一旦被选中，一段**规则判断**（不经过 Hu、不训练）决定具体怎么让，按优先级：

  ```text
  REROUTE   -- 找一条绕开人当前位置和目的地的路，继续推进任务
              （复用 collect_rule_teacher_dataset.first_action_to_feature，
              把人的当前格和目的格当成临时障碍）
  BACK_OFF  -- 找不到能绕开人的路时，退到离人和人的目的地都最远的相邻空格
              （原来就有的逻辑，但以前只在 2/4 种冲突类型下触发，现在对全部
              4 种冲突类型都生效）
  WAIT      -- 两者都不可行时，原地停住（STAY）
  ```

  实现在 `durf/baseline/coordination.py::choose_reroute_action`，`play_with_baseline.py`
  和 `sim_session.py` 共用同一份实现（不再各自维护一份重复代码）。

- 这次拆分选中的**具体方式**（`"reroute"` / `"back_off"` / `"wait"`）会写进
  `YIELD` 候选的 `reason` 字段（例如
  `yield_during_ai_blocking_human_route_via_back_off`），供审计和后续分析用，
  但**不会**出现在 `candidate_set` 或 Hu 的决策空间里——`build_coordination_candidates`
  的 `yield_mode` 参数只影响 `reason` 文本。

⚠️ **实测发现（2026-09-03）**：在默认的 `ring_tomato_onion_10x6_h0_full_task` 环形地图
上跑了 4 种仿真人格 × 6 个随机种子共 180 次协调冲突，`REROUTE` 一次也没有被选中——全部
落到 `BACK_OFF`。原因是几何上的：这张图确实是一个环（两条方向都能走到），但冲突发生时
AI 和人已经贴在一起，绕环一整圈的代价远高于退让一两步；而"人挡住的那两格"通常不在 AI
已经规划好的最短路径上（最短路径本来就没打算经过人），所以把这两格设成临时障碍对路径
规划毫无影响。这不代表 `REROUTE` 是死代码——`durf/baseline/test_coordination_yield.py`
在同一张地图上直接单测了 `choose_reroute_action`，构造了"最短路径确实经过人所在的那条
窄连接通道"的场景，证明它能正确绕到另一条通道。只是在这张图默认的对局动态下，这个分支
很少被触发，真人数据或更窄的地图布局可能会改变这个比例。

## 5. trajectory 中新增的决策证据

`trajectory.csv` 每一步新增：

```text
task_decision_json
coordination_decision_json
```

Task 记录示例：

```json
{
  "record_type": "runtime_decision",
  "decision_id": "task:e1:t42",
  "decision_level": "task",
  "condition_at_decision": {
    "pot_partially_filled": true,
    "recipe_needs_onion": true
  },
  "candidate_set": ["GET_ONION", "GET_TOMATO", "WAIT"],
  "candidates": [
    {
      "subgoal": "GET_ONION",
      "task_score": 70.0,
      "hu_score": 0.8,
      "final_score": 70.0
    }
  ],
  "selected": "GET_ONION"
}
```

Coordination 记录示例：

```json
{
  "record_type": "runtime_decision",
  "decision_id": "coord:e1:t42:n1",
  "decision_level": "coordination",
  "conflict_type": "human_entering_ai_tile",
  "task_subgoal": "GET_ONION",
  "condition_at_decision": {
    "human_trying_to_pass": true,
    "narrow_corridor": true,
    "ai_adjacent_to_current_subgoal_target": false
  },
  "candidate_set": ["CONTINUE_CURRENT_SUBGOAL", "YIELD"],
  "selected": "YIELD",
  "expires_at": 44
}
```

这两条记录共同回答：

- AI 原来想做什么；
- 当时有哪些候选；
- 基础模型和 Hu 各给了多少分；
- 是否发生协作冲突；
- 最终哪一层改变了动作。

## 6. 反馈到训练标签

主数据流：

```text
trajectory.csv + feedback.csv
-> trajectory.jsonl + feedback_events.jsonl
-> candidate_events.jsonl
-> attribution_preview.jsonl
-> hu_attribution_provenance.jsonl
-> hu_subgoal_preferences.jsonl
-> hierarchical_hu.json
```

其中：

- `candidate_events.jsonl` 保存程序检测到的行为证据。
- `attribution_preview.jsonl` 保存规则或 LLM 对语言、时间段和事件的自动归因。
- `hu_attribution_provenance.jsonl` 保存完整溯源链。
- `hu_subgoal_preferences.jsonl` 只保存 Hu 需要的 pairwise 标签。

训练标签必须属于同一个决策域：

```json
{
  "decision_level": "coordination",
  "condition_features": {
    "human_trying_to_pass": true,
    "narrow_corridor": true
  },
  "preferred_subgoal": "YIELD",
  "rejected_subgoal": "CONTINUE_CURRENT_SUBGOAL",
  "source_event": "AI_blocked_human_path",
  "source_decision_id": "coord:e1:t42:n1",
  "label_source": "human_review"
}
```

跨域标签会被拒绝，例如：

```text
YIELD > GET_TOMATO
```

## 7. 人工 review 如何进入训练

review 窗口写入：

```text
review_decisions.jsonl
```

原始自动归因文件不会被修改。重新运行 dataset builder 时：

```text
存在 review decision
-> approved time/event/condition/preference 覆盖自动猜测
-> use_for_hu_training=true 才生成 reviewed 训练样本
-> use_for_hu_training=false 则排除该条
```

`label_source=human_review` 用于区分人工确认标签和未审核的自动标签。

事件库操作还有额外校验：

- `accept_new_event` / `revise_new_event` 必须填写新事件定义 JSON。
- `map_to_existing` 必须填写已有 event。
- `add_condition_only` 必须填写至少一个 condition。

## 8. 主要代码文件

- `durf/group_a/play_with_baseline.py`
  游戏入口；串联 Task、Recovery、Coordination、Safety，并写运行日志。
- `durf/baseline/collect_rule_teacher_dataset.py`
  生成 Task candidates 和 `task_score`。
- `durf/baseline/coordination.py`
  检测冲突后的协调候选、短期 option 状态和 cooldown。
- `durf/feedback_attribution/condition_features.py`
  提取固定 condition schema，并区分 task/coordination 使用的字段。
- `durf/feedback_attribution/event_detectors.py`
  从轨迹和显式 runtime decision 生成 candidate events。
- `durf/feedback_attribution/hu_dataset_builder.py`
  生成溯源记录和双域 pairwise 标签，并消费人工 review。
- `durf/hu/subgoal_reranker.py`
  `HierarchicalHu` 及两个独立线性 pairwise heads。
- `durf/hu/train_subgoal_reranker.py`
  分域切分数据、训练并输出模型。
- `durf/hu/score_subgoals.py`
  离线查看 task 或 coordination head 的候选排序。

## 9. 运行命令

离线归因并重建标签：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 30 `
  --use-llm
```

训练双头 Hu：

```powershell
python -m durf.hu.train_subgoal_reranker `
  --dataset outputs\human_ai_sessions\<session_id> `
  --output-dir outputs\hu_models\pilot01 `
  --epochs 200 `
  --learning-rate 0.05
```

输出：

```text
outputs/hu_models/pilot01/hierarchical_hu.json
outputs/hu_models/pilot01/metadata.json
```

先做 shadow 验证：

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --hu-model outputs\hu_models\pilot01\hierarchical_hu.json `
  --hu-user-id PILOT01
```

确认分数合理后再实际应用：

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --hu-model outputs\hu_models\pilot01\hierarchical_hu.json `
  --hu-user-id PILOT01 `
  --hu-task-tolerance 10.0 `
  --hu-coordination-lambda 1.0 `
  --hu-apply
```

`--hu-lambda` 已不存在；task 层用 `--hu-task-tolerance`，单位是任务点数而不是权重，示例值
请按实际标定结果调整。

## 10. 当前边界

已经完成：

- Task 与 Coordination 分域；
- 显式 `CONTINUE_CURRENT_SUBGOAL` / `YIELD` 运行时候选；
- 双头 Hu 训练和旧单头模型兼容；
- decision-level 日志与 event 溯源；
- review 标签覆盖自动归因；
- bounded yield，避免一直退让；
- Task 层的 ε-约束分层满足式决策规则，替代了原来量纲不匹配的加法混合（见 §3）；
- 候选存废与队友位置解耦：队友挡路只影响路线可行性，不再让候选直接消失（见
  `durf/baseline/collect_rule_teacher_dataset.py::feature_candidate` 的
  `route_blocked_by_partner` 标记）；
- 低层执行器正式下线：候选生成器能产出的全部 subgoal 都直接走运动规划器，训练好的
  keras 执行器网络默认不再加载（`play_with_baseline.py` 和 `sim_session.py` 都新增了
  `--load-retired-executor` 标志，默认关闭，仅用于回归对比时手动加载）——这两个入口
  之前不一致：`sim_session.py` 曾经遗漏了这次下线，仍然无条件加载模型、保留一段永远
  不会命中的白名单分支，现已补齐一致；
- `PerUserAdapter` 的 task 头默认关闭 `user_bias`（`enable_task_bias=False`）：跨四个
  sim persona 验证，task 层的无条件个体偏移始终落在噪声范围内（±0.06），而
  coordination 层的个体偏移是有意义的信号（±0.05 到 ±1.74，随人格单调变化）。这是可逆
  开关，不是删除，真人数据到手后需要重新检验；
- YIELD 的语义拆分：Hu 的协调词表不变（仍然只有 `CONTINUE_CURRENT_SUBGOAL` / `YIELD`
  两个选项），但 `YIELD` 选中之后由规则判断具体怎么让——REROUTE（绕开人、继续推进任务）
  优先于 BACK_OFF（退到最远的空格，现覆盖全部 4 种冲突类型而非原来的 2 种）优先于 WAIT
  （原地不动）。选中的方式记录在 `reason` 字段供审计，不进入 Hu 的决策空间（见 §4.1）；

仍待真实数据验证：

- 两个 head 是否都有足够 pairwise 样本；
- Hu 是否在同一 probe condition 下稳定改变候选排序；
- Hu 改变偏好后是否保持任务完成率；
- `PerUserAdapter.enable_task_bias=False` 这个假设在真人数据上是否依然成立；
- REROUTE 在真人对局里的实际触发率——仿真环形地图上的 180 次冲突里一次都没触发（见
  §4.1），这是否是地图几何的特例，还是真人协调冲突普遍也是"贴身冲突、绕路不划算"。

仍未开工：

- 持物分支的候选生成放开（`mechanism_blueprint_v1.md` §3 的 P1，`WAIT_NEAR_POT` /
  `GET_USEFUL_INGREDIENT` 等候选目前不会在持物状态下出现）；
- Coordination 层是否也要从加法混合改成满足式。
