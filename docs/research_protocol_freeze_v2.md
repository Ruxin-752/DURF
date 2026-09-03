# 研究协议冻结 v2：共享协作常识与参与者独立偏好

> 冻结日期：2026-08-11  
> 状态：**研究方向冻结；工程实现按本文逐项对齐**  
> 适用范围：Overcooked-AI 双人协作、固定地图/任务、Task/Coordination 分层 Hu

本文是当前研究方向的正式协议。它覆盖模型边界、数据来源、标签格式、训练方式、评估指标和禁止事项。后续代码、实验记录和汇报材料应以本文为准；旧文档中与本文冲突的内容视为历史设计，不再作为主协议。

## 1. 研究目标

我们研究的不是重新训练一个会做饭的 agent，而是：

> 在基础 agent 已经具备任务完成能力的前提下，利用少量、条件化的人类 pairwise feedback，学习每位参与者独立的任务偏好与协作偏好，并检验这些偏好是否能改变 agent 的行为排序，同时保持任务性能。

基础任务能力由固定的 task backbone 提供。Hu 不负责学习走路、拿取、烹饪或低层动作控制；Hu 只在已有候选行为之间进行偏好重排序。

## 2. 固定的系统分层

运行时结构固定为：

```text
游戏状态
  -> Task candidate generation + task_score
  -> Hu_task：选择更偏好的任务 subgoal
  -> Coordination conflict detection
  -> Coordination candidates + coordination_prior
  -> Hu_coord：选择更偏好的协作行为
  -> Safety / Recovery
  -> subgoal executor / MotionPlanner
  -> 低层动作
```

两个决策域不能混成一个 pairwise 比较：

```text
Task：GET_ONION > GET_TOMATO
Coordination：YIELD > CONTINUE_CURRENT_SUBGOAL
```

禁止生成没有意义的跨域标签：

```text
YIELD > GET_TOMATO
GET_DISH > CONTINUE_CURRENT_SUBGOAL
```

Task 和 Coordination 可以共享状态事实与 condition schema，但使用独立的候选集合、评分头、训练标签和评估结果。

## 3. 三类模型对象

### 3.1 固定 Task Backbone

Task backbone 是已经能完成基本任务的执行系统，当前形式为：

```text
state -> task candidates -> task_score -> subgoal executor -> action
```

它提供：

- 合法候选 subgoals；
- 任务完成导向的 `task_score`；
- 低层动作执行能力；
- 基础任务性能约束。

它不被参与者反馈重新训练。这样可以把“任务能力”与“人类偏好”分离。

### 3.2 `Hu_general`：共享、persona-agnostic 的通用先验

`Hu_general` 不是把所有 sim persona 的 preference 平均混合出来的模型。

它只允许学习以下两类内容：

1. 不依赖参与者人格、在合理协作中具有一致方向的常识；
2. 从 sim 数据中跨 persona 方向一致的行为关系。

例如，以下关系可以进入 `Hu_general`：

```text
明显挡住人类当前通路时，不应无限期阻挡
汤已经准备好时，应考虑从等待转向取盘/送餐
当前 subgoal 已经失效时，不应重复执行无效动作
```

以下关系不能直接混入 `Hu_general`：

```text
遇到人类时一定让路
遇到冲突时一定坚持当前任务
人类经过通道时应该优先照顾谁
```

如果 cooperative 与 selfish persona 在同一条件下给出相反标签，该样本不能直接用于通用 preference head。它可以被保留为：

- persona-specific 模拟数据；
- 训练/测试协议调试数据；
- 检查模型能否表示相反偏好的数据。

`sim_both_v1` 的 0.92/0.923 结果只能说明它在当前 synthetic split 上拟合良好，不能单独证明它已经学到了真人通用偏好。今后报告时必须标注：`synthetic validation`。

### 3.3 `delta_user`：参与者独立的偏好适配器

每位参与者拥有独立的、条件化的偏好适配器。最终用户模型固定为三级结构：

```text
Hu_user(x) = Hu_general(x) + user_bias(x) + condition_delta_user(x)
```

三层的含义是：

1. **全局 `user × subgoal` bias**：表示该参与者总体上更偏好哪些 task/coordination subgoals。它是低样本下的稳定化层，但不能单独代表条件化偏好。
2. **用户实际观察到的 condition 修正**：只为该用户在 reviewed feedback 中真实出现过的 condition 学习小量 `condition_delta_user`。
3. **未观察 condition 的 general 回退**：如果该用户从未观察到某个 condition，则不估计个人修正，继续使用冻结的 `Hu_general`，不强行推断该用户的偏好。

因此更具体地写作：

```text
Hu_user(subgoal | condition)
  = Hu_general(subgoal | condition)
  + user_bias(user, subgoal)
  + sum(active_condition_k * delta_user(user, condition_k, subgoal))
```

其中：

- `Hu_general`：冻结，不使用其他参与者的真人反馈更新；
- `user_bias`：只使用当前参与者的 reviewed human feedback 训练；
- `condition_delta_user`：只使用当前参与者实际观察到的 active conditions 训练；
- `delta_user_A`、`delta_user_B` 之间不共享真人 preference 参数或训练样本；
- 未观察 condition 的权重保持为 0，行为由 `Hu_general` 提供。

这里的“用户适配”不是重新训练基础 agent，也不是重新训练 PPO。它只改变 Hu 对候选 task/coordination subgoal 的偏好分数。

第一版仍采用小参数模型，但必须从一开始保留上述三级接口：

```text
user_bias = user × subgoal bias
condition_delta_user = active_condition × subgoal weight
```

在工程实现上，可以先对第二层使用更强的正则化或较小学习率，而不是暂时删除它。数据不足时：

```text
已观察 condition：允许学习小量 delta
未观察 condition：delta = 0，回退 Hu_general
```

不在一开始使用大 MLP 或完整独立模型，以避免 30–40 条反馈下过拟合。无条件 `user_bias` 只能作为总体风格和正则化项，不能作为最终研究中“条件化个体偏好已经学会”的证据。

## 4. 数据来源与隔离规则

### 4.1 数据来源分层

每条数据必须带有来源标记：

```json
{
  "source": "simulated_prior | human_feedback",
  "label_status": "synthetic | automatic | reviewed"
}
```

来源用途固定为：

| 数据 | 可以训练什么 | 不可以训练什么 |
|---|---|---|
| sim 中跨 persona 一致的数据 | `Hu_general` | 参与者 A 的 `delta_user` |
| sim 中 persona 冲突的数据 | persona 调试/对照 | 混入通用先验 |
| 参与者 A 的真人数据 | `delta_user_A` | `Hu_general`、参与者 B |
| 参与者 B 的真人数据 | `delta_user_B` | `Hu_general`、参与者 A |
| 自动归因但未 review 的数据 | 诊断/候选训练 | 正式 Hu 结果 |
| reviewed human labels | 正式 `delta_user` 训练 | 其他参与者 |

### 4.2 参与者独立原则

每位参与者的数据流程是：

```text
participant A session
  -> A 的 review
  -> A 的 pairwise dataset
  -> A 的 delta_user
```

不能把多个参与者的真人 feedback 拼接成一个用户模型再声称它代表个体偏好。

共享的只是：

- 地图和任务定义；
- condition schema；
- candidate event taxonomy；
- `Hu_general` 的冻结参数；
- 归因程序和 review 工具。

## 5. 固定 condition schema

全局保留统一的 28 条 condition schema，不为不同参与者临时创造不同输入维度。

每条样本只对当前真实成立且与决策相关的条件产生有效输入：

```text
True  -> 1
False -> -1
Unknown/不可恢复 -> 0
```

未在该样本中激活的 condition 不产生用户增量更新；其行为继续使用冻结的 `Hu_general`。

这不是删除 condition，而是：

```text
统一 schema + 样本级 active mask
```

condition 只描述状态事实，例如：

```text
narrow_corridor
human_trying_to_pass
human_has_dish
pot_cooking_or_ready
ai_empty_handed
ai_on_human_path
```

condition 不表达：

- 用户喜欢什么；
- 反馈正负；
- 事件名称；
- preferred/rejected 结论。

高层条件类型可以用于组织 sim prior 或 active probe，但不能替代原始 condition，也不能把不同参与者的真实 preference 聚合在一起。

## 6. Event、condition 与 preference 的关系

三者职责固定：

```text
event      = AI 做了什么行为
condition  = 行为发生时状态是什么
preference = 用户更希望哪个候选行为
```

例子：

```json
{
  "event": "AI_blocked_human_path",
  "condition": {
    "narrow_corridor": true,
    "human_trying_to_pass": true,
    "ai_on_human_path": true
  },
  "preferred_subgoal": "YIELD",
  "rejected_subgoal": "CONTINUE_CURRENT_SUBGOAL"
}
```

`event` 主要用于溯源和解释；`condition` 与 `preferred/rejected` 才是 Hu 训练的核心输入/标签。LLM 可以从候选 event 中选择最符合语言的事件，也可以提出新 event schema，但不能绕过轨迹事实直接创造发生过的行为。

## 7. 反馈标签格式

正式 Hu 训练使用 pairwise preference：

```json
{
  "record_type": "hu_pairwise_subgoal_preference",
  "user_id": "PILOT01",
  "decision_level": "coordination",
  "condition_features": {
    "narrow_corridor": true,
    "human_trying_to_pass": true,
    "ai_on_human_path": true
  },
  "preferred_subgoal": "YIELD",
  "rejected_subgoal": "CONTINUE_CURRENT_SUBGOAL",
  "source_event": "AI_blocked_human_path",
  "source_feedback_id": "feedback_001",
  "source": "human_feedback",
  "label_status": "reviewed"
}
```

训练目标为：

```text
Hu_user(preferred | condition)
>
Hu_user(rejected | condition)
```

不把 `+1/-1` 作为当前主训练接口，不把自然语言直接当作 reward，不把 `event` 字符串直接当作模型输入。

## 8. 反馈处理流程

正式流程固定为：

```text
游戏轨迹 + 自然语言反馈
  -> 程序提取客观 state facts
  -> 程序生成 candidate events
  -> 时间窗口筛选
  -> LLM 从候选中做语义归因
  -> 生成溯源版 attribution
  -> 人工 review 必要字段
  -> 生成训练版 pairwise labels
  -> 只训练该参与者 delta_user
```

两个输出必须分开：

### 8.1 溯源版

`hu_attribution_provenance.jsonl`

保留：

- 原始反馈文本；
- 时间窗口；
- candidate events；
- LLM 选择理由；
- condition；
- event；
- preferred/rejected；
- confidence；
- review 修改记录。

用途是审计归因质量，不直接作为模型输入。

### 8.2 训练版

`hu_subgoal_preferences.jsonl`

只保留：

- `user_id`；
- `decision_level`；
- `condition_features`；
- `preferred_subgoal`；
- `rejected_subgoal`；
- 来源和审核状态。

用途是训练 `delta_user`。

## 9. 少量数据协议

每位参与者预计产生约 30–40 条反馈。此数据量不支持从零训练完整 28-condition preference model，因此采用以下限制：

1. `Hu_general` 先提供共享常识，且冻结。
2. 用户只训练小规模 `delta_user`。
3. 只在当前反馈真实涉及的 condition 上更新增量。
4. 使用 pairwise 标签，不要求绝对分数。
5. 优先收集能区分两个候选行为的反馈，而不是重复收集同类场景。
6. 未观察 condition 保留 general prior，不强行推断个人偏好。

7. 用户偏好模型必须同时保留总体 `user × subgoal` 层和 active-condition 修正层；不能只训练无条件 user bias。

建议的反馈使用方式：

```text
训练/适配：约 20–25 条
同参与者 probe 测试：约 5–10 条
```

如果总量不足以固定 train/test，应采用按时间顺序的 online learning curve 或 leave-one-feedback-out，而不是声称有独立测试集。

## 10. Probe 与 Active Collection

Probe state 分为两类：

### 10.1 离线 probe evaluation

由程序从轨迹或状态库检测出满足条件的 probe state，例如：

```text
Hu_general(YIELD) 接近 Hu_general(CONTINUE)
```

在相同 state、相同 candidate set、相同 condition 下，比较：

```text
Hu_general 排序
vs
Hu_user 排序
```

它回答：真人数据是否改变了正确的行为排序。

### 10.2 在线 active feedback collection

这是后续实验协议，不是当前 Hu-v0 的必要架构。未来可以在游戏中检测：

```text
候选分数接近
或
condition 覆盖不足
```

再请求用户进行 pairwise 选择。第一版不强制暂停游戏，不把 active collection 与基础数据流同时改动。

## 11. 正式对照条件

每位参与者至少比较：

1. `Backbone only`：没有 Hu，作为任务能力基线。
2. `Hu_general`：只有共享通用先验，没有该参与者真人适配。
3. `Hu_general + delta_user`：该参与者独立适配后的模型。

必要时增加：

4. `User from scratch`：不使用 general prior，只使用该参与者数据，检验 delta/先验是否帮助小样本学习。

所有条件使用相同：

- 地图；
- 任务；
- task backbone；
- candidate generator；
- condition extraction；
- 低层 executor；
- 评估 seed 与 episode horizon。

## 12. 评估指标

### 12.1 Preference learning

- pairwise accuracy：模型是否把 preferred 排在 rejected 前；
- pairwise margin：偏好差值是否稳定为正；
- probe ranking change：真人适配前后排序是否改变；
- condition-specific consistency：改变是否只出现在相关条件下。

### 12.2 Attribution quality

单独评估，不与 Hu accuracy 混为一谈：

- 时间窗口是否覆盖人工认定的目标区间；
- event 是否正确；
- preferred/rejected subgoal 是否正确；
- condition 是否来自目标事件发生时，而不是反馈结束时；
- LLM 归因与人工 review 的一致率。

### 12.3 Task and collaboration performance

- episode reward；
- delivered orders；
- task success rate；
- invalid/recovery actions；
- blockage duration；
- unnecessary yielding；
- human-reported preference satisfaction。

核心约束是：

```text
Hu_user 应改善相关偏好指标，且不能造成明显的任务完成退化。
```

不能只报告 Hu 是否改变了排序，而不报告任务能力。

## 13. 禁止事项

当前协议下禁止：

- 把不同真人的 preference 混合训练一个 participant model；
- 把 cooperative/selfish 的相反标签直接平均成 Hu_general；
- 用未经 review 的自动归因作为正式真人训练标签；
- 让 Hu 直接修改环境 reward；
- 让 Hu 直接输出低层动作；
- 把 Task subgoal 与 Coordination option 放进同一个 pairwise 集合；
- 把 `event` 当作 condition；
- 用 future trajectory 解释反馈发生前的状态；
- 用单次 synthetic accuracy 宣称理解了真人偏好。

## 14. 工程实现状态

当前已经存在或基本对齐：

- Task/Coordination 双头 Hu；
- candidate subgoal 与 task_score；
- coordination prior、`YIELD` 和 `CONTINUE_CURRENT_SUBGOAL`；
- 固定 condition schema；
- candidate event、LLM attribution、review 数据流；
- 溯源版/训练版数据分离；
- pairwise Hu-v0 训练；
- shadow/apply 在线运行模式；
- probe state detector 的离线框架。

必须按本协议补齐：

1. 从 sim persona 数据筛出跨 persona 一致的 `Hu_general` 训练集。
2. 明确并记录冲突 sim 样本，不再混入通用先验。
3. 在 `HierarchicalHu` 中实现冻结 general 权重，以及独立训练 `user_bias` 和 active-condition `condition_delta_user` 两层。
4. 为每位参与者生成独立的训练目录、metadata 和评估结果。
5. 实现 `Backbone only / Hu_general / Hu_general + user_bias / Hu_general + user_bias + condition_delta_user` 对照；至少保证前三项和最终完整模型可运行。
6. 在 probe state 上输出适配前后的候选排序、margin 和任务指标。
7. 将 active probe 作为后续采集协议，不提前与 Hu-v0 架构绑定。

## 15. 最小可交付闭环

第一版交付必须完成以下闭环：

```text
一名参与者的真人 session
  -> review 后的 pairwise labels
  -> 冻结 Hu_general
  -> 训练独立 delta_user
  -> 在同一参与者的 probe states 上比较排序
  -> shadow/apply 验证
  -> 同时报告 task 与 preference 指标
```

只有这个闭环跑通后，才进入多参与者、active collection、复杂模型或跨地图泛化。

## 16. 一句话版本

> 固定能完成任务的 backbone，用 persona-agnostic sim 数据建立共享协作常识，再为每位参与者独立学习总体 user bias 与 active-condition preference delta，并在固定 probe states 上验证条件化偏好排序改变是否伴随任务能力保持。
