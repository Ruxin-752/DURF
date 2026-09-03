# 研究协议 v2 落实计划

> 对应协议：[research_protocol_freeze_v2.md](research_protocol_freeze_v2.md)  
> 制定日期：2026-08-11  
> 目标：把“共享 persona-agnostic Hu_general + 每位参与者独立三级偏好模型 + probe 评估”落实为可运行、可审计、可复现实验。

本文是协议 v2 的工程执行清单。每完成一个阶段，都必须满足验收标准，才能进入下一阶段。

## 总体顺序

```text
P0 数据与目录冻结
 -> P1 筛选 persona-agnostic Hu_general
 -> P2 冻结 Hu_general
 -> P3 实现三级 per-user Hu
 -> P4 真人反馈转 reviewed pairwise labels
 -> P5 单参与者训练闭环
 -> P6 shadow/apply 运行时
 -> P7 probe 与正式对照
 -> P8 多参与者交付
```

原则：先打通一个参与者的最小闭环；先使用可解释线性模型；先 shadow，再允许 Hu 改变实际行为。

## P0：冻结数据、来源和目录规范

### 协议依据

- 协议第 4 节：数据来源与参与者隔离。
- 协议第 5 节：统一 28 条 condition schema。
- 协议第 7–8 节：pairwise 标签、溯源版与训练版分离。

### 完成方法

1. 为每条数据补充 `source`、`label_status`、`user_id`、`decision_level`、`protocol_version`。
2. 固定目录：

```text
outputs/hu_general/{raw_sim,filtered_training,hu_general.json,metadata.json}
outputs/hu_users/<user_id>/{raw_sessions,reviewed_labels,training_dataset,hu_user.json,metadata.json,evaluation.json}
outputs/hu_evaluation/{probes,comparisons}
```

3. 为 28 条 condition 生成 manifest，记录类型、来源、True/False/None 含义、Task/Coordination 归属和离线可恢复性。
4. dataset builder 加入来源和 user 隔离检查。

### 主要文件

- `durf/feedback_attribution/schemas.py`
- `durf/feedback_attribution/session_converter.py`
- `durf/feedback_attribution/hu_dataset_builder.py`
- `durf/feedback_attribution/condition_features.py`

### 验收标准

- 每条正式样本可追溯到 session、feedback、user、source。
- 不同用户的真人样本不会被自动合并。
- 28 条 condition 字段名稳定并有 manifest。
- 自动归因不会被标记为 `reviewed`。
- 旧 session 缺失字段时使用 `null`，不静默猜测。

## P1：筛选 `Hu_general` 数据

### 协议依据

- 协议第 3.2 节：general 只学习 persona-agnostic 常识。
- 协议第 4.1 节：persona 冲突样本不能混入 general preference head。

### 完成方法

1. 按 `(decision_level, condition_signature, candidate_pair)` 对 sim 样本分组。
2. 检查不同 persona 的 pairwise 方向：

```text
方向一致 -> invariant
方向相反 -> conflicting
只有一个 persona -> unsupported_for_general
```

3. 输出：

```text
hu_general_invariant.jsonl
sim_persona_conflicts.jsonl
sim_general_audit.json
```

4. `Hu_general` 只使用 `invariant` 数据训练。冲突样本保留用于诊断或 persona 对照，不得直接平均。
5. `sim_both_v1` 的 0.92/0.923 重新标注为 synthetic validation，不能直接作为新 general prior 结果。

### 主要文件

建议新增：

- `durf/hu/audit_sim_personas.py`
- `durf/hu/build_general_prior_dataset.py`

### 验收标准

- 每个 general pair 都有跨 persona 一致性记录。
- conflict 样本不会出现在 general 训练文件。
- 报告包含 invariant/conflicting/unsupported 数量和 Task/Coordination 分布。
- 人工抽查 20 条，没有发现相反 persona 被错误平均。

## P2：训练并冻结 `Hu_general`

### 协议依据

- 协议第 3.2 节：共享 general 冻结。
- 协议第 11 节：`Hu_general` 是正式对照条件。

### 完成方法

1. 用 P1 的 invariant 数据训练当前可解释的线性 pairwise model。
2. Task 与 Coordination 分别训练两个 head。
3. metadata 写明：`source=simulated_prior`、`frozen=true`、训练样本数、过滤规则、metrics。
4. 加载 general 后，真人训练只能进入 user adapter，不得覆盖 general 参数。
5. 提供离线命令，输入 condition 和候选集合，输出两个 head 的排序。

### 主要文件

- `durf/hu/subgoal_reranker.py`
- `durf/hu/train_subgoal_reranker.py`
- `durf/hu/score_subgoals.py`

### 验收标准

- fresh process 只加载 `hu_general.json` 即可打分。
- Task/Coordination 候选不会互相混排。
- 同一输入重复评分一致。
- general 文件 hash 在后续用户训练前后不变。

## P3：实现三级 per-user Hu

### 协议依据

- 协议第 3.3 节：`Hu_user = Hu_general + user_bias + condition_delta_user`。
- 协议第 5、9 节：active condition 和 general fallback。

### 固定模型结构

```text
general_score = frozen Hu_general(condition, subgoal)
user_bias = user × subgoal bias
condition_delta = active_condition × user × subgoal weight
final_score = general_score + user_bias + condition_delta
```

三层规则：

1. 全局 `user × subgoal bias`：表示参与者总体风格，只是稳定化层。
2. 用户实际观察到的 condition：只学习该用户真实见过的 condition 修正。
3. 未观察 condition：delta 为 0，继续使用 `Hu_general`。

### 完成方法

1. 扩展 `HierarchicalHu`，保存 `general_head`、`user_subgoal_bias`、`user_condition_weights`。
2. 加载 general 后冻结 `general_head`。
3. 用户训练只更新当前 user 的两层 adapter，并对 condition delta 使用强正则化或较小学习率。
4. 输出分数必须拆成 `general_score`、`user_bias_score`、`condition_delta_score`、`final_score`。
5. 支持从同一 general 创建 `user_A`、`user_B`，且 A 不改变 B 或 general。

### 主要文件

- `durf/hu/subgoal_reranker.py`
- `durf/hu/train_subgoal_reranker.py`
- `durf/hu/score_subgoals.py`
- `docs/hierarchical_hu_runtime.md`

### 验收标准

- 同一 state 能输出三层分数。
- 未观察 condition 的 delta 恒为 0。
- 人工构造相反 condition 时，同一用户可以产生不同排序。
- 训练 user A 后 general 和 user B 文件 hash 不变。
- 只用 user bias 的 ablation 与完整三级模型都能运行。
- 只有 user bias 的模型不能标记为最终 Hu。

## P4：真人反馈转 reviewed pairwise labels

### 协议依据

- 协议第 6 节：event、condition、preference 分工。
- 协议第 7–8 节：LLM 归因、人工 review、训练版/溯源版分离。

### 完成方法

```text
trajectory + language feedback
 -> state facts
 -> candidate events
 -> time window
 -> LLM semantic attribution
 -> provenance record
 -> human review
 -> same-level pairwise label
 -> user-specific training dataset
```

Review 必须确认：时间窗口、event、decision level、preferred、rejected、目标时刻 condition，以及 `use_for_hu_training`。

只有 `reviewed=true` 且 `use_for_hu_training=true` 的样本进入 `hu_subgoal_preferences.jsonl`。完整证据保留在 `hu_attribution_provenance.jsonl` 和 `review_decisions.jsonl`。

### 验收标准

- 每条正式 pairwise 样本都有 user 和 source feedback ID。
- preferred/rejected 属于同一个 decision level。
- 自动归因失败或人工拒绝的记录不进入训练集。
- 训练样本可以反查原始反馈、时间窗口、event 和 condition。
- 抽查 10 条，condition 来自目标事件时刻而不是反馈结束时刻。

## P5：单参与者训练闭环

### 协议依据

- 协议第 4.2 节：参与者独立。
- 协议第 9 节：30–40 条反馈的小样本策略。
- 协议第 15 节：最小可交付闭环。

### 完成方法

1. 先为 `PILOT01` 建立独立目录。
2. 约 20–25 条 reviewed pairwise labels 用于适配；后续 5–10 条用于 probe/test。
3. 不足以划分 test 时，使用按时间顺序的 online learning curve 或 leave-one-feedback-out。
4. 分别训练 `Hu_general + user_bias` 和完整三级模型。
5. 每加入一条 feedback，记录 accuracy、margin、active condition 数量和两类 user 参数变化。

### 验收标准

- 训练目录只包含 PILOT01 真人数据。
- metadata 记录 train/test 时间顺序和 feedback IDs。
- 已观察 condition 上完整模型相对 general-only 有正确排序改善或 margin 改善。
- 未观察 condition 与 general-only 一致。
- 不把训练集准确率写成泛化结果。

## P6：运行时 shadow/apply

### 协议依据

- 协议第 2–3 节：Hu 只重排候选，不替代 executor。
- 协议第 12 节：偏好改变不能造成明显任务退化。

### 完成方法

1. shadow 模式记录 backbone、general、user bias、完整 user Hu 的排序，但不改变 action。
2. 检查排序改变是否发生在相关 condition 下。
3. 用小 lambda 开启 apply。
4. 保留 Safety/Recovery，不允许 Hu 选择非法 candidate、跨域 pair、永久 WAIT 或无限 YIELD。
5. 记录 `selected_by_backbone`、`selected_by_hu`、`changed_by_hu`、condition、scores 和 decision level。

### 验收标准

- shadow/apply 都能启动并完成 session。
- apply 只改变 candidate ranking，不绕过 executor。
- 每次 Hu 改变都能追溯到 condition 和训练标签。
- 没有新增非法动作、永久 WAIT、无限 YIELD 或 executor 崩溃。
- 任务成功率、出餐数和 reward 有明确记录。

## P7：probe 与正式对照

### 协议依据

- 协议第 10 节：离线 probe 与在线 active collection 分开。
- 协议第 11–12 节：对照条件和三类评估指标。

### 完成方法

1. 建立固定 probe state，记录 decision level、condition、candidate set 和预期比较。
2. 至少覆盖：明显挡路、AI 拿关键原料、锅已好且人类拿盘、非冲突继续任务四类场景。
3. 固定比较：

```text
Backbone only
Hu_general
Hu_general + user_bias
Hu_general + user_bias + condition_delta_user
```

4. 每个 probe 输出候选分数、排序、margin、是否被用户适配改变、是否符合 reviewed preference。
5. 另外报告 attribution accuracy、pairwise preference accuracy、task success 和协作指标。

### 验收标准

- 同一 probe 在所有模型条件下输入完全一致。
- 至少一个已观察 condition 上，完整模型相对 general-only 产生预期排序变化。
- 未观察 condition 回退 general-only。
- 任务指标没有明显退化，任何退化都保留在报告中。
- preference、attribution、task 三类结果分开报告。

## P8：多参与者扩展与交付

### 协议依据

- 协议第 4.2 节：真人 preference 不跨用户共享。
- 协议第 11 节：每位用户独立对照。

### 完成方法

1. 对每位参与者独立重复 P4–P7：

```text
user_A -> delta_A
user_B -> delta_B
user_C -> delta_C
```

2. 所有人使用同一个冻结 `Hu_general`，但真人 labels 和 adapters 完全隔离。
3. 汇总时只描述个体差异，不把真人 preference 平均成一个模型。

### 验收标准

- 修改 user A 不改变 user B、general 或其他 artifacts。
- 每位用户都有独立 metadata、训练样本和 evaluation report。
- 汇总报告明确区分 general prior 与 user preference。

## 暂缓事项

P0–P7 完成前不做：大 MLP、DPO/PPO 重新训练、直接修改环境 reward、复杂在线 active dialogue、跨用户真人 preference pooling、大规模新增 condition、尚未固定决策域的 `REROUTE`。

## 最终完成判据

只有同时满足以下条件，才能称为“协议 v2 的最小研究闭环完成”：

1. general 只由筛选后的 persona-agnostic sim 数据训练并冻结。
2. 至少一位参与者有独立的 general、user bias、condition delta 三层模型。
3. 未观察 condition 自动回退 general prior。
4. 真人反馈经过 review，形成同一 decision level 内的 pairwise labels。
5. 完整模型在 probe state 上改变至少一个对应真人反馈的 subgoal 排序。
6. 改变可以由 condition delta 解释，而不是只由 user bias 解释。
7. Backbone、general、user bias、完整 user Hu 四组结果可比较。
8. preference、attribution、task 三类指标均有独立报告。

在此之前，项目只能称为“工程链路开发中”或“pilot feasibility”，不能声称已经证明个体化偏好学习。
