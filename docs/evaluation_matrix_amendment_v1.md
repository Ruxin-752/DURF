# 统一评估协议修订提案 v1.2：Probe 指标失效诊断与 Evaluation Matrix 修订

状态：**提案（未冻结；v1.1 主体已实现并通过测试）**　|　日期：2026-08-27　|　对应冻结文档：`docs/DURF_Ours_vs_Linguistic_统一评估协议_冻结版.docx` v1.0

> **v1.2 相对 v1.1 的变化**：接受两条方法边界修正（§1.1），据此新增 latent–expressed gap 诊断；固定 Ours top-1 的平局处理规则（§3.6）；补充 runtime 诊断的样本单位说明、孤儿 update 统计与题库校验器的词表检查。实现状态见附录 C。

> 本提案**不改变任何已冻结的方法口径**：Ours 的 event/condition 归因链路、Linguistic 的
> reference type / sentiment / feature grounding / Bayesian 更新链路、以及"两者的 F1 不得
> 混为同一指标"这一约束，全部保持不变。本文修改的只有**评估侧**：probe 题库的构造方式、
> 主指标的选取与统计口径、以及 `durf/evaluation/` 的实现细节。

---

## 0. 摘要

统一评估层（`durf/evaluation/`）的架构方向正确：两种方法各自完成推断后再归一化到相同的
feedback / probe / episode 记录，由同一套函数计分。ID 约定、缺文件容错、gold 与在线标签的
分离都经受住了真实数据检验。

但在真实 session 上实跑后发现：**C 层（偏好学习）的 probe 主指标目前不成立**。

`probe_pairwise_accuracy = 1.0` 这个数字，实际是在 177 条 probe 记录中筛出的 31 条
（17.5%）上算出来的；这 31 条中 27 条来自同一个探针；而该 session 中全部 575 个候选的
`hu_score` 恒为 0——也就是说这个"偏好学习准确率"完全来自底座任务规则的得分，与 Hu、与
Linguistic 的权重更新无关。H0 考同一份卷子同样会得到 1.0。

根因不在评分函数，而在**题目**：probe 评估的是"重排序器能否把正确候选排到前面"，但候选池
是由一棵带提前 return 的规则树生成的，**57% 的步骤候选池只有一个元素**，重排序器无从发挥。

本提案给出：(1) 问题清单与证据；(2) 冻结 probe 的字段规范与出题准则；(3) 修订后的
Evaluation Matrix；(4) 统计口径；(5) 代码改动清单；(6) 执行顺序与验收门槛。

---

## 1. 背景：Hu 与 Linguistic 到底作用在哪一层

Agent 每步决策分三段：

```
① 规则生成候选 subgoal
        ↓
② 打分：final_score = task_score + λ · hu_score      ← Ours 与 Linguistic 都只作用在这里
        ↓
③ 取最高分执行
```

`durf/baseline/collect_rule_teacher_dataset.py::generate_candidate_subgoals` 不是"枚举所有
可行选项"，而是一棵**带提前 return 的规则级联**：

```python
if 手持番茄/洋葱: return [PUT_X_IN_POT]        # 池大小 = 1
if 手持盘子:      return [PICKUP_SOUP]         # 池大小 = 1
if 手持汤:        return [SERVE_SOUP]          # 池大小 = 1
```

`outputs/human_ai_sessions/20260722_211635`（800 步）实测候选池大小分布：

| 候选数 | 步数 | 占比 | 含义 |
|---|---|---|---|
| 1 | 456 | 57% | **没有选择可做**，任何重排序都不改变行为 |
| 2 | 294 | 37% | 二选一 |
| 3 | 50 | 6% | 三选一 |

**推论（方法能力边界，需在论文中正面陈述）：**
Hu 与 Linguistic 都只能重排规则树愿意提出的选项。用户表达的任何指向"当前不会被生成的
subgoal"的偏好，**既学不到也测不出**。该天花板对两种方法对称，因此不破坏公平性，但会
显著压缩两者的可区分度。

### 1.1 两条方法边界（v1.2 修订）

1. **冻结 probe 只解决「如何公平测量」，不会扩大实时 agent 的候选能力。** 换题库不会让
   规则树多提出一个 subgoal；候选生成的局限需要单独立项处理。
2. **候选池里没有该 subgoal 时，Hu 仍可能已经更新了内部权重。** Hu 的训练样本来自反馈
   归因产生的 pairwise 标签，与运行时候选池无关，因此受阻的是**表达**而非**学习**。
   准确表述为：**「学到的偏好无法转化为可观察行为」**，而不是「学不到」。
   （v1.1 §1 的推论措辞据此修正。）

第 2 条带来一个可测量的量：对每条「gold preferred 从未进入过候选池」的偏好，离线用
`Hu_user` 直接给该 subgoal 打分。若 Hu 分数确实抬高而行为无变化，结论就是「方法学到了、
架构不让它表达」——责任归于候选生成器，而不是偏好模型。该量作为诊断指标写入 §3.2。

---

## 2. 问题清单

按严重度分级。P0 = 正式实验前必须解决，否则结论不成立；P1 = 影响结论强度或公平性；
P2 = 工程与可维护性。

### P0-1　probe 有效样本量被严重高估，且缺失非随机

`outputs/human_ai_sessions/20260722_211635` 实跑结果：

| 量 | 数值 |
|---|---|
| 原始 `probe_hits` | 346 |
| 折叠后 occurrence | 177 |
| 候选池大小为 1 的 occurrence | 116（66%） |
| 候选池为空的 occurrence | 4 |
| **gold preferred 与 rejected 两端都有分数（可评）** | **31（17.5%）** |
| 其中来自 `useful_object_adjacent_ai_should_consider_pickup` | 27（87%） |
| coordination 域 occurrence / 其中可评 | 4 / **0** |

`comparison_metrics.csv` 只导出 `probe_pairwise_accuracy`，**未导出 `records` 与
`pairwise_evaluable`**。读者看到的是一个没有分母的 1.0。

缺失并非随机：候选是否被生成与情境强相关，因此剩下的 31 条不能代表整体。

### P0-2　可评的题目是送分题，指标已在天花板

31 条可评记录 accuracy = 1.0，mean margin = 64.19。而该 session 中
**575 个候选的 `hu_score` 全部为 0**，`final_score = task_score + λ·hu_score` 意味着
这 64.19 分全部来自底座任务规则。

后果：F0 → F5/F10/F15 的 learning curve 在数学上不可能出现增量；H0、Ours、Linguistic
三者会得到并列的 1.0。

### P0-3　探针触发条件与候选生成器定义错配

`pot_ready_ai_should_get_dish` 触发 26 次，**`GET_DISH` 一次都未出现在候选池中**
（17 次候选池只有 `WAIT`）。

原因：探针的 `required_conditions` 用的是 `pot_cooking_or_ready`（锅在煮**或**已好），
而生成器只在 `ready_pots`（锅**已好**）时才构造 `GET_DISH`。锅仍在烹煮时探针即触发，
此时正确答案在物理上不可被生成，该题恒为不可评。

需要对全部 8 个 runtime 探针逐条核对：**触发条件必须蕴含"gold preferred 在该状态下可被生成"**。

### P0-4　coordination 决策域在主指标上完全缺席

`human_path_conflict_ai_should_yield` 仅触发 4 次，且候选池中 `YIELD` 与
`CONTINUE_CURRENT_SUBGOAL` 均不存在——在 `ai_mode=subgoal_executor` 下
`coordination_decision` 未落盘，`probe_candidate_pool` 只能返回空池。

研究设计中的两个决策域，有一个对 C 层主指标贡献为 0。

### P1-1　pairwise margin 不可跨方法比较，且可被超参操纵

Ours 的 margin 单位是 `task_score + λ·hu_score`；Linguistic 的是 comfort weights 与
feature 的内积。两者量纲无关。更关键的是：**只要调大 `hu_lambda`，Ours 的 margin 与
margin gain 就会变大，而候选排序一个都没变**。

冻结文档将 "Pairwise margin" 列为「主」指标，`comparison_metrics.csv` 也将两方法的
`probe_mean_margin` 并列导出，构成一次无意义比较的邀请。

> **更新（2026-09-03）**：这段描述的加法混合，在 Ours 的 **task 层**已经废弃——现在是
> ε-约束分层满足式，`task_score` 和 `hu_score` 不再相加，`final_score` 就是
> `task_score` 本身，超参 `hu_task_tolerance` 调大调小都不会人为放大 margin（因为已经
> 没有一个叫 margin 的合成量）。这条问题对 task 层已经结构性解决。**Coordination 层还
> 没有改**，仍然是 `coordination_prior + hu_coordination_lambda · Hu_coord`，本节的诊断
> 对 coordination 域原样成立。评估层（`durf/evaluation/`）里是否还有独立计算的 margin
> 指标受同样问题影响，未重新核实，不要假设这条已经在评估侧也修好了。

### P1-2　gold top-1 对 Ours 有系统性偏袒

`_gold_feedback_metrics` 中，Ours 的 `predicted_preferred` 是该条反馈**全部** pairwise
标签的 preferred 集合（真实数据常含 2 个，如 `["PUT_TOMATO_IN_POT","GET_DISH"]`），命中
判定为 `gold ∩ predicted ≠ ∅`；Linguistic 只有单个 `after_subgoal`。

这是"多选 vs 单选"打同一个 top-1 分数。代码中的 note 仅覆盖 `exact_pair`，未覆盖此项。

### P1-3　Linguistic 更新匹配存在强制兜底，会虚高转化率

`_match_linguistic_updates` 中：

```python
elif unused:
    chosen = min(unused)      # 文本、step、episode 全不匹配时仍强行绑定
```

被强绑的记录会带着 `status="updated"` 进入 conversion rate 与 gold top-1，造成静默错配。

### P1-4　learning curve 按文档命令跑不出来

文档 §6 让每个 checkpoint 各建一个 `--output-dir`（`ours_f10`），而 `_learning_gain`
只在**同一个 run 目录内**跨 checkpoint 计算。照文档操作 `learning_gain` 恒为 `null`。

### P1-5　公平性审计漏检验收标准明确要求的三项

验收表要求「probe/checkpoint 均匹配」，但 `_fairness_audit` 未检查：

- 两方法 `probe_id` 集合是否一致；
- 每条 probe 的候选集合是否一致；
- checkpoint 标签是否一致。

此外 `raw_feedback_budget` 使用严格相等，在自由聊天场景下几乎必然失败；而判定顺序规定
公平性不过即不得解释方法优劣——等于整套比较被一个不可能满足的检查卡死。

`_unique_or_list` 对多 session 的顺序敏感（`[A,B]` vs `[B,A]` 判为不一致）。实跑中
`agent` 键缺失导致该项 `pass: null`，在报告中不够显眼。

### P1-6　统计单位为 session 池化，无法支撑被试内设计

所有指标跨 session 池化，无 per-participant 记录、无 n、无置信区间。后果：

- conversion rate 被话多的参与者主导；
- 问卷跨人直接平均，丢失配对信息；
- 「任务能力保持」只有点估计 delta，没有非劣性边界，无法判定。

冻结文档要求 margin「报告均值与置信区间」，实现中未提供 CI。

### P2-1　双重去重，occurrence 定义是两个阈值的交互产物

`detect_probe_hits` 已有 `min_gap_steps=3`（保留间隔 ≥3 的命中），
`_collapse_runtime_probe_hits` 又对该已稀疏化的流按 `≤3` 折叠一次（346 → 177）。
两层同常数去重，语义不清；真正的"一次 occurrence"应由**原始轨迹上连续命中的区间**定义。

### P2-2　已有的可用性标志被统一层丢弃

`probe_hits.jsonl` 的 `evaluation` 块**本来就记录了** `preferred_available` 与
`evaluation_unavailable`，是 normalizer 未透传。这是成本最低的一处修复。

### P2-3　多个字段在真实数据上恒为空

| 字段 | 实测 | 原因 |
|---|---|---|
| `decision_levels`（Ours） | `[]` | 真实 `hu_subgoal_preferences.jsonl` 无 `decision_level` 字段；Linguistic 端却硬编码 `["task"]` → feedback 层的 task/coordination 拆分完全丢失 |
| `researcher_validated_count` | 0 | 真实 prefs 无 `label_status` |
| `median_update_latency_ms` | `null` | 两端都不写 latency，而文档将 update latency 列为指标 |
| `preference_source` / `review_decision` / `use_for_hu_training` | `None` | provenance 未写这些键 |

### P2-4　文档要求但未实现的项

- conversion rate 的**拒绝原因分布**（Linguistic 有 `rejected_duplicate` /
  `rejected_input_confidence` / `rejected_low_confidence` / `rejected_ungrounded` 四类，
  现全被压成一个布尔）；
- margin 的**置信区间**；
- probe 缺失时的**显式 `unavailable`**（现仅为 `null`，CSV 中即空格，易被误读为 0）。

### P2-5　其他

- `h0` 归一化复用 `_normalize_ours_feedback`，若 H0 session 目录存在 attribution 文件，
  `confidence` / `target_event` / `condition_features` 会泄漏进 H0 记录；
- `_collapse_runtime_probe_hits` 硬编码 gap = 3，coordination 域评估节奏未必相同；
- `feedback_id` 不含 session 前缀，跨 session 汇总 gold 时存在碰撞风险；
- 单元测试仅覆盖 happy path，未覆盖兜底匹配、折叠阈值、h0 路径、probe 不可用等分支；
- `durf/evaluation/`、`data/evaluation/`、两份文档、生成脚本在 git 中均为未跟踪状态
  （`??`），且 docx（04:48）早于 md 与 `common_evaluation.py`（04:51）。

---

## 3. 改进方案

### 3.1 冻结 Probe 规范（本提案核心）

**原则：把 probe 从"运行时碰巧遇到的状态"改为"人工设计、候选集写死的考题"。**
runtime `probe_hits` 降级为行为证据，不再进入 C 层主指标。

Linguistic 分支的 `data/subgoal_probe_states.json` 已经具备正确形态，可直接作为模板：

```json
{
  "probe_id": "SG01_do_not_take_dish_human_wants",
  "context": { "pot_status": "ready", "agent_holding": null,
               "human_intent": "pick_dish_then_serve" },
  "feasible_subgoals": ["GET_DISH", "WAIT"],
  "expected_subgoal": "WAIT"
}
```

其三个关键性质，恰是 Ours runtime 探针所缺：候选集由题目指定；正确答案与 task 贪心
**相反**（有判别力）；并配有"该上就上"的控制题（SG04/SG05），防止模型退化为一味顺从。

#### 3.1.1 `common_probe_scores.jsonl` 字段规范

在现有 `data/evaluation/common_probe_scores.example.jsonl` 基础上新增三个字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `probe_id` | ✓ | 全局唯一 |
| `decision_level` | ✓ | `task` / `coordination`，两域都必须有题 |
| `candidate_set` | ✓ | **题目指定的候选集合，长度 ≥ 2**；两方法必须在完全相同的集合上打分 |
| `candidate_scores` | ✓ | 键集合必须等于 `candidate_set` |
| `gold_preferred` / `gold_rejected` | ✓ | 必须 ⊆ `candidate_set`，否则该 probe 非法 |
| `discriminative` | ✓（新增） | H0 在该题上是否答**错**。`true` = 判别题，`false` = 控制题 |
| `probe_class` | ✓（新增） | `suppression`（该忍住）/ `promotion`（该主动）/ `control`（不该被带偏） |
| `condition_features` | ✓ | 与 Ours 的 condition 词表一致，供 condition_delta 使用 |
| `selected` | ✓ | 方法在该题上的最终选择 |
| `checkpoint` | ✓ | F0 / F5 / F10 / F15 |

#### 3.1.2 出题准则（硬约束）

1. **判别题占比 ≥ 60%**：`discriminative=true`（H0 会答错）的题必须占多数，否则主指标
   必然天花板；
2. **控制题占比 ≥ 30%**：正确答案与 task 贪心一致的题，防止"永远 WAIT"式退化在判别题上
   拿满分；
3. **两域各自够量**：coordination 域 ≥ 8–10 条（YIELD / CONTINUE / 让路 / 不抢同一食材）；
4. **总量 30–40 条**，每条只计一次，不按帧重复；
5. **每条 ≥ 2 个候选**，且 gold 两端都在候选集内。

#### 3.1.3 构建期校验

新增 `durf/evaluation/validate_probe_set.py`，硬性检查上述 5 条准则 +
`gold ⊆ candidate_set` + `candidate_scores.keys() == candidate_set`。
**校验不通过不得进入正式实验。**

#### 3.1.4 三个条件如何在同一题上打分

| 条件 | 打分方式 |
|---|---|
| H0 | λ=0 / 权重不更新，对 `candidate_set` 打分。**同时用于标注 `discriminative`** |
| Ours | 绕过 `generate_candidate_subgoals`，直接把 `candidate_set` 交给 `Hu_user` 打分（`durf/hu/score_subgoals.py` 已具备该能力） |
| Linguistic | 用其 comfort weights × features 对同一批候选打分（`evaluate_subgoal_reranking.py` 已是该逻辑） |

三者输出格式一致，交由统一 evaluator 计分。

#### 3.1.5 两层题库与 gold 的来源（v1.2 新增）

题库分两层，同一个文件里用 `layer` / `gold_source` 区分：

| 层 | `gold_source` | gold 谁定 | 作用 |
|---|---|---|---|
| `consensus` | `researcher_consensus` | 研究者事先写死，全体参与者相同 | 任务能力保持；控制题防止退化成一味等待/让路 |
| `preference_conditioned` | `participant_elicited` | **参与者本人在问卷上选** | 个性化主张的证据来源 |

只有共识层的题目，研究者才有资格写 gold。**偏好条件层没有客观正确答案**：同一局面，
希望 AI 主动的人和希望 AI 别添乱的人答案相反。若整个题库只有共识层，即使 Ours 全对，
也只能说明 agent 变得更懂通用协作，**不能支撑"学会了这个人的偏好"**。

**gold 必须来自模型训练不到的通道。** 参与者游戏中说的自然语言是模型的输入；probe 的 gold
来自另一个场合、看着具体局面做的独立选择。两者对上才叫理解了这个人。若 gold 是从模型学习
用的同一批话里推出来的，就是自我循环。

**采集时点（关键）：**

```
① 讲解 + H0 熟悉局
② ★ 填偏好问卷 —— gold 在此定死，之后不得修改
③ 条件 1 / 2 / 3（counterbalanced）
④ 每条件后填 Likert 问卷
⑤ 结束后重填同一份问卷 —— 不作为 gold
```

② 放在熟悉局之后（没玩过的人预测不准自己的偏好），但必须在所有处理条件之前：**放在实验
之后会被 AI 刚才的行为塑造**，且不同条件产生不同污染，gold 会变成条件的函数，三条件失去
可比性；适应最好的方法还会因此赢两次。⑤ 的重填只用于偏好稳定性检验，前后不一致本身可以
作为独立发现报告，但绝不能进入分母。

`discriminative` 对该层是**逐参与者**的（`h0_selected ∉ 该参与者的 gold_preferred`），
因此在冻结文件里保持 `null`，`h0_selected` 在 S2 填入。

#### 3.1.6 问卷实现

| 文件 | 作用 |
|---|---|
| `scripts/build_frozen_probe_set.py` | 题库唯一来源，含每题的中英文问卷措辞 |
| `data/evaluation/frozen_probe_set_v1.draft.json` | 生成结果：33 题（共识 13 / 偏好条件 20；coordination 12；控制 10；注意力检查 3） |
| `scripts/generate_participant_probe_questionnaire.py` | 由题库生成参与者问卷，`--seed` 可按参与者随机题序 |
| `docs/participant_probe_questionnaire_v1.docx` | 中英双语问卷，23 题（20 偏好条件 + 3 注意力检查） |
| `data/evaluation/probe_questionnaire_responses.template.csv` | 录入模板：`participant_id, round, probe_id, choice, notes` |
| `scripts/generate_evaluation_matrix_docx.py` | 生成评估矩阵 Word 文档 |
| `docs/DURF_Evaluation_Matrix_v2.docx` | **评估矩阵正式稿（v2.0，10 页）**。按导师的三侧框架重组：AI 侧（M1 反馈处理 / M2 反馈解释与偏好学习 / M3 行为落地）、人侧（U1–U5 主观体验）、人机共同侧（T1 任务能力 / T2 协作流畅度）。22 张指标卡，每张含「测量工具」字段；另含测量工具汇总表与量表出处附录 |
| `docs/DURF_Evaluation_Matrix_v1.2.docx` | 已被 v2.0 取代，保留备查 |

问卷措辞与题库放在同一份数据里，Word 表和评分器不可能走样。每道偏好条件题都必须提供
**"两种都行"** 选项：强迫二选一会制造假 gold，选了该项的题对该参与者不计分，报告里需
显示每题的实际计分人数。

### 3.2 修订后的 Evaluation Matrix

> 下表是变更记录（相对 v1.0 冻结版改了什么）。**完整的指标定义、计算口径、分母与禁止
> 比较项，以 `docs/DURF_Evaluation_Matrix_v1.2.docx` 为准**，由
> `scripts/generate_evaluation_matrix_docx.py` 生成。

| 维度 | 指标 | 现状 | 修订 |
|---|---|---|---|
| A. 输入与转化 | Feedback-to-update conversion rate | 主 | 保留；**补拒绝原因分布**（按 `update_status` 分类计数） |
| B. 反馈解释 | Gold preferred/top-1 accuracy | 主 | 保留；**Ours 改取单个 top-1**，与 Linguistic 的 `after_subgoal` 对等；**平局按 §3.6 确定性打破，不得判空** |
| B. 反馈解释 | exact pair accuracy | 主 | **降为方法内诊断**（Linguistic 仅在发生 policy switch 时才有 rejected） |
| B. 反馈解释 | event/condition 与 feature grounding 诊断 | 诊断 | 不变（保持分开报告，禁止合并为同一 F1） |
| C. 偏好学习 | Held-out pairwise accuracy | 主 | 保留为主，**但限定在冻结 probe 上**；runtime probe 降为诊断 |
| C. 偏好学习 | **Discriminative-subset accuracy** | — | **新增，核心主指标**。仅在 H0 答错的题上计算，反映净学习效应 |
| C. 偏好学习 | **Control-subset accuracy** | — | **新增，主**。必须与上一项**成对报告**，防止一味顺从的退化解 |
| C. 偏好学习 | Pairwise margin | 主 | **降为方法内诊断**（量纲不可比，且随 λ 变化） |
| C. 偏好学习 | **Normalized preference score** | — | **新增，替代 margin 做主指标**：preferred 在候选集内的排名，或 softmax 概率。无量纲、跨方法可比 |
| C. 偏好学习 | **Latent–expressed gap** | — | **新增，诊断**（v1.2）。对 gold preferred 从未进入候选池的偏好，离线取 `Hu_user` 分数变化，区分「没学到」与「学到但表达不出」 |
| C. 数据完整性 | **Probe evaluable rate** | — | **新增，硬门槛**：< 80% 时不得报告 accuracy，只能标 `unavailable` |
| C. 样本效率 | F0 → F5/F10/F15 gain | 主 | 保留，**仅在 discriminative 子集上计算** |
| D. 实际行为 | Preference adherence | 主 | 保留；须与 evaluable rate 同时报告 |
| D. 实际行为 | Repeated negative feedback rate | 主 | 保留；建议改为条件概率（"同一 `issue_id` 首次反馈后再次出现的概率"）并报告时间间隔 |
| E. 任务保持 | Reward / success 相对 H0 | 主 | 改为**非劣性判定**：预先约定 margin + participant 层 bootstrap CI，不用裸点估计 |
| F. 人因 | 四项 Likert | 主 | 改为**配对**分析（被试内），报告成对差值与 CI |
| G. 系统效率 | Update latency / failure rate | 次 | 两端补写 `update_latency_ms`；failure rate 由拒绝原因分布派生 |

**判定顺序新增前置条款：**

> 第 0 步：检查 probe evaluable rate 与 discriminative 题占比。**不过关则不解释任何 C 层结果。**

### 3.3 统计口径

1. **统计单位改为 participant**：所有比率型指标先按参与者×条件计算，再聚合；
   `comparison_metrics.csv` 增加 `participant_id`、`n` 两列，并输出一份 long 格式
   `comparison_metrics_by_participant.csv` 供 R/SPSS 做配对检验；
2. **置信区间**：accuracy / margin / adherence 全部给出 participant 层 bootstrap 95% CI；
3. **独立性**：同一 episode 内重复触发的 runtime probe 不作为独立样本；冻结 probe 天然
   每条计一次；
4. **非劣性**：任务保持采用预先约定的等效边界，报告 delta 的 CI 是否落在边界内。

### 3.4 代码改动清单

按实施顺序，标注对应问题编号。

| # | 改动 | 文件 | 对应 |
|---|---|---|---|
| 1 | 透传 `preferred_available` / `evaluation_unavailable` 到 common probe record | `common_evaluation.py::_normalize_probes` | P2-2 |
| 2 | `_probe_metrics` 增加 `evaluable_rate`；`probe_data_available` 显式布尔 + 原因 | `common_evaluation.py` | P0-1 / P2-4 |
| 3 | CSV 增列：`probe_records`、`probe_evaluable`、`probe_evaluable_rate`、`n`、CI | `_write_comparison_csv` | P0-1 / P1-6 |
| 4 | 新增 `discriminative` / `control` 子集指标 | `_probe_metrics` | 3.2 |
| 5 | margin 移出 CSV 主表，移入 `methods[m].diagnostics` | `build_comparison` | P1-1 |
| 6 | 新增 normalized preference score（排名 / softmax） | `_probe_metrics` | P1-1 |
| 7 | Ours `predicted_preferred` 取单个 top-1（按 pairwise 胜场或分数排序） | `_normalize_ours_feedback` | P1-2 |
| 8 | 删除 `_match_linguistic_updates` 的 `min(unused)` 兜底，改为 `unmatched` 并计数 | `common_evaluation.py` | P1-3 |
| 9 | conversion rate 增加 `rejection_reasons` 分布 | `_summarize_normalized` | P2-4 |
| 10 | 公平性审计增加 probe_id 集合、候选集合、checkpoint 一致性断言；预算改容差；`_unique_or_list` 改排序集合比较 | `_fairness_audit` | P1-5 |
| 11 | h0 独立归一化路径，不复用 Ours | `normalize_run` | P2-5 |
| 12 | 取消第二层折叠；occurrence 改由探测器记录连续命中区间 | `probe_state_detector.py` + `common_evaluation.py` | P2-1 |
| 13 | 逐条核对探针触发条件与候选生成器的可生成性 | `probe_state_detector.py::DEFAULT_PROBES` | P0-3 |
| 14 | coordination head 落盘确认（`coordination_decision` 必须写入 trajectory） | `play_with_baseline.py` | P0-4 |
| 15 | 两端补写 `update_latency_ms` | Ours provenance / Linguistic trace | P2-3 |
| 16 | `feedback_id` 增加 session 前缀 | `_feedback_id` | P2-5 |
| 17 | 新增 `validate_probe_set.py` | `durf/evaluation/` | 3.1.3 |
| 18 | 补测试：兜底匹配、折叠、h0、probe 不可用、gold 单/多选 | `test_common_evaluation.py` | P2-5 |
| 19 | 全部新增文件纳入 git；重新生成 docx 后与 md 一并提交打 tag | — | P2-5 |

### 3.5 明确不改动的部分

- Ours 的 event / condition / pairwise 归因链路与 `Hu_user = Hu_general + user_bias + condition_delta`；
- Linguistic 的 feedback/reference type、sentiment、feature grounding、Bayesian 更新；
- "两种方法保留各自语义处理，统一发生在推断之后"这一原则；
- "Ours 的 event F1 与 Linguistic 的 feature-grounding F1 分开报告、禁止直接比较"这一约束；
- 相同 backbone / 地图 / seed / horizon / 反馈预算 / counterbalancing 等公平性输入约束。

### 3.6 Top-1 平局处理规则（v1.2 新增）

Linguistic 的 `after_subgoal` 是其 scorer 已经 argmax 解过的单一结果；若让 Ours 的平局悬空
（返回空 top-1），Ours 就会在「归因产出两个等权 preferred」时被自动判错——这不是对称化，
而是反向惩罚。实测旧实现下 **8 条被接受反馈中有 4 条（50%）被判空**。

规则固定为：

1. pairwise 票数（preferred +1、rejected −1）；
2. 票数相同时，取 attribution provenance 中 `preferred_subgoals` 的**原始顺序**，即 LLM
   自己的排序——**不使用研究者 gold，也不使用运行时分数**，避免信息回流；
3. 仍相同时取字典序，保证可复现。

平局必须留痕：记录 `top1_status ∈ {single, tie_broken, no_output, not_applicable}` 与
`method_trace.top1_tie` / `top1_tie_break`。`preference_output_coverage` 只反映「方法完全没有
输出」，**不得把平局计为无输出**；汇总层报告 `top1_status_distribution` 与 `top1_tie_count`。
若某方法 tie 率异常高，应作为其归因粒度的诊断结果讨论，而不是隐藏。

---

## 4. 执行顺序与验收门槛

| 阶段 | 内容 | 通过门槛 |
|---|---|---|
| S1a | 按 3.1 编写题库（✅ 已完成：33 题） | `validate_probe_set.py` 结构检查通过 |
| S1b | 在模拟器中重建每题 `context`，验证候选可执行 | `feasibility_verified` 全为 true |
| S1c | 生成参与者偏好问卷（✅ 已完成） | 每题有中英文措辞与「都行」选项 |
| S2 | 用 H0 对全部 probe 打分：共识层标注 `discriminative`，偏好条件层填 `h0_selected` | 控制题 ≥ 30%，coordination ≥ 8 条；判别题 ≥ 60% **只在共识层的非控制题上计算**（控制题按定义就是 H0 会答对的，放进分母会让两个门槛互相矛盾）。偏好条件层的判别率逐参与者报告，不在冻结时设门槛 |
| S3 | 实施 3.4 表中 #1–#11、#16–#18 | 单元测试通过；CSV 含分母与 n |
| S4 | Linguistic 端导出 `common_probe_scores.jsonl` | probe_id 集合与候选集合与 Ours **完全一致** |
| S5 | 实施 #12–#15，确认 coordination 落盘 | coordination 域 evaluable rate > 0 |
| S6 | 跑 H0 / Linguistic / Ours 三条件真人数据（含 ② 的问卷采集与 gold 实例化） | 公平性审计全绿；probe evaluable rate ≥ 80%；注意力检查通过 |

**S1、S2 必须排在"导出 Linguistic frozen probe scores"之前。** 否则流程会顺利跑通并产出
一份三个 1.0 的表格，而该表格说明不了任何事情。

---

## 附录 A：复现命令

```powershell
python -m durf.evaluation.normalize_run `
  --method ours `
  --session outputs\human_ai_sessions\20260722_211635 `
  --checkpoint F10 `
  --output-dir outputs\formal_evaluation\_audit_ours
```

## 附录 B：本次审计的实测数据（session `20260722_211635`）

```
feedback: raw=12  accepted=8  conversion_rate=0.667
          researcher_validated=0   median_update_latency_ms=null
probe:    records=177  pairwise_evaluable=31  accuracy=1.0  mean_margin=64.19
          候选池大小分布（177 条）: 0→4, 1→116, 2→52, 3→5
          可评 31 条按探针: useful_object_adjacent=27, human_waiting=3, pot_needs_tomato=1
          pot_ready_ai_should_get_dish: 26 次触发, GET_DISH 入池 0 次, 仅 WAIT 17 次
          coordination: 4 次触发, 可评 0
          全部候选 hu_score 非零个数: 0 / 575
task:     episodes=1  mean_episode_reward=240.0  success_rate=1.0  steps=800
轨迹层候选池大小: 1→456 (57%), 2→294 (37%), 3→50 (6%)
fairness: agent 键缺失 → pass=null
```

---

## 附录 C：实现状态（截至 v1.2）

已在 `durf/evaluation/` 落地并通过测试的项：

| 项 | 状态 | 说明 |
|---|---|---|
| runtime probe 降级为诊断 | ✅ | 主 probe 块在无 frozen set 时报 `probe_data_available: false` 并给出原因 |
| `frozen_probe_set` / `common_probe_scores` 分离 | ✅ | 题目与分数分开，前者由校验器把关 |
| discriminative / control 子集指标 | ✅ | `discriminative_accuracy` / `control_accuracy` |
| evaluable rate + 分母 | ✅ | 进入 summary 与 CSV |
| rank 归一化 preference score | ✅ | `normalized_preference_score` |
| margin 降级 | ✅ | 改名 `mean_pairwise_margin_diagnostic`，移出跨方法主表 |
| 删除 Linguistic 强制兜底匹配 | ✅ | 无匹配即 `missing_update` |
| 删除 runtime probe 第二层折叠 | ✅ | 仅保留探测器的 `min_gap_steps` |
| `preferred_available` / `unavailable` 原因保留 | ✅ | 透传自 `probe_hits.jsonl` |
| 公平性审计增加 probe/候选/checkpoint 检查 | ✅ | `frozen_probe_ids` / `frozen_candidate_sets` / `frozen_checkpoints` |
| 反馈预算改容差 | ✅ | 20% 或 ±2，标注 `hard_requirement: false` |
| `feedback_id` 加 session 前缀 | ✅ | `_scoped_feedback_id` |
| 拒绝原因分布 | ✅ | `update_status_distribution` |
| h0 独立归一化路径 | ✅ | 不再复用 Ours 的 attribution 字段 |
| **top-1 平局确定性打破** | ✅ v1.2 | §3.6；旧实现会把 50% 的已接受反馈判空 |
| **孤儿 update 统计** | ✅ v1.2 | `unmatched_update_count`：日志里存在但无反馈认领的更新 |
| **runtime 诊断样本单位说明** | ✅ v1.2 | 标注 `unit: probe_hit_frames`，禁止当作 C 层样本量引用 |
| **题库 condition 词表校验** | ✅ v1.2 | 校验 `condition_features` 键属于 `condition_features.py` 的词表 |
| **校验器草案模式** | ✅ v1.2 | `--allow-unlabeled-h0` 时数量/配比降为 warning，新增 `freeze_ready` |

尚未实现，按 §4 的阶段推进：

| 项 | 阶段 | 说明 |
|---|---|---|
| 30–40 题冻结题库 | S1 | 当前 `frozen_probe_set_v1.draft.json` 只有 5 题、0 条 coordination |
| 模拟器中重建 context 并验证候选可执行 | S1 | `feasibility_verified` 目前全为未验证 |
| H0 全量打分与 discriminative 标注 | S2 | 决定题库是否达到 60% 判别题门槛 |
| 三方统一 frozen probe scorer | S4 | H0 / Ours / Linguistic 在同一候选集上打分 |
| coordination decision 运行时落盘 | S5 | `ai_mode=subgoal_executor` 下当前不写 `coordination_decision` |
| latent–expressed gap 计算 | S5 | 需要离线加载 `Hu_user` 对未入池 subgoal 打分 |
| participant 级配对统计与 CI | S6 | 目前仍为 session 池化，无 n、无 CI |
| 探测器改为记录连续命中区间 | 可延后 | runtime 已降级为诊断，优先级低 |
| 重新生成冻结版 docx | S2 之后 | 避免「冻结完又改口径」 |
