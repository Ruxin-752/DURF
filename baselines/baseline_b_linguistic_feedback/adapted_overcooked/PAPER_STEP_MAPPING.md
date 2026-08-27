# Baseline B 每一步 ↔ 原论文对应表

> **Route 2 update (2026-08-10):** full reward-vector supervision, nonzero
> trajectory features, 36 reward configurations, and the paper's ten-fold
> teacher/reward holdout rotation are implemented using 12 stable synthetic
> authors. Ten dev-selected models are hash-frozen before a one-time test. The
> network remains the paper's
> `EmbeddingBag(30) -> hidden 128 -> reward vector` architecture, adapted to the
> 53-dimensional Overcooked schema. The reference classifier also reproduces
> the paper benchmark at 87.16% accuracy. See
> [ROUTE2_AUDIT_REPORT.md](ROUTE2_AUDIT_REPORT.md); older text-only
> descriptions below are historical.
原论文：*Learning Rewards from Linguistic Feedback*（Sumers, Ho, Hawkins, Narasimhan, Griffiths, AAAI 2021, arXiv:2009.14715）。

本文件把 `adapted_overcooked/` 里的**每一步实现**和论文里的**对应概念/公式/流程**一一对应，方便审阅“这份 Overcooked 适配是不是忠实复现了论文思路”。

> 所有与原文不同之处、以及"训练前停止"的明确边界，集中记录在
> [`DIFFERENCES_FROM_PAPER.md`](DIFFERENCES_FROM_PAPER.md)。

---

## 0. 总体流程对应

论文的核心思路是：把人类的一句自然语言反馈，转成对「奖励特征权重」的一次更新，最后用这套权重去预测/选择动作。

```text
论文:      linguistic feedback -> (feedback type + sentiment) -> ground to reward features -> update belief over feature weights w -> act / predict
本实现:    feedback text        -> phrase-level f_G + sentiment -> constrained reference subtype -> ground_feedback -> posterior update
```

| 步骤 | 本实现代码 | 原论文对应 |
|---|---|---|
| ① 三类反馈形式 `f_G` | `feedback_form_classifier.predict_feedback_form` | 论文 `Evaluative / Imperative / Descriptive`；同一份 UI 预测控制 Route 1 grounding 与 precision gate |
| ①b 类内指涉细化 | `phrase_reference_classifier.predict_reference_type` | 原作者代码的五类 reference subtype；只可在三类分支内部细化，不能跨类覆盖 |
| ② 情感/极性抽取 | `sentiment_extractor.extract_sentiment` | 论文用情感分析得到反馈的正负 valence（VADER） |
| ③ 反馈落到特征上（grounding） | `overcooked_grounding.ground_feedback` → `observations.build_observations` | 论文把反馈映射到奖励特征 φ 上的“被指涉特征”（reference vector） |
| ④ 高斯信念 + 共轭贝叶斯更新 | `belief_model.GaussianBelief` + `reward_weight_model.BayesianRewardLearner` | 论文 `beliefs.py::MultivariateNormal` + `agents.py::MultivariateNormalLearner` 的共轭高斯更新 |
| ⑤ 采样信念预测偏好动作并评估 | `probe_evaluator.evaluate_probes` + `evaluate_probes_sampled` | 论文 `BaseAgent.execute_trajectories`：采样信念 → 线性打分 → argmax |
| ⑥ 特征表示 φ(s,a) | `probe_states.json` 里各动作的 `features` | 论文中每个动作/状态的特征向量 φ(s,a) |
| ⑦ 泛化评估（留一法） | `evaluate_leave_one_probe_out` | 论文的 held-out / 跨样本泛化评估 |

---

## 1. 逐步一一对应

### 步骤 ① 三类反馈形式与类内指涉细化
- **实现**：`feedback_form_classifier.py` 先逐短语预测论文三类 `f_G`；
  `phrase_reference_classifier.py` + `train_phrase_reference_classifier.py`
  再提供受约束的细粒度 subtype。
  - 按论文标点规则切短语。论文复现保留原预处理和 word TF-IDF；部署模型使用
    raw word 1–2 gram + char_wb 3–5 gram 和 LogisticRegression，预测
    `trajectory / feature / action_spatial / action_behavioral / other`。
- **论文对应**：三类 `f_G` 选择 grounding 路径：Evaluative → trajectory，
  Imperative → action，Descriptive → feature。五类是原作者发布代码里的细化；
  `action_behavioral` 只能细化 Evaluative 的轨迹时间归因，跨类预测会被记录并
  投影回三类默认路径，`other` 仍安全拒绝。
- **当前结果**：在论文原始 982 条人工标签和相同随机行 85/15 协议上复现
  accuracy 87.162%、macro-F1 75.252%；该协议存在重复文本和 task 重叠，只用于
  论文协议复现。v8 综合开发集为 accuracy 87.02%、macro-F1 87.08%；固定且
  无泄漏、但已经查看的 paper-v1 回归为 77.36% / 72.52%。861 条旧 Overcooked
  合成回归集为 92.68% / 92.40%，不是真人外部测试。尚无真人 Overcooked
  金标准，因此真人语言准确率仍未知。

### 步骤 ② 情感 / 极性抽取
- **实现**：`sentiment_extractor.py::extract_sentiment`
  - 输出连续 `sentiment_score ∈ [-1,+1]`，决定更新方向与强度。
  - 正向指令固定强化目标动作；`stop / don't / avoid` 等禁止式指令反向更新
    被禁止动作的特征。
- **论文对应**：论文用**情感分析**得到反馈的正负 valence，用来决定对被指涉特征的权重是加还是减。
- **对齐**：未显式标注的文本使用论文同款 modified VADER 连续分数。

### 步骤 ③ 把反馈落到奖励特征上（grounding）
- **实现**：`overcooked_grounding.py::ground_feedback`，按反馈类型分三条路径：
  - `evaluative` → 取 `trajectory_features`（**最近这段轨迹的特征**）。
  - `imperative` → 取 `target_action` 对应的动作特征（来自 `action_feature_library`）。
  - `descriptive` → `features_from_keywords`（按关键词命中的“被描述行为”特征）。
- **论文对应**：这正是论文三类反馈的差异化处理方式：
  - 评价式：把 valence 归到**刚发生的动作/轨迹**的特征上；
  - 指令式：把 valence 归到**被命令动作**的特征上；
  - 描述式：把 valence 归到**被描述行为**所对应的特征上。
- **对应关系**（见 `MIGRATION_NOTES.md` 里的映射表）：
  - 论文 MDP 特征 `φ(s,a)` ↔ `probe_states.json` 里手写的动作特征
  - 评价式反馈 ↔ 更新最近/给定轨迹特征
  - 指令式反馈 ↔ 落到目标动作的特征向量
  - 描述式反馈 ↔ 按关键词落到被提及行为特征

### 步骤 ④ 高斯信念 + 共轭贝叶斯更新（忠实复现）
- **实现**：`belief_model.py::GaussianBelief` + `reward_weight_model.py::BayesianRewardLearner`
  - 信念 = 特征权重上的**多元正态** `w ~ N(mean, cov)`，先验 `N(0, var=25)`（precision `1/25`），与论文 `PRIOR_MEAN=0 / PRIOR_PRECISION=1/25` 完全一致。
  - 每条反馈由 `observations.build_observations` 转成一次**共轭高斯观测**：把 grounding 得到的特征当作 reference vector `x`，把 `valence×valence_scale` 当作回归目标，观测精度 `precision_scale`。
  - `GaussianBelief.multiply_observation` 做论文实验实际启用的高斯因子乘法：
    - `Λ' = Λ + xxᵀ·p`
    - `mean' = Σ'(Λ·mean + p·(x·x)·valence·x)`
  - 实现使用与上述公式严格等价的 Sherman–Morrison rank-one 形式，避免
    每条反馈反复做 53×53 矩阵求逆。
- **论文对应**：逐行对应实验 active path 的
  `beliefs.py::MultivariateNormal.multiply` 与 learner 的 `belief_state.multiply(obs)`。
- **两种论文变体**（由 `--mode` 切换）：
  - `literal`（`ExpLiteralLearner`）：只更新被提及特征；
  - `pseudopragmatic`（`ExpPseudoPragmatic`）：对**未提及特征**额外加一个负 `pragmatic_valence` 观测（语用蕴含：没被夸的默认是差的），对应 `observations_from_utterance` 的 inverse-reference 分支。
- **默认超参**与论文一致：`valence_scale=30, precision_scale=2, pragmatic_valence=-30, pragmatic_precision=2`。

### 步骤 ⑤ 采样信念预测偏好动作 + 评估（忠实复现）
- **实现**：`probe_evaluator.py`
  - `evaluate_probes`：用**后验均值**给候选动作线性打分并 argmax（`E[value]=φ·E[w]`，确定性口径，用于 probe accuracy / margin / tie）。
  - `evaluate_probes_sampled`：忠实复现论文 `BaseAgent.execute_trajectories`——**从后验采样 n 组权重**，对每个样本 argmax 选动作，统计每个动作的选择概率与期望正确率。
  - 打平（tie）算失败，不用 JSON 顺序破平。
- **论文对应**：论文用学到的奖励（采样式 Thompson 策略）**预测/选出人类偏好的动作**并算 accuracy。
- **对应关系**：论文“原始任务得分” ↔ 这里的 **probe accuracy**（均值口径）与 **expected accuracy**（采样口径）。

### 步骤 ⑥ 特征表示 φ(s,a)
- **实现**：`probe_states.json`（每个候选动作带一个 `features` 向量）、`overcooked_features.json`、`trajectory_featurizer.py`。
- **论文对应**：论文中每个动作/状态的**特征向量 φ**（原论文是颜色/形状特征）。
- **差异**：论文特征针对 colored-shape 任务；本实现换成 Overcooked 协作特征（如 `blocks_human_path`、`respects_human_intent`、`supports_serving` 等），domain 变了但结构一致。

### 步骤 ⑦ 泛化 / 对照评估
- **实现**：`run_baseline_b_pipeline.py` + `evaluate_route1_subgoal.py`
  - `evaluate_leave_one_probe_out`：**留一 probe** 训练，再在被留出的 probe 上评估（防止“背答案”）。
  - 同时报告四条基线：`zero_weights` / `initial_weights` / `learned_weights` / `leave_one_probe_out`。
  - `classification_accuracy`：分类器和标注类型的一致率。
  - Route 1 subgoal 消融同时比较 oracle / inferred、Literal /
    PseudoPragmatic，以及只使用三类反馈中某一类时的 held-out 表现。
- **论文对应**：论文的**留出泛化评估**，以及和“未学习/先验”基线的对照，用来证明反馈确实带来了奖励学习增益。

---

## 2. 端到端调用顺序（对照论文 pipeline）

`run_baseline_b_pipeline.py::run_pipeline` 的实际调用顺序，对应论文一次“反馈→学习→评估”循环：

```text
validate_feedback_examples                # 数据校验（论文数据集准备）
for each feedback:                        # 论文：逐条反馈做贝叶斯更新
    classify_feedback                     # ① 反馈类型
    extract_sentiment                     # ② valence
    ground_feedback                       # ③ 落到特征（reference vector）
    BayesianRewardLearner.update          # ④ 共轭高斯更新 GaussianBelief
evaluate_probes(learned_mean)             # ⑤ 后验均值预测动作
evaluate_probes_sampled(learned_belief)   # ⑤ 采样信念预测动作（execute_trajectories）
evaluate_probes(zero/initial)             # 对照基线
evaluate_leave_one_probe_out              # ⑦ 泛化评估
```

---

## 3. 忠实复现程度（诚实标注）

现在**奖励学习核心（路线1）已忠实复现**论文的贝叶斯 learner：

| 论文做法 | 本实现做法 | 状态 |
|---|---|---|
| 权重 w 的高斯信念，先验 `N(0,25)`，共轭更新 | `GaussianBelief` + `BayesianRewardLearner`，同先验同公式 | ✅ 忠实复现 |
| Literal / PseudoPragmatic 两个 learner 变体 | `--mode literal / pseudopragmatic` | ✅ 忠实复现 |
| 采样信念 → argmax 选动作 | `evaluate_probes_sampled`（对应 `execute_trajectories`） | ✅ 忠实复现 |
| VADER 情感做 valence | `sentiment_extractor.py` 用 NLTK VADER 复刻 `modified_vader_observation`（英语单语） | ✅ 已对齐（见 `DIFFERENCES_FROM_PAPER.md` §1） |
| 学习式推断网络（路线2） | `paper_v5` 语料上做完整 53 维奖励训练；逐折 dev-only 选模，teacher/reward 双轴隔离，checkpoint 与 manifest 冻结后一次性 test | ✅ 合成 full-reward 协议通过；真人 local probe 未通过 |
| 指涉分类/grounding | 论文结构的短语级 TF-IDF+LR 五类分类器；类别概率参与 grounding confidence，低置信更新降权 | ✅ 结构对齐；训练数据仍以合成为主 |
| colored-shape 任务特征 | Overcooked 协作特征 | 换 domain，结构不变 |
| 真实教师-学习者交互数据 | Route 2 正式语料为 83,592 条合成样本（12 authors × 36 configs 的完整覆盖）；另有独立手写 probe | 真实人类语料仍是缺口，见 `MIGRATION_NOTES.md` / `DIFFERENCES_FROM_PAPER.md §9` |

> 论文原始代码保留在 `../original_rewards_repo/` 仅作参考，不被 Overcooked 运行时导入。

**当前冒烟结果**（100 条种子反馈，零均值先验起步，14 个动作级 probe）：
- `literal`：learned probe accuracy 14/14（均值口径与采样口径均 100%）；
- `pseudopragmatic`：learned probe accuracy 14/14；
- 对照：`zero_weights` 0/14（全平局），`initial_weights` 14/14，`leave_one_probe_out` 14/14。
- probe 覆盖 5 类：HumanComfortCoordination / RespectHumanIntent / RecipeCorrectness / ServingReadiness / Safety / Efficiency，含 4 个 `hard` 权衡样本。

---

## 4. 一句话总结

奖励学习核心已复现论文的贝叶斯高斯信念、共轭更新、Route 2 网络结构和十折 teacher/reward holdout。Route 1 live chat 执行“短语五类指涉分类→grounding→来源感知后验更新→subgoal 重排”；Route 2 部署时平均十个完整奖励预测。原始分类器 benchmark 复现为 87.162% accuracy；Route 2 冻结 synthetic test 的变化维 MSE 为 0.025101、最近配置 accuracy 为 99.9883%、偏好敏感行为 majority 为 97.5490%，但真人风格 local probe 只有 1/4。所有 live 决策固定使用 `w · phi`（`lambda=1`）；真人 Overcooked 数据仍是最终缺口。数据来源与诚实边界见 `ROUTE2_AUDIT_REPORT.md` 和 `DIFFERENCES_FROM_PAPER.md §7/§9`。

---

## 5. Overcooked/H0 集成：subgoal 兼容（超出原文范围）

原论文只学奖励、在候选动作上做预测；DURF 里还要把这套奖励接到 H0（8 子目标的任务执行策略）上。做法是**训练与运行共用同一套特征 schema**：

```text
语言反馈 --训练--> 奖励信念 w（over overcooked_features.json）
状态 + 候选子目标 --subgoal_featurizer--> phi(state, subgoal)
在 H0 的“可行子目标”里按 task_score + λ·comfort_score 重排
```

- `subgoal_featurizer.py`：在线算出与学习时同口径的 `phi`（任务特征来自配方/锅进度，协作特征来自人类当前目标）。
- `subgoal_reranker.py`：只在 H0 给的**任务可行**子目标里重排（任务安全下限留给 H0，协作偏好加在上面）。
- 评估：`scripts/evaluate_subgoal_reranking.py`，当前 learned 5/5、hand-authored-prior 5/5、zero 0/5（全平局），说明是**语言反馈**驱动了子目标让位/互补选择。
- 这一层是原论文没有的、面向 DURF 主线（`H_u` + H0）的集成扩展。
