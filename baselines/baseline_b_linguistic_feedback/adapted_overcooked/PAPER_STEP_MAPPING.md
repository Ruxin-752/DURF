# Baseline B 每一步 ↔ 原论文对应表

原论文：*Learning Rewards from Linguistic Feedback*（Sumers, Ho, Hawkins, Narasimhan, Griffiths, AAAI 2021, arXiv:2009.14715）。

本文件把 `adapted_overcooked/` 里的**每一步实现**和论文里的**对应概念/公式/流程**一一对应，方便审阅“这份 Overcooked 适配是不是忠实复现了论文思路”。

> 所有与原文不同之处、以及"训练前停止"的明确边界，集中记录在
> [`DIFFERENCES_FROM_PAPER.md`](DIFFERENCES_FROM_PAPER.md)。

---

## 0. 总体流程对应

论文的核心思路是：把人类的一句自然语言反馈，转成对「奖励特征权重」的一次更新，最后用这套权重去预测/选择动作。

```text
论文:      linguistic feedback -> (feedback type + sentiment) -> ground to reward features -> update belief over feature weights w -> act / predict
本实现:    feedback text        -> classify + extract_sentiment -> ground_feedback           -> RewardWeightModel.update(w)        -> probe evaluation
```

| 步骤 | 本实现代码 | 原论文对应 |
|---|---|---|
| ① 反馈类型分类 | `feedback_form_classifier.classify_feedback` | 论文三类语用反馈：evaluative / imperative / descriptive |
| ② 情感/极性抽取 | `sentiment_extractor.extract_sentiment` | 论文用情感分析得到反馈的正负 valence（VADER） |
| ③ 反馈落到特征上（grounding） | `overcooked_grounding.ground_feedback` → `observations.build_observations` | 论文把反馈映射到奖励特征 φ 上的“被指涉特征”（reference vector） |
| ④ 高斯信念 + 共轭贝叶斯更新 | `belief_model.GaussianBelief` + `reward_weight_model.BayesianRewardLearner` | 论文 `beliefs.py::MultivariateNormal` + `agents.py::MultivariateNormalLearner` 的共轭高斯更新 |
| ⑤ 采样信念预测偏好动作并评估 | `probe_evaluator.evaluate_probes` + `evaluate_probes_sampled` | 论文 `BaseAgent.execute_trajectories`：采样信念 → 线性打分 → argmax |
| ⑥ 特征表示 φ(s,a) | `probe_states.json` 里各动作的 `features` | 论文中每个动作/状态的特征向量 φ(s,a) |
| ⑦ 泛化评估（留一法） | `evaluate_leave_one_probe_out` | 论文的 held-out / 跨样本泛化评估 |

---

## 1. 逐步一一对应

### 步骤 ① 反馈类型分类
- **实现**：`feedback_form_classifier.py::classify_feedback`
  - 用关键词把一句反馈粗分成 `evaluative` / `imperative` / `descriptive`。
- **论文对应**：论文的核心贡献之一，是区分**三种语用类型的反馈**（评价式、指令式、描述式），并对每一类用不同方式更新奖励信念。
- **差异**：论文用**学习到的推断网络（inference network）**来做这件事；本实现用**确定性关键词规则**替代，作为可离线、可复现的近似。

### 步骤 ② 情感 / 极性抽取
- **实现**：`sentiment_extractor.py::extract_sentiment`
  - 输出 `sentiment_score ∈ {-1, 0, +1}`，决定这次更新是“奖励”还是“惩罚”。
  - `desired_action_sentiment`：指令式反馈固定给正号，表示“强化被命令的动作特征”。
- **论文对应**：论文用**情感分析**得到反馈的正负 valence，用来决定对被指涉特征的权重是加还是减。
- **差异**：论文用连续/模型化的情感信号；本实现用离散 ±1 规则。

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
  - `GaussianBelief.update_from_observation` 做贝叶斯线性回归的**共轭更新**：
    - `Λ' = Λ + xxᵀ·p`
    - `mean' = Σ'(Λ·mean + x·valence·p)`
- **论文对应**：逐行对应 `beliefs.py::MultivariateNormal.update_from_observation` 与 `agents.py::MultivariateNormalLearner.update_beliefs`。
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
- **实现**：`run_baseline_b_pipeline.py`
  - `evaluate_leave_one_probe_out`：**留一 probe** 训练，再在被留出的 probe 上评估（防止“背答案”）。
  - 同时报告四条基线：`zero_weights` / `initial_weights` / `learned_weights` / `leave_one_probe_out`。
  - `classification_accuracy`：分类器和标注类型的一致率。
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
| 学习式推断网络（路线2） | `scripts/train_route2.py` 已在 **DeepSeek 合成语料**上训练（text-only，分组 CV 早停）；held-out subgoal accuracy 18/18=100%。严格对齐版 `train_route2_inference_network.py` 仍保留"训练前停止"边界 | ✅ 已训练（合成真值，验证 language generalization） |
| 反馈类型/grounding | 关键词规则 `classify_feedback` / `ground_feedback`（**核心方法原样保留**，未替换为学习式分类器） | ⚠️ 有意近似（旁路，不在学习通路） |
| colored-shape 任务特征 | Overcooked 协作特征 | 换 domain，结构不变 |
| 真实教师-学习者交互数据 | **DeepSeek 合成语料** 2693 条（validated 2619，遵循 `.cursor/skills/llm-feedback-corpus`）+ 手写 probe 对照 | 真实人类语料仍是缺口，见 `MIGRATION_NOTES.md` / `DIFFERENCES_FROM_PAPER.md §9` |

> 论文原始代码保留在 `../original_rewards_repo/` 仅作参考，不被 Overcooked 运行时导入。

**当前冒烟结果**（100 条种子反馈，零均值先验起步，14 个动作级 probe）：
- `literal`：learned probe accuracy 14/14（均值口径与采样口径均 100%）；
- `pseudopragmatic`：learned probe accuracy 14/14；
- 对照：`zero_weights` 0/14（全平局），`initial_weights` 14/14，`leave_one_probe_out` 14/14。
- probe 覆盖 5 类：HumanComfortCoordination / RespectHumanIntent / RecipeCorrectness / ServingReadiness / Safety / Efficiency，含 4 个 `hard` 权衡样本。

---

## 4. 一句话总结

奖励学习核心已从“点估计单步更新”升级为**忠实复现论文的贝叶斯高斯信念 + 共轭更新（active `multiply()` 形式）+ 采样式动作选择**，并同时提供 Literal 与 PseudoPragmatic 两个论文变体；情感已用 NLTK VADER 对齐（英语单语）。**路线2 的学习式推断网络现已真正训练**（`train_route2.py`，DeepSeek 合成语料、text-only、分组 CV，held-out subgoal accuracy 18/18=100%）。收尾采用 **subgoal-only**：学到的 comfort 奖励直接重排 H0 子目标来体现 agent 行为提升（6 seed 均值 comfort/step +0.97→+2.76、discomfort 0.200→0.003、soup 80→76.7），**不训练 PPO**（DURF 集成扩展，非论文复刻）。仍为近似的是反馈分类/grounding 的规则版（**核心方法有意保留、不替换**，且不在学习通路上）。数据来源与诚实边界见 `DIFFERENCES_FROM_PAPER.md §7/§9` 与 `.cursor/skills/llm-feedback-corpus`。

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
