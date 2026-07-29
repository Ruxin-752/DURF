# 与原论文/原实验的差异记录（Baseline B, Overcooked 适配）

原文：*Learning Rewards from Linguistic Feedback* (Sumers et al., AAAI 2021)，
原始仓库见 `../original_rewards_repo`。

本文件逐条记录本适配实现与原始实验的**所有已知差异**，并标注每一项是
"已严格对齐 / 有意近似 / 数据缺口 / 超出原文范围"。

> **更新（迁移收尾）**：Route 2 神经推断网络现已**真正训练**，语料由 **DeepSeek
> 合成**（rule teacher 出标签 + LLM 只出语言 + skill 门控，见 §7/§9 与
> [`.cursor/skills/llm-feedback-corpus`](../../../.cursor/skills/llm-feedback-corpus/SKILL.md)）。
> 收尾方式为 **subgoal-only**：学到的 comfort 奖励直接重排 H0 子目标来体现 agent
> 行为提升，**不训练 PPO**（PPO 属 DURF 集成扩展、非论文复刻）。**保留 Baseline B
> 核心方法不改造**：linguistic 反馈归类（`feedback_form_classifier.py` /
> `overcooked_grounding.py`，规则版）与基本策略（`GaussianBelief` +
> `BayesianRewardLearner` 共轭更新、`subgoal_reranker` 子目标重排）均按原样保留。
> 短语分类器（TF-IDF+LR）、nested-logit、R 统计仍不执行。

---

## 0. 训练边界（明确停止点）

原实验里"训练"发生在三处（另有 R 统计的回归拟合）：

| 训练步骤 | 原文位置 | 本实现处理 |
| --- | --- | --- |
| 神经奖励推断网络（Route 2） | `notebooks/aaai_inference_network_training.ipynb` 第 648 行 `loss.backward()` | **已训练**：`scripts/train_route2.py` 在 DeepSeek 合成语料上真正训练（text-only，分组 CV 早停）。严格对齐版 `scripts/train_route2_inference_network.py` 仍保留"训练前停止"边界仅作参考（见 §7） |
| 短语引用类型分类器（LogisticRegression + TF-IDF） | `notebooks/aaai_phrase_classifier_training.ipynb` | 已按原文预处理与超参训练 Overcooked 五类模型；合成数据为主（见 §3） |
| nested-logit 轨迹归因（SciPy Powell） | `science/observations/behavioral_analysis.py` | 不调用（原文主评估默认也走 feature counts，不走它） |
| R 统计（`lm/lmer`） | `notebooks/aaai_statistics.Rmd` | 不做；属结果分析，非 reward learner |

> 结论：Literal / PseudoPragmatic learner 本身是**贝叶斯推断，不含训练**，
> 已完整复现；Route 2 神经网络和短语指涉分类器均已训练。仍被跳过的是
> nested-logit 与 R 统计；分类器缺少真人 Overcooked 金标准。

---

## 1. 已严格对齐的部分（非训练）

| 项 | 原文 | 本实现 | 状态 |
| --- | --- | --- | --- |
| 权重先验 | `MultivariateNormal.from_labels`，均值 0、协方差 `25I`（precision `1/25`） | `GaussianBelief.prior` 同 `N(0,25)` | ✅ 严格对齐 |
| 信念更新 | 实验 learner 使用 `belief_state.multiply(obs)`，obs 为 `N(mean=r·v, precision=p·rrᵀ)`（`agents.py` 第 184 行） | `GaussianBelief.multiply_observation`，逐项复刻 multiply（信息项含 `p·(r·r)·v·r`） | ✅ 严格对齐（见 §5.1） |
| Literal / PseudoPragmatic | `ExpLiteralLearner` / `ExpPseudoPragmatic`（valence×30、precision 2、pragmatic valence −30） | `BayesianRewardLearner` 同参数，`--mode literal/pseudopragmatic` | ✅ 严格对齐 |
| 情感/valence | NLTK VADER，`modified_vader_observation`（zero 词→0；VADER 非零→compound；中性→默认 +0.5） | `sentiment_extractor.py` 逐条复刻 | ✅ 严格对齐 |
| 短语分词→多观测 | `limited_punc_tokenization` 按 `! . , ; |` 切分，逐短语建观测 | `text_analysis.limited_punc_tokenization` + pipeline 逐短语观测 | ✅ 机制对齐（见 §4） |
| 参考向量归一化 | 参考向量归一化到和为 1（`reference_vector_from_*`） | `observations.reference_vector(normalize=True)` + `normalize_reference_vector` | ✅ 严格对齐 |
| pragmatic 反向参考 | 未提及维度取反并归一化，valence −30 | `build_observations` 同逻辑 | ✅ 严格对齐 |
| 采样式选动作 | `execute_trajectories`：从后验采样权重，逐样本 argmax | `evaluate_probes_sampled` | ✅ 严格对齐 |
| 在线学习曲线评估 | `aaai_model_evaluation`：逐条反馈在线更新→固定 benchmark 重决策→多次运行均值+CI | `scripts/evaluate_learning_curve.py`（多 seed、95% CI、literal vs pseudopragmatic、随机基线） | ✅ 协议对齐（度量为 probe accuracy，见 §6） |
| Route 2 网络结构 | `EmbeddingBag(vocab,30)` + 拼接轨迹特征计数 → `Linear(·,128)`+ReLU→`Linear(128,·)` | `neural_inference.TrajectoryFeedbackRewardPredictor` 同结构 | ✅ 结构对齐（已训练，见 §7） |
| 神经文本预处理 | `nn_preprocess_chat_phrase`（去标点、小写、word_tokenize、WordNet lemmatize、数字转词） | `text_analysis.nn_tokenize` 逐条复刻 | ✅ 严格对齐 |

---

## 2. 环境与特征空间（有意近似：不同任务域）

| 维度 | 原文 | 本实现 |
| --- | --- | --- |
| 环境 | 单智能体积木采集（collect colored/shaped objects） | 双人 Overcooked（协作做汤、出餐） |
| 特征 | 15 维 sign/magnitude 组合（3 shapes + 3 colors + 9 conjunctions），`SignMagnitudeFeaturizer` | 53 维手工 schema（食材/配方/锅态/空间接近/路径代价/时间/人机协作舒适度），`data/overcooked_features.json` |
| reward 输出维度 | 9 维 conjunction reward | 53 维奖励权重向量（Route 2 输出维度随之改为 53） |

> 原因：任务域从积木采集迁移到 Overcooked 协作；这是 DURF 的目标环境。
> 该差异是**根本性任务迁移**，其余所有差异都在此前提下理解。

---

## 3. 短语指涉分类（结构对齐，数据仍近似）

- 原文：TF-IDF + LogisticRegression 学习式短语引用类型分类器（5 类，
  accuracy ≈ 0.87），发布运行时用 augmented pickle。
- 本实现：`phrase_reference_classifier.py` / `train_phrase_reference_classifier.py`
  复刻标点切句、lemmatization、TF-IDF 1–2 gram（`min_df=5`）与
  LogisticRegression，输出 `trajectory / feature / action_spatial /
  action_behavioral / other` 及概率。`feedback_form_classifier.py` 的三类
  evaluative/imperative/descriptive 只保留为 speech-act 分析，不再控制 grounding。
- 训练使用 validated DeepSeek/模板语料，并对 `feature`、`action_behavioral` 与
  `other` 单独增强；完整场景 group-disjoint untouched test 为 accuracy **74.2%**、
  macro-F1 **68.3%**、balanced accuracy **72.6%**。旧 83.0% 来自较宽松的
  intent-family 拆分，现不再作为最终数字。当前 `action_behavioral` recall
  **51.7%**，`other` recall **62.5%**（但 precision 仍低）。
- 状态：**模型与推理结构对齐，训练域有意近似**。论文约 87% 来自真人原任务，
  当前数字来自合成 Overcooked，不能直接宣称等价。

---

## 4. 短语级 grounding 的取舍（有意近似）

- 原文：逐短语分类引用类型（trajectory / feature / action_spatial /
  action_behavioral），再分别用轨迹频率 / 字符串匹配 / 空间簇得到参考向量。
- 本实现：
  - 按 `limited_punc_tokenization` 逐短语分类 reference type、逐短语产观测并取
    VADER valence；分类概率与 grounding confidence 一起缩放 observation precision；
  - 但**显式 `target_features` / `trajectory_features` / `target_action`**
    （手工标注的参考向量）优先，此时整句作为**单一参考**（单观测），
    valence 取整句 VADER。
- 原因：本数据集大量使用手工标注的参考向量（更可靠），而非从真实轨迹自动
  归因；因此多子句的评价/描述句按整句参考处理，属有意近似。
- 状态：**路径结构已对齐，referent 解析仍为 Overcooked 规则近似**。

---

## 5. 数值细节

### 5.1 更新形式（已对齐到"active"路径）

原仓库 `beliefs.py` 有两种等价形式：
- `multiply()`（实验实际使用）：信息项 `p·(r·r)·v·r`；
- `update_from_observation()`（BLR 形式，**被注释掉**）：信息项 `p·v·r`。

之前的实现用的是 BLR 形式；为"严格对齐原实验"，现默认使用
`multiply_observation`（active 路径）。BLR 形式仍保留在
`belief_model.update_from_observation` 供对照。两者在参考向量归一化
（`r·r ≠ 1`）时数值不同，测试 `BeliefUpdateFormTests` 予以覆盖。

### 5.2 情感取值范围

- 原文与本实现一致：VADER compound ∈ [−1,1]，再乘 `valence_scale=30`。
- 数据为英语单语（VADER 仅支持英语），原种子集中的中文反馈已全部翻成英语，
  规则模块中的中文标记也已移除（见 §8）。

---

## 6. 评估度量差异（任务域导致）

- 原文度量：`pct_max = 选中轨迹奖励 / 关卡最大奖励`，在 100 个 benchmark 关卡上
  逐交互重决策，10 折 held-out teachers + held-out reward configs，5 次重复。
- 本实现度量：probe 决策**准确率**（chosen ∈ acceptable，tie 记失败）与
  **采样期望准确率**；held-out 用 leave-one-probe-out / 按 probe 分组的 CV
  作为 teacher/reward-config 的类比。
- 原因：Overcooked probe 是离散"可接受/不可接受"决策，没有连续 ground-truth
  奖励标量，故用准确率与期望准确率替代 pct_max。
- 状态：**协议对齐、度量近似**。

---

## 7. Route 2 神经推断网络（已训练，用 DeepSeek 合成语料）

本仓库有**两套** Route 2 入口，请勿混淆：

1. `scripts/train_route2_inference_network.py`——**严格对齐版**，故意停在训练前
   （前向校验后打印 `TRAINING BOUNDARY REACHED` 退出）。它保留论文"训练前
   停止"边界，仅作结构/形状对齐参考。
2. `scripts/train_route2.py`——**DURF 可训练版（现为主用）**，在 DeepSeek 合成
   语料上真正训练 `语言 → 53 维 comfort 奖励向量`，导出冻结权重供 subgoal 重排。

- 结构、文本预处理、词表、按组 CV 折均按原文搭好（`EmbeddingBag(vocab,30)` +
  `Linear(30+n,128)` + ReLU + `Linear(128,53)`）。
- **已训练**（可训练版）：默认 **text-only**（`use_feature_counts=False`，运行时
  只有人类文本，不泄露 grounding 参考向量，保证指标诚实）；按 `group_id`
  分组 CV 早停在**未见场景**上。当前语料 2619 条（validated），vocab 408，
  best val MSE ≈ 0.0015。
- **数据来源**：不再是手工种子，而是 **DeepSeek 合成语料**（rule teacher 出
  标签 + LLM 只出语言 + skill 门控，详见 §9 与
  [`.cursor/skills/llm-feedback-corpus`](../../../.cursor/skills/llm-feedback-corpus/SKILL.md)）。
- 目标向量近似：原文回归目标是 reward config 的真值奖励向量；本实现的真值来自
  手写 gold 规则 `w*`（`data/gold_comfort_weights.json`）经 featurizer 得到的
  comfort 奖励方向。因此**只验证 language generalization（自由文本→正确子目标），
  不验证规则 `w*` 本身**。
- 状态：**结构对齐 + 已训练（合成真值）**；真实人类语料仍是后续缺口（§9）。

---

## 8. 语言：英语单语（有意约束）

- VADER 只支持英语；按项目决定，反馈语料统一英语。
- `data/feedback_examples.json` 原有约 48 条中文反馈已翻成英语，并调整措辞
  使 VADER 情感符号与该样本的目标特征极性一致（正向行为→正情感，负向行为→
  负情感）；测试 `VaderSentimentTests.test_all_feedback_texts_are_ascii_english`
  强制全英语。
- `feedback_form_classifier.py` / `overcooked_grounding.py` 中的中文关键词
  已移除。

---

## 9. 数据规模与真实性（DeepSeek 合成 + 数据缺口）

| 项 | 原文 | 本实现 |
| --- | --- | --- |
| 反馈来源 | 真实 human-human / human-agent 实验记录（约千条 dyad） | **DeepSeek 合成语料** 2693 条（validated 2619）：90 个决策场景 × rule teacher intent × 模板底线 + LLM 释义；另有手工种子 100 条 + 14 动作 probe + 5 子目标 probe 作对照 |
| 语料生成方法 | 真人采集 | 遵循 [`.cursor/skills/llm-feedback-corpus`](../../../.cursor/skills/llm-feedback-corpus/SKILL.md)：**标签来自 rule teacher（gold `w*`），语言来自 LLM**（observer/critic 锚定 prompt），按语义 hash 缓存、不缓存空结果 |
| 语料门控 | — | `validate_feedback_corpus.py`：非矛盾情感 + grounding 双门控（模板恒留、LLM 需通过）；当前 diversity 45.1%、valence 正 1410/负 1209、三类 speech-act 齐全、非矛盾率 99.9% |
| 轨迹归因 | 真实游戏轨迹（对象采集序列） | 手工标注参考向量为主；`trajectory_featurizer.py` 仅覆盖部分 53 维 |
| held-out 划分 | 真实 teacher / reward-config 双维度 held-out | 按 `group_id`（场景）分组 CV；另留 5 个手写 subgoal probe 作从不训练的真实 held-out |

> **诚实边界**：合成真值来自手写规则 `w*`，因此本收尾验证的是 **language
> generalization（自由文本 → 正确子目标）**，**不是** comfort 规则本身正确。真实
> 人类 Overcooked 语言反馈仍是后续缺口（`convert_session_feedback.py` 已就绪，
> 但 ring session 目前零语言反馈）。

---

## 10. 其他实现层差异（超出原文范围或口径不同）

1. **子目标兼容层**（`subgoal_schema.py` / `subgoal_featurizer.py` /
   `subgoal_reranker.py`）：把学到的奖励接到 H0 的 8 个子目标上重排，是 DURF
   集成扩展，原论文没有。
2. **采样评估的 tie 处理**：`evaluate_probes_sampled` 用 `np.argmax`，精确平局
   时偏向首个动作；确定性 `evaluate_probes` 则把 tie 记为失败。两者口径不同
   （原文采样式评估同样用 argmax）。
3. **离线范围**：不修改 PPO checkpoint 或 pygame 实时策略；只评估学到的奖励
   信念是否在受控 probe 状态下偏好期望动作。
4. **`initial_weights.json`**：是手工对照权重，不是贝叶斯 learner 的先验均值
   （先验仍为 `N(0,25)`）。
5. **连续 WAIT 防停滞**：子目标重排的安全下限/防停滞留给 live caller（H0），
   本包不强制。

---

## 11. 迁移收尾最终结果（subgoal-only，不含 PPO）

- **Route 1（贝叶斯 learner，核心方法原样保留）**：pseudopragmatic learned probe
  accuracy **14/14 = 100%**（采样期望准确率 100%，tie=0）；在线学习曲线 final
  **100%**（literal & pseudopragmatic），随机基线 33.3%。
- **Route 2（神经推断网络，DeepSeek 合成语料训练，text-only）**：专类增强语料
  的严格 untouched test 为 **subgoal accuracy 17/18 = 94.4%**（多数投票；
  逐条 89.8%），对照 gold 18/18、route1-global 17/18、zero 0/18。严格模型另存
  为 `model_deepseek_strict.pt`；其导出均值权重在手写 probes 上仅 3/5，因此未
  替换默认冻结权重。
- **Agent 行为（20 seed 严格协议）**：在 dev seeds 0–9 选择 `lambda=0.75`，
  untouched test seeds 10–19 上平均 soup **76 → 88**、discomfort rate
  **16.6% → 14.7%**；comfort/step 因 seed 方差从 1.635 降到 0.941，不能宣称
  comfort 分数稳定提升。结果见 `outputs/route2/comfort_multiseed_strict.json`。
- **未做**：PPO（DURF 集成扩展，当前范围为 subgoal-only）；真人
  Overcooked 五类 reference 金标准采集与最终测试。

### Live Route 1（原论文机制显式进入 subgoal loop）

`ComfortSubgoalAgent` 不再把所有聊天默认等同于 Route 2 blend，而是提供四个
明确且可对照的模式：

- `frozen`：冻结权重控制组；
- `route1-literal`：类型分类 → 类型化 grounding → Literal 高斯后验更新；
- `route1-pseudopragmatic`：同上，并加入未提及特征的 inverse-reference 更新；
- `route2`：神经网络逐句预测后按 α 混合。

Route 1 默认从论文先验 `N(0,25I)` 开始，也可用 `--route1-prior frozen` 把
Route 2 导出权重作为先验均值。每次反馈都记录 type、grounding、valence、
posterior mean/variance delta，以及更新前后的 subgoal 到
`feedback_updates.jsonl`。这使 Route 1 不再只是离线 probe 对照，而是能直接
改变下一步 H0 subgoal 排序。评价型反馈使用最近 25 个实际执行 step 的
trajectory features；若窗口中没有可检测事件，再回退到同期 selected-subgoal
features，避免只凭暂停瞬间的静态状态做错误归因。

真人 live 评论使用来源感知 precision（默认是普通合成观测的 4 倍），但会再乘
reference 分类概率与 grounding confidence；含糊评论会降权或拒绝，避免错误解释
被高权重放大。完整 mean/covariance 以版本化 JSON 原子保存，可显式跨进程恢复。
DeepSeek 回复与本地学习解耦，API 延迟或失败不会阻塞下一条真人评论。

行为层增加连续 WAIT guard。seed 42、400 step 的当前回归结果为：H0 与 comfort
agent 都完成 80 soup；comfort/step `+0.829 → +2.030`，discomfort rate
`0.1425 → 0.0725`，结果见 `comfort_subgoal_eval_wait_guard.json`。

5 折 held-out subgoal 消融（`outputs/route1_subgoal_ablation_reference_classifier.json`）：
Route 1 Literal 的 oracle / inferred 分别为 **97.8% / 91.1%**，冻结 Route 2
为 **95.6%**。专类 DeepSeek 增强后 inferred grounding coverage 为 **80.4%**
（旧规则路径为 72.6%）；descriptive-only coverage 为 56.6%，说明剩余瓶颈
主要在 referent/关键词解析。三类 speech-act 的 57.3%
仅作为独立分析指标，不再用于选择 grounding 路径。

---

## 复现命令速查（subgoal-only 全流程）

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$B="baselines/baseline_b_linguistic_feedback/adapted_overcooked"
# 1. 枚举决策上下文
python $B/scripts/enumerate_subgoal_contexts.py
# 2. DeepSeek 合成语料（需 DEEPSEEK_API_KEY；遵循 .cursor/skills/llm-feedback-corpus）
python $B/scripts/generate_synthetic_feedback.py --llm-per-intent 4
# 3. 门控校验 -> synthetic_feedback.validated.json
python $B/scripts/validate_feedback_corpus.py
# 4. Route 1：贝叶斯 learner + probe（literal / pseudopragmatic）+ 学习曲线
python $B/scripts/run_baseline_b_pipeline.py --mode pseudopragmatic
python $B/scripts/evaluate_learning_curve.py
# 5. Route 2：训练语言->comfort 奖励（text-only）
python $B/scripts/train_route2.py --feedback $B/data/synthetic_feedback.validated.json --epochs 120
# 6. Route 2 held-out subgoal 评估 + 导出冻结 comfort 权重
python $B/scripts/evaluate_route2_subgoal.py --feedback $B/data/synthetic_feedback.validated.json
# 7. 回合级 agent 行为对照（gold w* 裁判）
python durf/baseline/evaluate_comfort_subgoal.py --horizon 400
```
