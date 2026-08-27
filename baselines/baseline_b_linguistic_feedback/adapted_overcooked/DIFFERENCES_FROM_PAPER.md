# 与原论文/原实验的差异记录（Baseline B, Overcooked 适配）

> **2026-08-10 current status:** Route 2 predicts the complete 53-dimensional
> teacher reward from language plus trajectory features and uses the paper's
> ten-fold teacher/reward holdout rotation with 12 stable synthetic authors and
> 36 reward configurations. The frozen seed-137 synthetic test passes all
> constant-baseline gates, but the separate local-language probe is only 1/4.
> The reference classifier reproduces the paper's
> 982-row benchmark at 87.16% accuracy and 75.25% macro-F1. See
> [ROUTE2_PAPER_ALIGNMENT.md](ROUTE2_PAPER_ALIGNMENT.md) for the authoritative
> protocol, metrics, and remaining human-data boundary. Older single-model
> numbers later in this file are historical ablations.
原文：*Learning Rewards from Linguistic Feedback* (Sumers et al., AAAI 2021)，
原始仓库见 `../original_rewards_repo`。

本文件逐条记录本适配实现与原始实验的**所有已知差异**，并标注每一项是
"已严格对齐 / 有意近似 / 数据缺口 / 超出原文范围"。

> **更新（迁移收尾）**：Route 2 神经推断网络现已在 83,592 条
> `paper_v5` 可识别合成完整奖励样本上训练；旧 DeepSeek/local-only 结果只保留为
> 历史消融，不能作为当前完整奖励推断证据。见 §7/§9。
> 收尾方式为 **subgoal-only**：学到的 comfort 奖励直接重排 H0 子目标来体现 agent
> 行为提升，**不训练 PPO**（PPO 属 DURF 集成扩展、非论文复刻）。**保留 Baseline B
> 核心方法不改造**：linguistic 反馈归类（`feedback_form_classifier.py` /
> `overcooked_grounding.py`，规则版）与基本策略（`GaussianBelief` +
> `BayesianRewardLearner` 共轭更新、`subgoal_reranker` 子目标重排）均按原样保留。
> 短语分类器（TF-IDF+LR）已经训练并复现原论文 accuracy；nested-logit 与 R
> 统计不属于当前 Overcooked 标量奖励输出范围，仍不执行。

---

## 0. 训练边界（明确停止点）

原实验里"训练"发生在三处（另有 R 统计的回归拟合）：

| 训练步骤 | 原文位置 | 本实现处理 |
| --- | --- | --- |
| 神经奖励推断网络（Route 2） | `notebooks/aaai_inference_network_training.ipynb` 第 648 行 `loss.backward()` | **已训练**：`select_route2_candidate_on_dev.py` 对完整奖励语料做逐折 dev-only 选择，`finalize_route2_selection.py` 冻结 checkpoint，随后一次性测试。`train_route2_inference_network.py` 仅保留训练前 shape-check（见 §7） |
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

## 3. 三类反馈形式 + 类内指涉分类（层级对齐，数据仍近似）

- 原文正文：三类 `f_G`（Evaluative / Imperative / Descriptive）决定
  trajectory / action / feature grounding。原作者发布代码进一步使用 TF-IDF +
  LogisticRegression 五类短语引用分类器（accuracy ≈ 0.87）。
- 本实现：`phrase_reference_classifier.py` / `train_phrase_reference_classifier.py`。
  原论文复现实验保留 lemmatization、word TF-IDF 1–2 gram（`min_df=5`）与
  LogisticRegression；部署的 Overcooked 模型仍用 LogisticRegression，但输入为
  raw word 1–2 gram + char_wb 3–5 gram。输出 `trajectory / feature /
  action_spatial / action_behavioral / other` 及概率。Route 1 先使用 UI 同源的
  三类 `f_G` 选择粗 grounding；五类只在该分支内细化。跨类五类预测会记录
  conflict 并投影回粗类，不能覆盖三类结果；`other` 保留安全拒绝。
- 同一实现按论文的 982 条人工标签和随机 85/15 协议复现 accuracy **87.162%**、
  macro-F1 **75.252%**，与论文约 87%/75% 对齐。
- 原论文随机行协议存在 48/148 个测试行同句重叠和 66 个测试 task 重叠，因此
  87.162% 只称为 **paper protocol reproduction**。v8 模型只用 train/dev
  选择来源权重和超参数，在 601 条综合 dev 上为 accuracy **87.02%**、macro-F1
  **87.08%**。
- 固定的 task_uuid 与 normalized-text 双重隔离 paper-v1 回归为 accuracy
  **77.36%**、macro-F1 **72.52%**，且 active 训练集与该分区的 ID/text/task
  overlap 都是 0；但它已经被查看，不能再称为 sealed external test。
- 861 条旧 Overcooked 合成回归集为 accuracy **92.68%**、macro-F1 **92.40%**。
  它用于发现代码回归，不是外部真人测试。
- 状态：**综合开发集达到约 87%，但真人封存泛化尚未证明达到 87%**；真人
  Overcooked gold labels 仍是最终证据。完整审计见 `REFERENCE_CLASSIFIER_REPORT.md`。

---

## 4. 短语级 grounding 的取舍（有意近似）

- 原文：正文三类 `f_G` 选择参考向量来源；发布代码逐短语细化为 trajectory /
  feature / action_spatial / action_behavioral，再用轨迹频率 / 字符串匹配 /
  空间簇得到参考向量。
- 本实现：
  - UI 与 learner 共享同一组逐短语三类预测；三类 confidence、类内 reference
    confidence、grounding confidence 与 valence confidence 一起缩放 precision，
    三类 abstention 会原子拒绝本条消息；
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

## 7. Route 2 神经推断网络（已训练，完整奖励目标）

请区分三类入口：

1. `train_route2_inference_network.py`：只做原论文网络 shape-check，故意停止在训练前；
2. `train_route2.py`：通用训练函数及历史消融入口；
3. `select_route2_candidate_on_dev.py` → `finalize_route2_selection.py` →
   `evaluate_route2_paper_crossval.py`：当前严格完整奖励流程。

当前流程保留 `EmbeddingBag(vocab,30)` + `Linear(30+n,128)` + ReLU +
`Linear(128,53)`。正式 `paper_v5` 语料有 83,592 条样本、12 个稳定 synthetic
authors、36 个 reward configurations；每条基础反馈跨全部配置扩展。相同
`(text, trajectory_features)` 指向多个完整 `w` 时数据构建直接失败。

十折同时隔离 teacher 与 reward configuration。每折候选只使用 train/dev，且选择
视图物理删除 test；最终 checkpoint 与 selection/CV/ensemble manifest 通过 SHA256
绑定，test 在读取前创建一次性 receipt。seed 137 的 8,513 条 test 上，变化维 MSE
为 `0.025101`、最近配置 accuracy 为 `99.9883%`、偏好敏感行为 majority 为
`97.5490%`、空手烹饪 majority 为 `93.5185%`，三项预注册常数基线 gate 均通过。

但 text-only 的最近配置 accuracy 仍为 `99.9178%`，trajectory-only 为 `0%`；独立
局部语言探针仅 `1/4`。因此当前状态是**结构与合成完整奖励协议对齐**，不是已验证
真人语言或真人 credit assignment。详见 `ROUTE2_AUDIT_REPORT.md`。

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

## 9. 数据规模与真实性（多套合成语料 + 真人数据缺口）

| 项 | 原文 | 本实现 |
| --- | --- | --- |
| 反馈来源 | 真实 human-human / human-agent 实验记录（约千条 dyad） | Route 1/分类器仍含 DeepSeek 与模板语料；正式 Route 2 使用 83,592 条 deterministic `paper_v5` 合成完整奖励样本 |
| 语料生成方法 | 真人采集后跨 reward configs 切换 token/trajectory features | 12 个独立 synthetic authors；每条基础反馈跨 36 个 reward configs 扩展，语言使用 reward-conditioned 普通语义表达 |
| 语料门控 | — | 完整奖励 schema、teacher/config ID、作者覆盖、teacher–reward 连通性及输入目标冲突均 fail closed |
| 轨迹归因 | 真实游戏轨迹（对象采集序列） | 合成 Overcooked 上下文的 53 维特征计数；text-only 很强，尚未证明真人 temporal credit |
| held-out 划分 | 真实 teacher / reward-config 双维度 held-out | 10 折 synthetic author + reward-config 双轴隔离；8,513 条一次性 test，真人 participant/game-disjoint test 尚缺 |

> **诚实边界**：正式结果验证的是可识别的 synthetic full-reward 组合泛化，不是
> 真人语言泛化。普通真人评论没有独立完整 `w`，必须保留为 unlabeled，不能把模型
> prediction 或恢复 posterior 当监督 gold。

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
- **Route 2（当前主结果）**：冻结 seed-137 十折 teacher/reward holdout 的
  8,513 条 test 上，完整 MSE **0.012748**、变化维 MSE **0.025101**、cosine
  **0.997620**、最近配置 accuracy **99.9883%**、偏好敏感 behavior majority
  **97.5490%**、空手烹饪 majority **93.5185%**。三项常数基线 gate 均通过；
  但 local-language probe 只有 **1/4**，不能外推真人效果。
- **旧 Agent 行为消融（20 seed，已不作为当前策略）**：曾在 dev seeds 0–9
  选择 `lambda=0.75`，
  untouched test seeds 10–19 上平均 soup **76 → 88**、discomfort rate
  **16.6% → 14.7%**；comfort/step 因 seed 方差从 1.635 降到 0.941，不能宣称
  comfort 分数稳定提升。当前论文对齐路径固定 `lambda=1`，语言只能修改权重。
- **未做**：PPO（DURF 集成扩展，当前范围为 subgoal-only）；真人
  Overcooked 五类 reference 金标准采集与最终测试。

### Live Route 1（原论文机制显式进入 subgoal loop）

`ComfortSubgoalAgent` 不再把所有聊天默认等同于 Route 2 blend，而是提供四个
明确且可对照的模式：

- `frozen`：冻结权重控制组；
- `route1-literal`：UI 同源三类 `f_G` → 受约束的类内 reference subtype →
  类型化 grounding → Literal 高斯后验更新；
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
三类 form、类内 reference、grounding 与 valence confidence；含糊评论会降权或拒绝，避免错误解释
被高权重放大。完整 mean/covariance 以版本化 JSON 原子保存，可显式跨进程恢复。
DeepSeek 回复与本地学习解耦，API 延迟或失败不会阻塞下一条真人评论。

行为层增加连续 WAIT guard。seed 42、400 step 的当前回归结果为：H0 与 comfort
agent 都完成 80 soup；comfort/step `+0.829 → +2.030`，discomfort rate
`0.1425 → 0.0725`，结果见 `comfort_subgoal_eval_wait_guard.json`。

5 折 held-out subgoal 消融（`outputs/route1_subgoal_ablation_reference_classifier.json`）：
Route 1 Literal 的 oracle / inferred 分别为 **97.8% / 91.1%**，冻结 Route 2
为 **95.6%**。专类 DeepSeek 增强后 inferred grounding coverage 为 **80.4%**
（旧规则路径为 72.6%）；descriptive-only coverage 为 56.6%，说明剩余瓶颈
主要在 referent/关键词解析。三类 `f_G` 现在实际选择 grounding 路径；当前
AI-candidate、人工确认标签 holdout 为 64.04%，因此低置信预测必须拒绝，不能
把 UI top label 当作确定事实。

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
# 5. Route 2：新目录中逐折 dev-only 选择并冻结（正式 test 只能运行一次）
python $B/scripts/select_route2_candidate_on_dev.py --split-seed 137 --output-dir <new_output_dir>
python $B/scripts/finalize_route2_selection.py --model-dir <new_output_dir>
# 6. Route 2 一次性 teacher/reward held-out 测试
python $B/scripts/evaluate_route2_paper_crossval.py --model-dir <new_output_dir>
# 7. 回合级 agent 行为对照（gold w* 裁判）
python durf/baseline/evaluate_comfort_subgoal.py --horizon 400
```
