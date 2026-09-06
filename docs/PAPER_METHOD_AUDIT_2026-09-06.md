# 原文方法核查（2026-09-06）

## 已确认的结论

原文是 [Learning Rewards from Linguistic Feedback](https://arxiv.org/abs/2009.14715)（Sumers 等，AAAI 2021）。原文 Table 2 的真人在线平均归一化任务分数为 Pragmatic 42.8、Inference 35.0、Literal 34.7、Human 44.3。因此用户要求的“原文性能最好的模型”对应 Route1 的 Pragmatic 方案；Route2 的端到端奖励网络不是总体最优。这些是任务分数，不是三分类准确率。

原文正文 §5.1 的分类器是人工标注 + TF-IDF unigram/bigram + Logistic Regression，报告 weighted-F1 0.86。§5.2 在 Literal 上增加中性情感 +15、未指及特征 -30 的两项语用更新。[原文 PDF](https://arxiv.org/pdf/2009.14715)

## 可用训练来源及 685/982 的口径

本地附录 `baselines/baseline_b_linguistic_feedback/original_rewards_repo/Appendix.pdf` 第 5–6 页说明标注了 **685 interactions**；正文使用 **685 utterances**。附录同页表格是五类、测试 148 行：accuracy 0.87、macro-F1 0.75、weighted-F1 0.86，与发布 notebook 的输出相符。不能把 0.87 宣称为三分类准确率。

可定位原始人工标注文件：

`baselines/baseline_b_linguistic_feedback/original_rewards_repo/notebooks/data/pilot_chat_messages_labels.csv`

2026-09-06 只做统计、未改标签：982 行、82 个 task、686 个 task × level；类别为 trajectory 442、features 277、object_spatial 132、object_behavior 93、other 38。SHA-256：`b80963abc737c088cb7ff45865a9a73532aa4d7953e9db9a9dad8d7912595fa4`。附录写 81 pairs，因此正文、附录和发布文件存在单位或版本差异；目前没有确认独立的、带成员清单的“685 行 gold”。不应自行截取 685 行冒充原文训练集。

这份文件可用于发布代码协议复现及原领域 grounding 训练；不能直接当作厨房领域自然语气三分类 gold。其三类子集（排除 object_behavior/other）为 851 行，仍不是 685 行。

原始 notebook `.../notebooks/aaai_phrase_classifier_training.ipynb`：

- 154 行：读取上述 CSV；304–321 行：去标点、分词、词形还原、去英文停用词、单数字转词。
- 347–349 行：`TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1,2), stop_words='english')`。
- 461–468 行：随机行切分 85/15、`random_state=1`，仅在训练行拟合向量器，再训练默认 `LogisticRegression()`。
- 502–504 行：148 行测试的 accuracy/macro-F1/weighted-F1。随机行划分不等于任务、教师或重复文本隔离。

## 实现偏差

1. 论文的 E/I/D 是用于奖励归因的功能分类。原实现将 trajectory 和 action_behavioral 都映射到既往轨迹（`original_rewards_repo/science/observations/observations.py:19`）。附录明确：行为动作原标注属于 Imperative，但最终主实验把 object_behavior 当作 Evaluative。厨房命令与玩家理解的语气标签不能无条件沿用这项特殊处理。
2. 当前三分训练仍折叠 reference 标签：`adapted_overcooked/scripts/train_feedback_form_classifier.py:54–71`；模型特征是 word 1–3 + char 2–5（532–558 行），还做来源权重及 C/类别权重搜索（764–784 行），不是原 notebook 的精确配置。更换超参不能消除标签定义冲突。
3. Web 的 `web/lib/route-inputs.ts:93–99` 使用关键词情感 ±30、默认 +10，并非论文 VADER。`web/components/kitchen-game-app.tsx:744–754` 调用 Literal 更新。`web/lib/browser-models.ts:1204` 有 Gaussian 更新，缺少未指及特征的负更新。Python 的 `adapted_overcooked/src/observations.py:90–103` 已有后者。

## 本轮独立实现与下一步

新增 `web/lib/pragmatic-route1.ts`：接受原始情感分数 [-1,1]，复用现有 Gaussian 更新，先执行原文情感规则，再对未指及特征的 L1 归一化补集执行 -30 更新；拒绝的反馈不更新。它是独立纯函数，未接入组件、未切换默认路由。专用测试覆盖闭式 Gaussian 结果、相关协方差、命令/轨迹目标、中性与负情感、拒绝路径以及输入状态不被修改。

本地验证：`tests/pragmatic-route1.test.ts` 的 12 项测试通过，`tsc --noEmit` 通过。首次测试启动受沙箱 `spawn EPERM` 限制，经允许在沙箱外重跑后通过。

本轮另有独立训练实验 `adapted_overcooked/scripts/train_paper_feedback_recipe.py`，输入原论文功能标签 train 448 / dev 99；训练方式保留 word unigram/bigram、`min_df=5`、sublinear TF、英文停用词、Logistic Regression，但使用原始文本和组隔离切分，未复现作者的词形还原和随机行切分。该模型 dev accuracy 为 85.8586%、macro-F1 为 0.84324；这些是当前功能标签开发集结果。固定自然语气探针仅 25/72，与该模型训练标签定义不同，不能证明三分修复，详见 `docs/CLASSIFIER_AUDIT_2026-09-06.md`。

独立复核确认 train/dev 的 group、规范化文本、原始 paper_task_uuid 重叠均为 0，功能标签映射不一致为 0，4 项数据校验测试通过；未做近似重复语义审计。训练脚本只在不存在的新输出目录写模型与报告，本次产物位于 `adapted_overcooked/outputs/feedback_form_classifier_candidates/paper_recipe_20260906/`，未包含生产模型读写逻辑。模型 SHA-256：`647eb77dc20c4d0a5ad06cc0bd9ed894365ac4d11a7ccbb0b49f7835612d2119`；报告明确 `production_promotion_eligible=false`。未以此模型部署。

后续应先确认 E/I/D 是否指日常语气，还是论文 grounding 功能。若要日常语气，应独立标注 speech_act 与 reference_type，用 task/文本/模板组隔离测试，分别报告每类 recall、macro-F1、拒绝覆盖率和最小对照错误；grounding 和行为效果另测。保留原始标注，不能把模型预测写成 gold。

启用 Pragmatic 前还要接入经验证的原始情感输出、核对厨房特征的未提及惩罚是否适用、测试学习后的实际行为。此文档及纯函数不能证明三分已修好，也不构成上线证据。
