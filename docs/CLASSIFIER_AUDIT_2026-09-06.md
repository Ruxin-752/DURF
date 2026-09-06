# Web 三分类审计（2026-09-06）

当前生产三分类与现有候选都不能据本次证据宣布“自然语言三分已修复”。本次只读审计没有修改训练集、模型、默认 Web 配置或线上部署。

## 测试范围

- 固定 72 条英文单句：评价、命令、描述各 24 条，其中 12 条为位置词反事实；另有 10 条混合句检查分句及标签顺序。
- **72 条是 AI 编写的语义诊断，不是独立人类金标，也不是当前真实玩家准确率。** 标签按字面 speech act；个别状态句在游戏语境中也可能表达隐含评价或请求。
- 所有受测模型先于诊断固定。没有用诊断训练、调参或改标签；原始人工 frozen test 文件没有被本审计脚本打开。现在已暴露的诊断不得再当独立验收集。
- 浏览器模型通过实际 `loadBrowserModelsFromManifest` 验证 SHA-256 后运行；分句函数从当前 `route-inputs.ts` 原样提取执行。
- Python 候选使用仓库 `.venv` 的 Python 3.10.20 / scikit-learn 1.7.2，与序列化模型版本一致。使用原始概率比较 argmax，不把温度缩放或原始分数当正确率。

## 定量结果

| 模型 | 正确数 | 诊断准确率 | Macro-F1 | 评价 recall | 命令 recall | 描述 recall |
|---|---:|---:|---:|---:|---:|---:|
| Web production | 36 / 72 | 50.0% | 0.481 | 79.2% | 45.8% | 25.0% |
| Web boundary-shadow-raw-v2 | 52 / 72 | 72.2% | 0.714 | 91.7% | 79.2% | 45.8% |
| direct_fg_v2 | 60 / 72 | 83.3% | 0.831 | 91.7% | 66.7% | 91.7% |
| direct_fg_clean_v3 | 52 / 72 | 72.2% | 0.694 | 87.5% | 100.0% | 29.2% |
| direct_fg_clean_v4 | 61 / 72 | 84.7% | 0.847 | 87.5% | 95.8% | 70.8% |
| paper_recipe_20260906 | 25 / 72 | 34.7% | 0.217 | 95.8% | 8.3% | 0.0% |

这些结果均未达到先前讨论的 Macro-F1 ≥ 0.87；每类 recall ≥ 0.85 也均未满足。该门槛只能作为诊断检查，不能替代独立玩家验证。

生产模型在 ≥ 0.8 分数的错误有 10 条；采用现有 0.55 拒绝阈值后，62 条被接收，接收准确率仍只有 51.6%。shadow 接收 50 条，接收准确率 88.0%，覆盖率 69.4%；不能忽略另外 22 条或把这个子集准确率当整体能力。

## 已复现的错误

| 输入 | 字面期望 | production | 模型分数 |
|---|---|---|---:|
| Go to the stove. | Imperative | Evaluative | 72.4% |
| Bring me a clean plate. | Imperative | Descriptive | 68.5% |
| The serving window is on the right. | Descriptive | Imperative | 94.0% |
| There are two onions in the pot. | Descriptive | Imperative | 97.0% |

12 条位置反事实中，production 正确 5 条，shadow 正确 8 条。改变位置词仍会暴露空间词与类型判断的混淆。

10 条混合句中，当前分句器只有 6 条分段数量正确；逗号、`and`、`so` 连接的异类子句未拆分。两个 Web 模型都只有 3 / 10 条得到完整正确标签顺序。均值 softmax 表达平均类别分数，不等于一句话中三种语义成分的比例。

本轮新训练的 paper recipe 在同一固定诊断上仅执行一次：25 / 72，混合完整标签顺序 0 / 10，训练精确重合 0，推理前后模型 SHA-256 不变。它仅使用原论文 human train 448 / dev 99 的功能标签，采用 word unigram/bigram TF-IDF、`min_df=5`、English stop words、sublinear TF、LogisticRegression；没有复现作者 lemmatization，因此不能称为精确复现。其功能标签 dev accuracy 85.9% / Macro-F1 0.843 不能用作表面语气能力证明。推理使用与本轮训练匹配的 Python 3.12.10 / scikit-learn 1.8.0。

该新模型没有导出到 Web。当前 JS word analyzer 未实现 scikit-learn 的 English stop words；直接导出会产生预处理不一致。即使补齐导出一致性，上述语义失败也仍然存在。

## 语义根因和数据证据

`train_feedback_form_classifier.py` 把 `trajectory / action_behavioral → evaluative`、`feature → descriptive`、`action_spatial → imperative`。这是论文反馈功能/引用对象定义，与玩家直观的评价、请求、状态描述并非同一标签任务。仅换特征或重训相同标签不能保证修复表面语气。

当前 Route1 又把三类结果映射回轨迹、动作和特征向量。直接用 speech-act 模型替换 fG 会改变信用分配，需要分开评估玩家显示与 Route1 grounding。Route2 输入与更新不依赖 fG，切换 Route2 不能证明三分类修复。

数据完整性检查：

- direct_fg_v2：917 条，train 755 / dev 162；规范化文本、group_id、split_family 的 train/dev 重合均为 0。但通过门槛来自选模 dev，人工 test/frozen 未测，manifest 仍为 `not_promoted`。
- clean_v3：train 1,020 / calibration 120 / final_eval 300；clean_v4：train 1,620 / calibration 90 / final_eval 450。文本和声明的 component/surface/scenario ID 的分区重合均为 0。它们在本次 72 条中与训练精确重合为 0，仍明显失败；零精确重合不代表句式、词汇或标注独立。
- clean_v3 历史合成 final accuracy 100%，但冻结记录已指出描述句失败和温度落在搜索下界；其本次描述 recall 仅 29.2%。合成满分不能外推到玩家。
- clean_v4 历史合成 final accuracy 91.3%，命令 recall 74.0%，历史状态为 `failed_research_candidate`。
- clean_v5 没有训练模型；clean_v6 明确 `training_performed=false`，独立语义审查失败，历史审查记录有 47 对 train-to-final-eval 核心近重复，以及仅前 3 词就能达到 94.4% 的标签规则。
- shadow 仅有 240 条合成训练行。既有一次性诊断为 59 / 72，报告写明失败且 `promotion_eligible=false`。
- production 的纸面 frozen benchmark 自身标明 `previously_exposed_task_held_out_proxy_regression`、`independent_sealed_test=false`。其历史表现只能作为回归证据。

上述历史指标来自现有报告的本次只读复核；本次新增数值来自下列可复跑脚本。没有声明已完成新的独立人类标签审查。

## 身份与复现

| 对象 | SHA-256 |
|---|---|
| production Web JSON | `2280f07b71b5790bbd53a90cea38dd4e84fa9e7f1cf34be1510e304b85bdcb48` |
| production source joblib（manifest 绑定） | `3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31` |
| shadow Web JSON | `2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef` |
| direct_fg_v2 joblib | `c06d2ab585f8abe3cced01ebbcb0958188b793ddd720893d9a2d71705a19d268` |
| clean_v3 joblib | `0d0b46f3266cb8e8d7757177a6f76dff77cbda89d793f8ce674e31000be150a2` |
| clean_v4 joblib | `ec22916795d58583caf746d8dca61fab49b75a7f1d60d9db91411c02e8c04be7` |
| paper_recipe_20260906 joblib | `647eb77dc20c4d0a5ad06cc0bd9ed894365ac4d11a7ccbb0b49f7835612d2119` |

从仓库根目录执行：

```powershell
node artifacts/classifier-audit-20260906/audit-browser.mjs
.\.venv\Scripts\python.exe artifacts/classifier-audit-20260906/audit-candidates.py
```

完整逐条输出：`artifacts/classifier-audit-20260906/browser-semantic-report.json`、`candidate-semantic-report.json`、`paper-recipe-semantic-report.json`。固定输入：`probes.frozen.json`。复跑只用于复现，不能把已看过的结果当新的独立验证。新 paper recipe 的 `audit-paper-recipe.py` 已生成一次性收据，重复执行会拒绝覆盖或再次推理。

线上实际资源、后台可达性及训练数据导出由并行审计负责，本报告没有以本地结果代替线上状态。
