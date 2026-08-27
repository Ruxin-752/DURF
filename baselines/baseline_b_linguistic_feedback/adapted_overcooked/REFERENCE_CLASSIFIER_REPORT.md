# Reference classifier 评测与优化报告

## 当前结论

本轮优化后，classifier 在**只用于选模的 v8 开发集**上达到 `87.02% accuracy / 87.08% macro-F1`。这个结果达到了约 87%，但它不是封存真人测试，因此不能写成“真人泛化准确率 87%”。

模型仍采用与原论文主线一致的轻量分类器：word/character TF-IDF + LogisticRegression。选模过程只读取 train/dev，测试样本参与选模数量为 0。

最终开发集配置：

- word n-gram `(1,2)`，character `char_wb (3,5)`，`min_df=2`；
- LogisticRegression：`C=0.125`，`class_weight=balanced`；
- 原论文真人来源训练样本权重 `12`；
- 困难对比训练样本权重 `4`；
- 其他合成样本权重 `1`。

## 指标

| 数据分区 | Accuracy | Macro-F1 | 样本数 | 身份 |
|---|---:|---:|---:|---|
| v8 overall dev | **87.02%** | **87.08%** | 601 | train/dev 选模结果，不是外部测试 |
| v8 paper-human dev | 79.25% | 71.18% | 106 | 原论文真人域开发集 |
| v8 hard-contrast dev | 60.00% | 59.83% | 75 | 五类最小对比开发集 |
| v8 legacy-synthetic dev | 93.81% | 93.44% | 420 | 旧合成域开发集 |
| legacy synthetic regression | 92.68% | 92.40% | 861 | 已使用的合成回归集 |
| hard contrast regression | 45.33% | 38.56% | 75 | 未用于选模，但现已查看；暴露回归集 |
| fixed paper strict v1 | 77.36% | 72.52% | 106 | task/text 无泄漏，但已多次查看，不能再称 sealed |
| original-paper protocol reproduction | **87.16%** | 75.25% | 148 | 精确复现论文随机行协议；存在重复文本/任务重叠 |

原论文协议的 87.16% 证明代码可以复现论文 notebook 的分类器结果，但不能证明跨玩家、跨 session 的泛化。当前最诚实的说法是：

> 开发集综合指标已达到约 87%；无泄漏但已暴露的 paper v1 回归指标为 77.36%；Overcooked 真人封存集仍为空，因此真人准确率未知。

## 本轮修复了什么

1. 不再让数量更大的 DeepSeek 合成语料压过原论文真人语料；训练权重完全由开发集选择并写入模型 artifact。
2. 新增 375 条五类成组的困难对比语料；每个场景同时包含 `trajectory`、`feature`、`action_spatial`、`action_behavioral` 和 `other`。
3. 训练报告分开显示 paper-human、旧 synthetic 和 hard-contrast，避免一个高分掩盖另一个低分。
4. 固定 paper v1 继续检查 feedback ID、规范化文本和 task UUID 的真实重叠；任一重叠都会失败。
5. 保留原论文 87.16% 协议复现，但明确披露它不是外部测试。

## 仍然没有解决的问题

- hard-contrast 的跨模板回归只有 45.33%，说明模型仍会依赖表达形式，不能仅凭合成高分宣称理解语义。
- paper-human 中 `action_behavioral` 和 `other` 的边界仍较弱；原始短句本身也存在脱离上下文难以唯一分类的问题。
- 现在没有经过独立复核的 Overcooked 真人 gold test，无法证明真实玩家语言达到 87%。

这些问题不能通过继续查看固定 test 再调参来解决，否则会把 test 变成 dev。当前
两字段人工入口可建立 text-disjoint benchmark；若以后要证明跨玩家或跨 session
泛化，仍必须另外采集玩家/session ID。

## 真人两字段标注入口

原论文的 87% 指标对应上面的**五类细粒度 reference classifier**。用户要求的
`Evaluative / Imperative / Descriptive` 是论文用于分析的三类上层策略，二者不能
直接改名互换。因此项目保留五类模型做 credit assignment，同时新增独立三分类器。

真人 JSONL 每行严格只有两个字段：

```json
{"language":"Please fetch an onion next.","classification_label":"Imperative"}
```

追加位置：`data/human_feedback_form_annotations.jsonl`。模板和三类示例：

- `data/human_feedback_form_annotations.template.jsonl`
- `data/human_feedback_form_annotations.example.jsonl`

运行：

```powershell
python -B scripts/prepare_human_feedback_form_annotations.py
python -B scripts/train_feedback_form_classifier.py
python -B scripts/evaluate_human_feedback_form_test.py
```

合成监督严格采用原 notebook 的三类映射：`trajectory → Evaluative`、
`feature → Descriptive`、`action_spatial → Imperative`；`action_behavioral`
和 `other` 不强行塞进三类。generator 后来扩展的交叉句式标签不会作为训练目标。

因为两字段输入没有玩家或 session ID，系统只能保证规范化文本/模板族不交叉，
不能声称 teacher/session-held-out 泛化。当前另有 89 条用户确认标签的
AI-candidate 三分类 holdout，Accuracy 为 64.04%；它不是玩家原创语言，也不是
本报告五分类的真人 gold。五分类 Overcooked 真人 gold 仍为 0，因此五分类真人
accuracy 必须报告为 unknown。
