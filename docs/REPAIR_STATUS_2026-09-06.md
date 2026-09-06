# 三分类、论文方法与 Web 数据修复状态

> 本文保存的是用户确认方案之前的审计记录。用户随后已批准分离 speech-act / grounding，并明确只使用合成数据。本轮后续实现与验收见 `SYNTHETIC_RELEASE_2026-09-06.md`；下文“待确认”是历史状态，不再要求重复批准。

本轮已完成原文核查、隔离训练对照、语义诊断、后台只读核验及本地修复。**三分类尚未通过验收，尚未部署。** 用户要求的最终目标仍未完成；需要先确认分类标签定义，再取得足够的厨房领域独立标注和验证证据。

## 原文方法与训练

[原文](https://arxiv.org/pdf/2009.14715)总体表现最好的是 Pragmatic sentiment model，对应 Route1，而非 Route2。其三分类器采用 TF-IDF unigram/bigram + Logistic Regression。原文的任务分数、分类器 weighted-F1 和玩家三分准确率是不同指标。

新增 `baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/train_paper_feedback_recipe.py`，明确要求标签语义；仅读取 train/dev、只在 train 拟合向量器、拒绝组或文本重叠，生成隔离模型及报告。实验使用原论文领域功能标签 train 448 / dev 99，开发集 accuracy 85.86%、macro-F1 0.84324、Imperative recall 80%。原始文本预处理和分组划分不同于作者 notebook，不能称作精确复现。

新增 `web/lib/pragmatic-route1.ts` 及 12 项测试，实现原文中性情感 +15、未指及特征 -30 的 Gaussian 更新。尚未接入游戏组件；还需原始 VADER 情感输出、厨房 grounding 和实际行为验证，不能把现有关键词情感函数当作 VADER。

详见 [方法核查](PAPER_METHOD_AUDIT_2026-09-06.md)。

## 三分测试

固定 72 条单句（每类 24）及 10 条混合句，覆盖厨房动作、位置反事实、请求、否定和分句。它们是 AI 编写的诊断，不是独立真人测试；没有将这些诊断回灌训练。

| 模型 | 单句答对 | Macro-F1 |
|---|---:|---:|
| 本地 production | 36/72 | 0.481 |
| boundary shadow | 52/72 | 0.714 |
| direct_fg_v2 | 60/72 | 0.831 |
| clean_v3 | 52/72 | 0.694 |
| clean_v4 | 61/72 | 0.847 |
| 本轮 paper recipe | 25/72 | 0.217 |

paper recipe 的测试标签是日常 speech act，与它学习的原文功能标签不同，因此这不是对原文报告分数的复现或反证；它直接说明该模型不能用作已修复的玩家语气显示。其 English stop words 还未被 Web 分析器实现，不能直接导出。

当前分句器在 10 条混合句中只有 6 条分段数量正确。现有候选尚无足够独立厨房玩家验收证据，不能以合成集高分、调阈值或换 Route2 代替修复。

详见 [语义审计与逐项结果](CLASSIFIER_AUDIT_2026-09-06.md)。

## 线上与数据

实际页面为 https://durf-kitchen-lab-study.xc3083.chatgpt.site/ 。只读浏览器核验确认页面加载、未发现脚本错误，实际请求 manifest、feedback-form-v3 和 route2-v5。未勾选研究同意、未创建测试参与者或提交研究反馈。

受保护的管理员导出成功（200），未授权请求返回 401。导出有 13 个会话、3,240 个事件、21 条反馈；无孤立关联或重复会话序号。最近历史写入为北京时间 2026-09-04 10:10。该证据确认历史写入和当前读取，不能冒充本轮新的端到端写入验证。

符合当前 consent v3 的数据只有 5 个会话、2,296 个事件、14 条反馈；提交反馈的参与者只有 1 个，固定划分全在 train，没有独立 dev/test。三分预测不是人工 gold；Route2 模型预测权重也不是独立奖励目标。现有数据可进入待标注池，不能直接支撑可靠的有监督重训与晋升。

本地训练导入修复包括：区分 classifier/updater 模型哈希、优先保存实际 recentTrajectoryFeatures，以及显式按 consent 版本筛选关联记录。旧线上数据没有 recentTrajectoryFeatures，不能通过导出补造过去未记录的真实输入。

线上与本地反馈 JSON 的文件哈希不同，但实际比对确认五组推理字段（classes、transformers、classifier、calibration、minimum_confidence）全部相等。因此相同单句和预处理的分类计算可比；不能据此假定两个客户端的分句、界面和数据采集也完全相同。详见 [后台核验与转换修复](WEB_BACKEND_AUDIT_2026-09-06.md)。

## 待确认的重大选择

建议网页按日常语义显示评价／命令／描述，Route1 单独判断指代并采用原文 Pragmatic 更新。另一种研究目标是严格保留原文功能分类，两者需要不同标注和验收。

按用户“重大决定先询问”的要求，已发出分类定义确认。在收到答复且通过真实验收前，不切换生产三分或默认算法。后续必须完成标签协议、独立厨房数据、分句与两条语义输出、情感／grounding／行为测试、浏览器概率一致性以及部署后验证。

本轮 Web 测试 167/167、typecheck、lint 通过；训练数据转换测试 18/18、新增训练输入校验测试 4/4 通过。修复后的转换器在真实导出上仅在内存重放成功。这些工程测试不代表分类器语义已经达标。
