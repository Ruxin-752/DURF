# 合成数据三分修复、Web 接入与发布记录

当前状态：已部署为现有公开站点的第 6 版（2026-09-06 09:18，北京时间），release `durf-synthetic-v3-20260906`。第三轮合成验收及去精确重叠后的补充分析通过预设门槛；568 项 Web 测试、typecheck、lint 和浏览器检查通过。线上实际模型哈希核验一致，管理员隔离 D1 写读删成功。结果仅证明本报告界定的能力，剩余评价类错误和真人效果未测的限制仍然成立。

## 用户确认的范围

- 本轮训练、开发与测试只使用合成数据，不将后台玩家文本用于训练。
- 玩家显示日常 `speech_act`：Evaluative / Imperative / Descriptive。
- Route1 另用独立模型预测 action / feature / trajectory，再运行 Pragmatic 奖励更新。
- 原文最佳 Pragmatic 的任务得分不是三分类准确率；本项目迁移其方法，不声称复现原文实验得分。
- 达标后发布至现有 DURF Kitchen Lab 站点，保持现有访问范围。

## 方法与实现

当前训练入口是 `scripts/train_synthetic_feedback_v3.py`，输出为 `outputs/synthetic_feedback_v1/final-v3/`。报告记录 92,191 个合成训练行、88,971 个不同的规范化训练文本、1,293 个开发行，train/dev 精确规范化重叠为 0。训练行包含加权重复，不能当成同等数量的独立表达。现存玩家文本未用于本轮训练。

分类器采用原文的 word TF-IDF（unigram/bigram）+ logistic regression 模型族，词表仅拟合训练集，正则化等选择使用开发集；标签、合成领域语料、预处理和开发协议经过厨房适配，不是原文实验的完整复现。显示模型学习 `speech_act`（Evaluative 评价 / Imperative 要求或建议 / Descriptive 事实），独立模型学习奖励指向 `grounding`（action / feature / trajectory），不做三个类别的一一硬映射。原始 softmax T=1；两模型的 0.55 是预先固定的拒绝阈值，未按 dev/test 准确率调优，分数未独立校准。

Route1 使用 VADER compound 与原文中性 +15 / sentiment ×30、固定 precision=2 的高斯更新；浏览器 VADER 与官方 Python 的 287 个对照逐字段一致。负向补集现已恢复为全部未指及的 53 维奖励特征，再按原文归一化，标识为 `paper-full-feature-complement-v1`。实际发生的动作或事实归 trajectory；未来要求/明确反事实归 action；状态、空间事实、一般偏好归 feature。分类为 feature 不等于允许更新奖励，纯状态/位置描述没有可操作偏好时仍可拒绝。

厨房 action 绑定取自真实可行 subgoal，trajectory 绑定需要近期轨迹，含糊、域外或没有依据的输入保守拒绝。明确禁止命令另记指令极性：目标 valence=-30、无补集惩罚，保留原始 VADER 并记录 `explicit_prohibition`。该分支、厨房绑定和适用性拒绝均为迁移规则，不能称为完全不变的原文复现。

全句更新是事务：任一短句失败则 posterior 不变、近期轨迹保留，逐句候选结果与最终 `committed` 分开记录。游戏测试验证更新确实影响随后执行的动作，并保留同一前状态、同一人类动作的反事实执行比较。

## 独立合成验收

训练作者不读取新验收集；另一代理独立编写并先封存。每轮分类器及其文本推理代码冻结后仅运行一次分类终验。失败后该套数据明确转为开发回归，下一次分类结论必须使用全新封存集。奖励更新修复后的状态回放单独记录，不冒充新的分类终验。独立作者仍是同一 AI 会话中的代理，标签不是人工验证真值。

预设门槛：speech 与 grounding macro-F1 ≥0.87 且每类 recall ≥0.85；混合句组件集合准确率 ≥0.85。另检查 20 条不明确/域外输入在无历史及合成近期历史两个上下文中均不触发实际奖励学习。

| 终验 | speech | speech Macro-F1 | grounding | 混合句组件 | 结论 |
|---|---:|---:|---:|---:|---|
| 第 1 轮 | 119/150 (79.33%) | 0.7916 | 132/150 (88.00%) | 17/20 | 不通过 |
| 第 2 轮 | 123/150 (82.00%) | 0.8155 | 115/150 (76.67%) | 18/20 | 不通过 |
| 第 3 轮原始 | 143/150 (95.33%) | 0.9533 | 146/150 (97.33%) | 18/20 (90.00%) | 通过 |
| 第 3 轮去精确重叠 | 142/149 (95.30%) | 0.9528 | 145/149 (97.32%) | 17/19 (89.47%) | 通过 |

第三轮证据为 `artifacts/synthetic-acceptance-20260906/round3-report.json` 与 `round3-clean-subset-report.json`；grounding macro-F1 分别为 0.9727 与 0.9725，两个报告的三个 gate 及 aggregate gate 均为 true。主要剩余弱项是 Evaluative + feature 组合：speech 仅 15/20 正确，容易把针对一般属性或价值的评价判成 Descriptive。总体通过不能掩盖这一弱项，也不能外推为真实玩家准确率达到 95%。

第三轮训练文本发现 2 个精确重叠：1 个单句与 1 个混合句子句，开发集精确重叠为 0。补充分析删除该单句，并保守排除含重叠子句的整条混合消息；它沿用已冻结预测，是敏感性分析，不是新一轮盲测，也不证明不存在模板或分布重叠。

恢复 full53 后，第三轮的 20 条不明确/域外输入在无历史与合成近期 onion pickup 两个上下文中仍均 0/20 实际学习，见 `round3-uncertain-stateful-full53-report.json`。这是 40 个状态化拒绝检查，不是新的分类终验。

第 1 轮曾有 5/20 含糊输入在近期历史存在时错误学习；修复后同组回归变为 0/20。第 2 轮全新不明确集在两个上下文中均 0/20 学习。不能把拒绝后的安全性当作分类正确率。

第 1 轮训练集与验收的规范化重复为 5 条（其中单句 2 条），开发集 0 条，详见 `artifacts/synthetic-acceptance-20260906/first-training-overlap.json`。原始 150 行指标不排除这些重复，失败结论不变；未宣称零泄漏。

第 2 轮的 speech 训练集与新验收无规范化重复；speech 开发集及旧 grounding 训练集各有同一条混合句子句重复，150 条单句均无重复。见 `round2-training-overlap.json`。

第一、二套覆盖六种清晰 speech/grounding 组合；训练包含的九组合不能宣称都经过独立验收。中文、讽刺及域外输入不在三分能力的已验证范围。

## 负面反馈的真实动作修复

原先只对可行策略特征构造补集，归一化后每个备选维度的负惩罚过强。审计用 `stepGame` 产生真实 onion pickup 轨迹，在同一合成烹饪快照和同一高斯 prior 下对比方法。该 prior 明确设置已有任务进度奖励为 30、其余均值为 0、协方差为 25I，是受控合成条件，不是声称由训练得到的 posterior。

`That onion pickup was bad.` 曾把决策从 GET_TOMATO / interact 反向改为 GET_ONION / right，onion−tomato 分差为 +17.5457。恢复原文完整补集后分差为 -9.0321，真实执行 interact 并取得 tomato；`terrible` 同样修复。正面 `good` 和中性反馈仍支持 onion，明确禁止句继续避开 onion。这是已证明的厨房迁移错误修复，不是对所有 prior 的普遍单调性保证。

证据保留在 `artifacts/pragmatic-negative-audit-20260906/report-before-full53.json` 与 `report-after-full53.json`。方法及六项动作回归详见 `WEB_GROUNDED_PRAGMATIC_2026-09-06.md`。恢复完整补集也意味着未提及的未来协调、安全、上下文等维度会按原文接收负向证据，不能再宣称这些维度保持不变。

## 数据与发布

隔离发布副本通过 22 个测试文件、568 项 Web 测试，数据转换器另通过 19 项 Python 测试。分类检查页对比度修正后重新通过 typecheck、lint 和最终构建的浏览器复核。浏览器中 `That was good. Please get an onion. The pot is empty.` 的显示依次为 evaluative / imperative / descriptive，grounding 依次为 trajectory / action / feature。主页面和检查页均加载新 manifest；没有浏览器运行时错误、研究请求或同意勾选。工程证据见 `artifacts/synthetic-acceptance-20260906/local-release-verification.json`。这些是工程/冒烟测试，独立分类准确率仍使用上面的封存终验。

新事件记录两个模型 hash、语义版本、原始/有效情感、是否最终生效以及策略执行回执。转换器允许新 Route1 的独立 classifier/updater hash，但要求声明匹配，拒绝来源不一致记录。预测和生成的奖励值均不是人类真值。

新增管理员 `/api/research/diagnostic`：在独立诊断表中用合成值执行写入、读回、删除，不写 session、consent、event、feedback 表。工程测试验证鉴权与事务逻辑；部署后的真实线上请求返回 200，inserted / read_back_matches / deleted 全为 true，no_research_rows_written=true。该检查证明生产 DB binding 的隔离读写链路，不等于验证真人反馈的训练质量，也不冒充完整的参与者同意与上传流程。

本地 Miniflare 实际 D1 引擎的集成检查也已通过：HTTP 200、写入与读回一致、删除后 0 行、未创建研究表。证据是 `local-d1-roundtrip.json`，其范围明确是本地。公开页面的只读基线检查仍加载旧 manifest，未勾选同意，研究接口请求数为 0。

新 trace 使用 `groundingVectorEncoding='indexed-dense-v1'`：`groundingFeatureNames` 是共享特征名数组；`groundingFeatureVectors` 中每个引用值是等长数值/`null` 数组，`null` 表示缺失，显式 0 保留。逐句 `targetFeaturesRef` / `pragmaticAlternativesRef` 指向该表，`contextRef` 指向 `groundingModelContexts` 中的 modelHash、threshold、adaptation。原始/有效情感、独立类别、分数及最终 `committed` 仍逐句保留。导出与恢复必须携带共享表，不能只保留引用。

已修复重复向量造成的上传大小问题：单事件 payload 仍限 16,384 字符，共享编码不截断或取整特征。队列按实际 UTF-8 字节将请求切批至 256 KiB 以内，避免重连后过大批次阻塞后续事件；不可容纳的单事件不会靠裁剪伪装成完整记录。

新增 `/model-check` 在浏览器本地显示逐短句三分及独立 grounding，输入不提交研究接口。发布脚本将通过的终验、模型/数据/预测/源码 hash 及状态化拒绝证据绑定，构建收据另绑定 clean Git HEAD、源码文件 SHA 和 dist SHA；源码或产物变化后要求重建。打包排除运行时 secret、`.env`（示例模板除外）、数据库和链接目录。

本轮发布指针为 `manifest-synthetic-v1.json`，旧 `manifest.json` 作为历史版本保留。源码已推送至现有 Sites 专用仓库，提交 `6ff32984ea9753fe99018b63eec851b4703675e2`；第 6 版部署状态 succeeded，线上地址为 https://durf-kitchen-lab-study.xc3083.chatgpt.site 。发布包 SHA-256 为 `fe8aeb5de5254440ff5ef75429d4abf7f9fdfb31caae9cf3b0c8006e50a07181`，只包含根 `.openai` 与 `dist/client`、`dist/server`，源码与构建密钥扫描均为 0。最初错误的归档目录被平台拒绝，未上线；修正后的包保留了平台要求的 `dist/server/index.js`，且从同一源码干净重建。

部署后公开 manifest 的 SHA-256 为 `91a3dc2611782864b6bb9a27045255ffd7d798461f538f48169c2765b43dadc4`，speech、grounding、Route2 三个文件均与验收所用字节一致。正式主页显示 `AI ready`，检查页的评价/指令/描述及独立 grounding 输出正确，无浏览器运行时错误；未勾选同意，研究请求数为 0。未授权 export 和 diagnostic 均返回 401；授权隔离诊断返回 200。本机核验脚本的默认 Python User-Agent 被托管层以 403/1010 拒绝，使用明确标识的浏览器兼容客户端后完整验证通过，未修改应用鉴权。

最终证据为 `artifacts/synthetic-acceptance-20260906/deployment-receipt.json`、`public-release-verification.json`、`public-model-check.png`、`public-after.png`。检查过程中没有下载参与者数据、创建研究会话或写入研究同意。后续训练继续使用合成训练入口，后台玩家记录仍被排除。

当前旧线上记录的读取与结构审计保留在 `WEB_BACKEND_AUDIT_2026-09-06.md`。本轮只合成数据的限制优先于旧文档中的人工标注方案，后台玩家记录不自动进入训练。
