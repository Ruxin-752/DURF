# Web 后台与训练数据审计（2026-09-06）

核验时间：2026-09-06 08:25（Asia/Shanghai）。本报告区分线上读取、已有写入记录、本地转换测试与新写入验证；本次没有创建研究会话、接受同意、提交伪参与者数据或部署站点。

## 线上证据

- Sites project：`appgprj_6a8fcd0cf29081919d04d5c852953347`。
- 线上地址：[DURF Kitchen Lab](https://durf-kitchen-lab-study.xc3083.chatgpt.site)，public，Sites 返回最新版本号 5，站点更新时间 2026-08-27。
- 主页 HTTP 200，`/models/manifest.json` HTTP 200。
- `/api/research/export` 无凭据 HTTP 401；使用现有管理员令牌 HTTP 200。令牌只在内存使用，未输出。
- 受保护导出 3,685,568 字节，SHA-256：`2b6c66a551d22da8dcfaa34cdc7a7d04c48f723d75b631e3161f98ba27d60443`。原文只在内存校验，未保存到报告或公开目录。
- Sites 的 DB binding 可读，包含 `sessions`、`events`、`feedback` 及三个研究安全表。安全表内容未读取，未进入研究导出。
- 最近 24 小时错误日志查询为空；这只表示查询窗口未返回错误，不代表曾发生新写入。

| 统计 | 完整导出 | 当前同意版本 v3 |
| --- | ---: | ---: |
| sessions | 13 | 5 |
| events | 3,240 | 2,296 |
| feedback | 21 | 14 |
| 匿名参与者 | 4 | 1 |
| 可生成盲标短语 | 23 | 14 |

完整数据中 3 个匿名参与者提交过反馈，19 条反馈走 Route1，2 条走 Route2。线上预测的类别计数是 Evaluative 16、Imperative 3、Descriptive 2；这些是模型预测，不是人工标签，也不是准确率证据。旧同意版本有 7 条反馈，当前 `durf-anonymous-research-en-2026-08-27-v3` 有 14 条。

反馈写入时间范围为 2026-08-27 13:44:36 至 2026-09-04 10:10:13（北京时间）。导出验证没有孤立 event/feedback，没有重复 session + sequence。共有 1,904 条 tick summary，其中 1,827 条带前后状态；13 个会话只有 3 个记录了结束。

因此可以确认：线上已有成功写入，当前数据库读取和受保护导出正常。最近记录停留在 9 月 4 日，本次未触发新的研究数据写入；不能据此声称 9 月 6 日或下一次部署后的浏览器 → API → D1 新写入已经验证。

## 线上与当前本地版本差异

| 项目 | 线上 manifest | 审计时本地 manifest |
| --- | --- | --- |
| manifest schema | durf-browser-model-manifest-v1 | durf-browser-model-manifest-v2 |
| feedback artifact SHA-256 | db5da937ef483a25e92940e8195c7352b6c9f09d0ea3c2e358470cbb6721cc9d | 2280f07b71b5790bbd53a90cea38dd4e84fa9e7f1cf34be1510e304b85bdcb48 |
| Route2 artifact SHA-256 | fa9ddb2aff1a451713e62417fcd3bca56a9e06987fbc6d7dbec2cf1651b80177 | fa9ddb2aff1a451713e62417fcd3bca56a9e06987fbc6d7dbec2cf1651b80177 |

随后实际下载了线上 feedback artifact，其 SHA-256 与线上 manifest 一致。对 `classes`、`transformers`、`classifier`、`calibration`、`minimum_confidence` 五组推理字段分别比对，全部相等；两端 canonical JSON SHA-256 均为 `87ab79ed940eada93d372488029a5b69a03698d524cdf3dc5b9988973e26355b`。`canonical_labels`、`model_type`、`model_version` 和 artifact schema 也相同，source 元数据不同。因此本地和线上具有相同的这组模型推理参数；分句、聚合和 UI 等不同客户端行为仍需分别测试。本地仍用 `durf-web-event-v2`，旧线上数据包含 v1/v2；相同 event schema 字符串不保证所有客户端字段相同。

线上 21 条反馈均有 `classifierModelHash` 和 `updaterModelHash`，其中两条的 event hash 与 feedback hash 不同，符合 Route2 双模型来源。线上反馈全部缺少 `recentTrajectoryFeatures`；线上 tick summary 全部缺少新的 `receiptSchemaVersion`。本地客户端已有这些字段，尚不能用旧线上记录证明它们已采集成功。

## 已修复的本地数据转换问题

`prepare_web_retraining_data.py` 修复了以下问题：

1. 原导入器把 event 的 updater hash 与 feedback 的 classifier hash 强制相等，拒绝合法 Route2 记录。现在对 payload 声明分别严格比对，要求小写 64 位 SHA-256，只有 Route2 允许不同 hash；`selectedRoute` 存在时必须与 feedback.route 一致。保留两种来源，部分声明或不一致时拒绝。完全无双模型声明的旧格式仍保持原有共享 hash 规则。
2. 原导入器只取单帧 `gameSnapshot.features`，忽略客户端实际使用的 `recentTrajectoryFeatures`。现在优先保留实际轨迹向量，拒绝非法数值；旧 snapshot fallback 明确标为 `trajectory_is_exact_model_input=false`，不伪造过去未记录的输入。
3. 新增显式 `--filter-consent-version`，按完整会话集合筛选并保留关联，记录源数量、排除数量和选中内容 hash。原 `--consent-version` 的严格拒绝语义保留，两者互斥。

对真实导出只在内存重放：完整结构审计成功，生成 23 个待标注任务和 21 个无监督 Route2 候选；新同意版本过滤成功，排除 8 sessions、944 events、7 feedback，保留 14 个待标注任务。选中记录 canonical JSON SHA-256 为 `4c197d964b5f1ad7dcd0e0cf8d37115b166fab645930062192ced4d3a056b897`。本次未写出任务原文、未标注、未训练。

## 训练可用性的结论

可以继续积累数据并建立盲标池，但目前不能直接使用线上预测作为三分类真值。当前同意版本只有 1 个匿名参与者，稳定 participant-disjoint 切分的 14 条任务全部属于 train，dev/frozen-test 都为空。完整数据的 23 个任务也全部进入 train；不能人为移动同一参与者到测试集补数量。

可靠的三分类重训练需要独立人工标注、分歧裁决，以及更多独立参与者构成的 dev 和冻结测试集。Route2 还缺完整 53 维独立 `teacher_reward_weights`、`reward_config_id` 和合法目标来源；游戏分数、线上后验或当前模型输出不能补成真值。现有历史向量只有单帧 snapshot，不能声称已经恢复当前 Route2 的近期轨迹输入。

本地队列仍是内存队列，关闭页面可能丢失未上传数据；该限制已有文档，不属于本次转换器修复。长时间离线和超过两小时的会话令牌续期也仍需独立验证，不能把短回合的模拟测试推广为所有断网场景都可靠。

## 验证

- Python `test_web_retraining_data`：18/18 通过，覆盖双模型 hash 正确匹配/篡改/部分缺失/格式/route 一致性、实际轨迹优先/空向量/非法值、同意版本过滤/关联/审计 hash，以及既有盲标、固定切分与拒绝未审核标注。
- Web `research-security.test.ts` 和 `tick-receipt-validation.test.ts`：19/19 通过，使用本地模拟数据。首次沙箱 esbuild `spawn EPERM`，授权提升后相同命令通过。
- 真实线上完整导出和所选同意版本的转换只在内存执行成功，没有把模拟测试写进 D1。

实际命令和训练步骤见 [WEB_DATA_RETRAINING.md](WEB_DATA_RETRAINING.md)。本报告不构成三分类语义达标或新 Web 发布完成的证明。
