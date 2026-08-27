# Web 玩家数据存储与重新训练说明

## 数据现在存在哪里

浏览器先把事件放在当前页面的内存队列中；断网时会重试，但关闭页面后未上传的数据可能丢失。匿名参与者 UUID 保存在浏览器 `localStorage`。同步成功后，数据写入公开站点绑定的 Cloudflare D1 数据库 `DB`，不写进 Git 仓库。

D1 的主要研究表为：

- `sessions`：匿名用户和会话 ID、同意版本、开始/结束时间、客户端与 schema 版本；
- `events`：按序号保存游戏动作、逐步状态、反馈事件、模型 hash 与 route trace；新 `durf-web-event-v2` 每 0.5 秒保存 `stateBefore/stateAfter`、请求与执行的联合动作、reward、done 和结构化事件；
- `feedback`：玩家原话、Route1/Route2、三类完整概率、逐短语预测和低置信标记；
- `research_session_tokens`、`research_rate_limits`、`research_security_secrets`：会话验证和防滥用，不进入训练导出。

研究表不主动写入姓名、邮箱或原始 IP。托管平台仍可能保留必要的运维日志。当前 D1 没有自动过期、玩家删除接口或自动备份，正式招募前需要先确定保留期限与删除流程。

## 如何导出

管理员接口为 `GET /api/research/export`，必须提供保存在 Sites secret 和本机 `web/.env.local` 中的 `ADMIN_EXPORT_TOKEN`。它返回 JSONL，每行的 `recordType` 为 `session`、`event` 或 `feedback`。不要把管理员令牌提交到 Git，也不要公开训练导出文件。

当前导出实现会一次性在内存中拼出整个 JSONL；数据量变大后应改成分页或流式导出。

## 现在会不会自动重新训练

不会。当前仓库没有“D1 导出 → 训练格式转换 → 自动训练 → 自动部署”任务，这是有意的安全边界。线上模型的预测标签不能直接当作真实标签回灌，否则模型会重复学习自己的错误。

## 三分类的正确训练流程

1. 从 D1 导出并校验 `schemaVersion`、`modelHash`、事件唯一性和同意版本。
2. 按 `anonymousUserId/sessionId` 去重和分组，移除测试提交与个人信息。
3. 由不知道模型预测结果的人工标注者，把每条短语标成 `Evaluative`、`Imperative` 或 `Descriptive`；分歧样本需要复核。
4. 按参与者或会话切分 train/dev/frozen-test，不能让同一玩家同时进入训练集和测试集。
5. 将人工确认结果转换为训练脚本要求的 `language` 与 `classification_label`，再使用 `prepare_human_feedback_form_annotations.py` 和 `train_feedback_form_classifier.py`。
6. 只在 dev 上选模型和校准；frozen-test 只评一次。通过准确率、每类召回率、macro-F1、ECE 和浏览器/Python 概率一致性后，再人工批准新模型。
7. 导出新的版本化浏览器 artifact，更新 model hash，灰度部署并保留旧版本以便回滚。

## Route2 的正确训练流程

普通玩家的一句话和当前模型算出的 53 维权重都不是 Route2 的监督真值。Web v2 已开始逐步保存可回放的状态、联合动作、reward 和 done；旧 Web 会话没有这些完整字段。Route2 监督训练仍需要：

- 与反馈对应的轨迹窗口；
- 独立收集或人工确认的完整 53 维 `teacher_reward_weights`、`teacher_id` 与 `reward_config_id`。

补齐独立 reward assignment 并实现 D1 导出转换器后，才能生成 `export_human_route2_corpus.py` 所需输入，按参与者独立切分，训练十模型 ensemble，并在真人 frozen set 上验证。现有 Web 文本仍只能先作为未标注语言池、游戏行为分析和人工标注候选，不能直接宣称已用于 Route2 监督训练。

## 推荐闭环

`D1 分页导出 → schema/hash 校验 → 去重与脱敏 → 人工标注/奖励配置 → participant-disjoint split → 离线训练 → frozen eval + calibration + replay → 人工审批 → 版本化部署/回滚`

不要边玩边自动替换生产模型。
