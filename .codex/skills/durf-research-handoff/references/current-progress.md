# Current Progress and Open Work

## Contents

1. Snapshot identity
2. Completed work
3. Verified artifacts
4. Current gaps
5. Ordered next steps
6. Delivery gates
7. Update rule

## 1. Snapshot Identity

Snapshot date: 2026-08-10 (P0 checkpoint commit; freeze-v0/v1 batch committed).

Repository root on the original workstation:

```text
C:\Users\my185\Desktop\研究\durf\DURF
```

Prefer repository-relative paths in code and documentation.

Git state after P0 commit:

```text
branch: refactor/trim-coordination-vocabulary
HEAD: 85f7b86 Checkpoint: freeze-v0/v1 config, sim-human pipeline, condition decoupling, per-participant protocol
origin/dev/human-ai-feedback: c773cdc Implement hierarchical Hu pipeline and complete feedback data flow
```

Working tree is clean. Previous uncommitted batch (freeze-v0, freeze-v1, sim pipeline, attribution keyword fixes, per-participant protocol, probe domain routing, event detector hardening) is now committed as `85f7b86`.

## 2. Completed Work

### Task backbone

- Added the full-task H0 ring layout with empty pot and no pre-staged ingredients.
- Explored PPO curriculum, successful-trajectory reuse, BC, recovery data, rule teacher, and hybrid execution.
- Established a practical subgoal-based task backbone that can complete the complex recipe with a human.
- Fixed or mitigated stale subgoals, unneeded ingredients, dish-before-ready behavior, counter-object use, pickup/drop loops, and several recovery cases.

### Logging and attribution

- Record human/AI positions, actions, held objects, map terrain, pot/object state, rewards, candidate subgoals, and state-before/state-after.
- Record natural-language chat feedback; legacy J/K files are archived or read only for compatibility.
- Convert session CSV to JSONL and generate candidate events without using feedback-future frames.
- Integrate DeepSeek with auditable prompts/responses, clarification fields, fallback to deterministic baseline, and schema proposals.
- Expand candidate events, actor/valence classification, condition features, and useful-counter/labor-division evidence.

### Review and probes

- Build a Tkinter review window with concise evidence, legal subgoal multiselect, schema actions, and map-based trajectory replay.
- Keep automatic attribution immutable; save human decisions separately.
- Consume approved review decisions in Hu dataset construction.
- Detect reusable probe states independently of feedback timing.

### Hierarchical Hu

- Separate Task and Coordination decision domains.
- Emit explicit runtime Task and Coordination decision logs.
- Implement bounded `YIELD` with short commitment and cooldown.
- Implement `HierarchicalHu` with independent linear pairwise heads and legacy single-head loading.
- Reject mixed-domain training pairs.
- Support shadow scoring and `--hu-apply` runtime behavior changes.

### 2026-03 P0/P1/3 + 4b batch (uncommitted)

- P0 coordination probe no-op: `probe_state_detector.py` gains `domain`; `probe_candidate_pool()` splits by domain, coordination probes read the coordination candidate pool (YIELD/CONTINUE) instead of the task pool.
- P0 new trigger surface: `path_conflict_type` extended in `coordination.py` (human blocked but target tile is not the AI's tile); new event `AI_failed_to_yield_or_clear_path` registered and mapped in `subgoal_preferences.py`; `play_with_baseline.py` extended condition features and yield-action choice via `action_moves()`.
- P1: probe definitions aligned with the runtime candidate pools (`pot_ready_ai_should_get_dish` etc.); P4 probes trimmed to 2-3 frames each.
- P1/3: 7 streak-event sites in `event_detectors.py` now use the decision-start frame for condition features/time/moment fields.
- Candidate pool: `collect_rule_teacher_dataset.py` "holding unneeded item" branch adds `PUT_DOWN_OBJECT`.

### 2026-03-14 freeze + sim 数据质量批次（uncommitted）

- 新增 `docs/experiment_config_freeze_v0.md`：冻结数据协议/协调词汇/17 事件 taxonomy/Hu 词汇/28 布尔条件/8 probe/训练协议/变更治理。训练前配置不再改动，改动需升级 freeze-vN。
- 协调让路先验落地（用户决策：condition-sensitive，双向保留）：`build_coordination_candidates` 在 `ai_adjacent_to_current_subgoal_target` 时反转先验（CONTINUE>YIELD），默认 YIELD>CONTINUE，Hu 可翻转任一。
- 新增 `durf/group_a/sim_session.py`：headless sim-human 会话（cooperative/selfish persona），与真实会话同 schema，`data_source: synthetic_sim_human`；反馈模板四向（YIELD/CONTINUE × 正/负）。
- 归因 keyword 对齐（数据质量）：`sample_builder.EVENT_KEYWORDS` 注册 4 个协调事件（blocked / failed_to_yield_or_clear_path / successfully_yielded / maintained_current_subgoal）；sim 模板文本与之对齐。
- 修复归因 bug：`AI_maintained_current_subgoal_during_conflict` 缺默认映射，正面坚持反馈曾被错误标为 `preferred=[YIELD]`；已补 `preferred=[CONTINUE_CURRENT_SUBGOAL]`。
- 4b: per-participant train/eval enforced in `train_subgoal_reranker.py`: `--test-dataset` requires every test user to have training data (ValueError otherwise), warns when a test feedback predates the participant's latest training feedback, drops reused `source_feedback_id` samples, dedupes exact pairs inside training, and writes `test_metrics_by_user` plus `protocol_checks` into `metadata.json`.

### 2026-03-14 sim 双 head 数据批次（uncommitted，freeze-v0 之后）

- sim 接入 task 反馈（用户决策：coordination 和 task 都用 AI 数据）：`sim_session.py` 新增 `_TASK_NEG_READY_POT` / `_TASK_NEG_PREP_WAIT` 模板，`SimHuman.maybe_feedback` 增加 task-event streak 检测（阈值与 `event_detectors` 对齐：ready-pot >=3 步、prep-wait >=4 步），模板文本经 keyword 匹配核对无跨事件歧义。cooperative 会话反馈产出 2 → 11 条/会话。
- coordination 二元域单边补全：`hu_dataset_builder.build_training_samples` 对 `decision_level == coordination` 且只有单边 preferred/rejected 的 provenance 自动补另一边（`CONTINUE_CURRENT_SUBGOAL>YIELD` 或反向）。修复"cooperative 正向坚持反馈无法进训练集"的已知局限。
- 批量生成 8 会话（cooperative/selfish × seed 7/11/13/17，300 步 × 2 episodes）：共 76 pairs = 48 task + 28 coordination。task 覆盖 `GET_DISH>PUT_DOWN_OBJECT`、`PICKUP_SOUP>PUT_DOWN_OBJECT`、`GET_DISH>GET_TOMATO/GET_ONION`、`GET_USEFUL_INGREDIENT>WAIT`；coordination 双向覆盖（cooperative 会话产 CONTINUE>YIELD，selfish 会话产 YIELD>CONTINUE）。
- 双 head 训练 `outputs/hu_models/sim_both_v0`（train 4 会话 seed 7/11，test 4 会话 seed 13/17，protocol_ok=true）：
  - task head：test 25 pairs，accuracy 0.92，Wilson 95% CI [0.75, 0.98]，mean_margin 6.60 —— 强信号达成。
  - coordination head：test 12 pairs，accuracy 0.67，Wilson 95% CI [0.39, 0.86] —— CI 下界低于随机，不构成信号。
  - coordination 错误诊断：4/12 全错在 `AI_maintained_current_subgoal_during_conflict`（CONTINUE>YIELD），且这 4 条与正确预测的同源样本条件特征**完全相同**（`ai_adjacent_to_current_subgoal_target=true, ai_on_human_path=true`）→ 条件同质重复对，模型学出的全局偏差方向随 persona 混合翻转，同条件正反标签互相抵消。加样本量不解决：该条件只有一类反馈。
- Wilson 置信区间功率分析（z=1.96，CI 下界 > 0.65 视为强信号）：acc 0.90 需 13 test pairs；acc 0.85 需 21；acc 0.80 需 37；acc 0.75 需 82。task head 25 pairs（acc 0.92，CI [0.75,0.98]）已达强信号标准。
- coordination 的瓶颈是**条件覆盖**而非数量：`ai_adjacent=true + on_path=true` 下需要反向标签（本批次全同向），或按 P4 opposite-condition probe 协议分条件评估准确率。

## 3. Verified Artifacts

Focused verification last run after the 2026-03 batch:

```text
41 focused tests passed (36 pre-existing + 5 new: dedupe sibling pairs,
  exact-duplicate drop, protocol rejects unseen user, temporal warning,
  per-user evaluation)
key modules passed py_compile
real-session end-to-end (20260722_211635): candidate_events 73 -> 92,
  probe_hits 240 -> 346, new AI_failed_to_yield_or_clear_path x12 with
  target_was_ai=false, streak event condition features confirmed at
  decision-start frame (39/48 tomato-held), coordination probe hits carry
  domain=coordination and YIELD/CONTINUE vocabulary
per-participant train/test smoke on real data: train 20260716_134436
  (2 task pairs after vocabulary filtering) -> test 20260722_211635
  (14 task pairs after dedupe), protocol_ok=true, no temporal warnings,
  test task pairwise_accuracy 0.64; coordination head has NO real samples
sim 全链路验证（2026-03-14, sessions 20260806_045237 selfish /
  20260806_045432 cooperative）: sim -> 归因 -> 训练 全通；
  selfish 5 反馈 -> 4 协调 pair（YIELD>CONTINUE，条件分布有区分度）；
  cooperative 正向反馈归因正确（单边，需 review 补边）；4 pair 训练
  smoke: coordination head 3 train / 1 val, pairwise_accuracy 1.0
  （样本极少，仅链路信号非实验证据）

### 2026-03-14 首次 Hu 训练（sim + 真实数据，per-participant 协议）

- coordination head（sim，`outputs/hu_models/sim_coord_v0`）:
  train 20260806_045237（3 train / 1 val）-> test 20260806_052031
  （同 persona selfish、同 PILOT01 新会话, 4 test pair）;
  test pairwise_accuracy 1.0 / mean_margin 5.89; protocol_ok=true。
  数据质量复核：两个会话各 4 条 pair 全部与 coordination 决策记录一致
  （label YIELD>CONTINUE 对应 decision selected=CONTINUE, AI 坚持被骂）。
  已知局限：全部 8 条都是 selfish persona 单向 YIELD>CONTINUE 负反馈，
  top weight `ai_adjacent_to_current_subgoal_target -> YIELD promotes /
  CONTINUE discourages` 与冻结先验方向相反——模型只学了"冲突就让路"，
  未学到"AI 靠近目标时坚持"的反向偏好（cooperative 正向 CONTINUE
  反馈为单边 preferred，需 review 补 rejected 边才能进训练集）。
- task head（真实数据，`outputs/hu_models/real_task_v0`）:
  train 20260716_134436（2 train / 0 val）-> test 20260722_211635
  （14 test pair）, test pairwise_accuracy 0.64; protocol_ok=true。
  样本过少，0.64 仍是弱信号。
- 结论：pipeline 端到端可用、协议强制生效、标签与决策一致；
  但两个 head 都不构成实验证据（coordination 单方向数据、
  task 仅 2 train 样本）。sim 数据可先用，需补反向条件反馈。

### 2026-03-14 sim 双 head（`outputs/hu_models/sim_both_v0`）

- 数据：8 sim 会话（cooperative/selfish × seed 7/11/13/17，
  20260806_133421~133527），76 pairs = 48 task + 28 coordination。
- 训练：train 4 会话（seed 7/11）-> test 4 会话（seed 13/17），
  protocol_ok=true，无 temporal warnings。
- task head test：25 pairs，acc 0.92，Wilson 95% CI [0.75, 0.98]，
  mean_margin 6.60 —— 强信号达成（25 pairs > acc 0.90 所需 13 对）。
- coordination head test：12 pairs，acc 0.67，Wilson 95% CI [0.39, 0.86]，
  mean_margin 1.61 —— CI 下界低于随机 0.5，不构成信号。
  12 对中 4 错全为 CONTINUE>YIELD（AI_maintained...），其条件特征与
  正确预测的同源样本完全相同（adj_target=true, on_path=true）：
  条件同质 + 正反标签同条件抵消 → 全局偏差翻转。加样本不解决。
- 结论：sim task 数据已可用且信号强；sim coordination 数据的瓶颈是
  条件覆盖（需对同一条件同时采集 YIELD>CONTINUE 与 CONTINUE>YIELD
  反向反馈），不是样本总量。
```

### 2026-03-14 freeze-v1：条件解耦 + 人格矩阵（`outputs/hu_models/sim_both_v1`）

- 根因证据（v0 协调数据 31 样本）：`ai_adjacent` 与 `ai_on_human_path`/`human_trying_to_pass`
  在 v0 代码里被赋同一布尔（恒等特征）；adjacent=True 时 P/T 共现 70%；
  A=0 样本仅 13%（selfish push 触发要求曼哈顿距离==1，AI 几乎必然贴目标）。
- 解耦改动（全部在 `sim_session.py`，freeze-v1，见 `docs/experiment_config_freeze_v1.md`）：
  1. `_ai_blocks_target_line` 扩为两步检测 + `_step_toward_action`，AI 距目标 2 格也能制造冲突；
  2. 在线 `AiRuntime.step` 拆分 `ai_on_human_path`（几何事实，含静止时用 last_delta 投影）
    与 `human_trying_to_pass`（当步移动意图）；
  3. `_evaluate_coordination` 极性重排：挡路+远离目标→必让；挡路+一步到位→按人格；
  4. 人格矩阵扩展 polite（礼貌行为+严格极性）/ lenient（礼貌行为+极端宽容），
     `--sim-human` 4 值：cooperative/selfish/polite/lenient。
- 条件分布变化（v1 数据 30 协调样本）：adjacent=True 时 P/T 共现 70%→46%；
  A=1 非挡路 25.8%→43.3%；A=0 占比 12.9%→20%；A=1 桶标签 14Y/10C（多样性保留）。
- 训练对比（同协议 train coop/selfish×seed7/11 → test ×seed13/17）：
  - task head：0.92（25 pairs，与 v0 持平）；
  - coordination head：**0.667 → 0.923**（12→13 pairs），mean_margin +1.61→+4.60，
    min_margin -4.82→-2.74。判定解耦生效，实行改动。
- 跨人格泛化（sim_both_v1 在 4 persona × 4 会话上）：cooperative 1.000 /
  selfish 0.950 / polite 0.900（未见）/ lenient 1.000（未见），
  未见人格组合 ≥0.90，学到的是条件→极性规律而非记忆 persona。
- 残留（如实记录）：v1 coordination head 的 `ai_on_human_path→CONTINUE +1.51`、
  `human_trying_to_pass→CONTINUE +1.50` 仍为正方向——这是"挡路+坚持被表扬"
  （cooperative/lenient 极性）的自然结果，模型正确地学到"人格决定极性"。
  彻底消除需按 persona 分流训练或加 persona 条件特征，属后续工作。
- 环境修复：`durf310` env 的 `__editable__.overcooked_ai-2.0.0.pth` 原为 GBK 编码
  （含中文路径），Python 3 site 初始化 UTF-8 解码崩溃；已重写为 UTF-8（备份 .bak）。
  运行 sim/归因/训练需 `$env:PYTHONPATH="$PWD;$PWD\src"`。
- 56 focused tests 通过（feedback_attribution / coordination / baseline_task_logic / review_replay）。

Relevant test-only artifacts:

```text
outputs/human_ai_sessions/_validation_schema_v2_20260722_211635
outputs/hu_models/_validation_hierarchical_hu_20260727
outputs/hu_models/smoke_sim_coord
outputs/hu_models/sim_coord_v0
outputs/hu_models/real_task_v0
outputs/hu_models/sim_both_v0
```

Do not report the validation model as a successful experimental Hu. It was trained from only three Task samples and has no valid Coordination head evidence. The 0.64 test accuracy above is also a smoke signal on 2 train / 14 test samples, not experimental evidence. The `smoke_sim_coord` model is a pipeline signal only (4 sim pairs), not experimental evidence. `sim_coord_v0` / `real_task_v0` are protocol-valid runs but remain weak evidence (coordination data single-direction, task train n=2).

## 4. Current Gaps

### P0: repository safety ✅

- ~~Review the uncommitted batch, rerun tests, create a clean checkpoint commit, and push it.~~
- **Done (2026-08-10)**: 56 focused tests pass. Full batch committed as `85f7b86` on `refactor/trim-coordination-vocabulary`. Working tree clean. Model/output artifacts remain `.gitignore`-d. Push pending.

### P1: real pilot data

- Collect real natural-language feedback under both Task and Coordination situations.
- Obtain valid opposite Coordination pairs under different conditions:
  - `YIELD > CONTINUE_CURRENT_SUBGOAL` when the human should pass.
  - `CONTINUE_CURRENT_SUBGOAL > YIELD` when the AI is near its task target.
- Review time, event, condition, decision level, and pair labels.
- Treat 8-12 reviewed Coordination pairs as a minimum pipeline pilot, not a formal dataset.
- Coordination head 无真实训练样本；20260722 probe hits 全部 `evaluation_unavailable`
  （轨迹早于运行时决策日志）。sim 会话（synthetic_sim_human）已有 4 条协调 pair
  可用，按用户决策先试用 sim 数据，真实 pilot 仍是正式证据来源。

### P2: Hu evidence

- Train both real Hu heads with session/user-aware splits (the per-participant
  `--test-dataset` protocol is implemented).
- Compare no-Hu, shadow, and apply on repeated probes.
- Show condition sensitivity rather than a global always-yield/always-continue bias.
- Verify task completion does not materially degrade.

### P3: attribution evaluation

- Implement a unified evaluation report over automatic attribution and separate gold review.
- Compute temporal IoU, boundary error, candidate recall, event top-1/top-2 accuracy, condition F1, label exact match, and clarification quality.
- Double-annotate a pilot subset and measure inter-rater agreement.

### P4: comparison and formal experiment

- Freeze schemas, prompts, candidate sets, model settings, probes, outcomes, and non-inferiority margin before formal data collection. **已完成**：`docs/experiment_config_freeze_v0.md`（2026-03-14 冻结）。<br>后续改动须升级 freeze-vN。
- Implement the localized Linguistic Feedback baseline on the same task backbone.
- Use pilot effect sizes for participant power analysis.
- Add human-facing Likert/NASA-TLX measures and objective feedback-burden metrics.

### P5: delivery

- Verify fresh-clone setup and all documented commands.
- Package representative sample data, model metadata, evaluation reports, and one-command workflows.
- Make repository state clean and reproducible.

## 5. Ordered Next Steps

Follow this order unless a newly discovered blocker changes it:

1. ~~Audit and commit the current implementation batch.~~ **Done** (`85f7b86`, 2026-08-10).
2. **← CURRENT**: Run sim sessions at scale (several seeds/personas) to grow coordination pairs, or switch to real pilot sessions; sessions must carry decision logs.
3. Run offline attribution with LLM and review all pilot feedback.
4. Rebuild reviewed Hu datasets and inspect domain/sample counts.
5. Train hierarchical Hu (sim data first per user decision) and run offline probe scoring on decision-logged sessions.
6. Run shadow sessions, then cautiously enable apply with small lambdas.
7. Build the unified evaluation report.
8. Implement the localized baseline; formal protocol is frozen in freeze-v0.

Per-participant split (4b), probe/event alignment (P0/P1/3), freeze-v0, freeze-v1, sim pipeline, and attribution keyword fixes are committed. sim_both_v1 coordination head at 0.923. Next bottleneck: coordination condition coverage (need reverse-preference data under same conditions for the coordination head).

## 6. Delivery Gates

Use these as preliminary engineering gates, then freeze final thresholds before formal experiments:

```text
cross-domain labels = 0
JSON parse rate = 100%
trajectory/Task-decision completeness >= 99%
candidate recall >= 0.90 on pilot gold
event top-1 accuracy target >= 0.75
condition F1 target >= 0.80
Task and Coordination validation pairwise accuracy target >= 0.75
opposite-condition probe ranking accuracy target >= 0.80 per condition
task success relative degradation <= predeclared 10% provisional margin
```

Scientific results must include confidence intervals/effect sizes; gates alone are not evidence.

## 7. Update Rule

Update this file after any of these events:

- checkpoint commit or branch change;
- real pilot collection;
- schema freeze or major event/condition revision;
- first real Task or Coordination Hu model;
- evaluation script completion;
- baseline implementation;
- formal experiment start or delivery.

Record dates, commit IDs, artifact paths, sample counts, tests, and what remains unproven.

