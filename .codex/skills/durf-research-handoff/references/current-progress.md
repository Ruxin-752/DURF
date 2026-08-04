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

Snapshot date: 2026-08-04.

Repository root on the original workstation:

```text
C:\Users\my185\Desktop\研究\durf\DURF
```

Prefer repository-relative paths in code and documentation.

Git state when this skill was created:

```text
branch: dev/human-ai-feedback
HEAD: bf15653 Expand Hu conditions and collaboration event coverage
origin/dev/human-ai-feedback: 1d8aa48 agent更智能
```

A substantial Task/Coordination Hu and review implementation is uncommitted. Run the snapshot script before work and never discard these changes. The skill itself is also initially untracked until committed.

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

## 3. Verified Artifacts

Focused verification last run before this snapshot:

```text
28 focused tests passed
key modules passed py_compile
git diff --check passed except line-ending warnings
3-step headless runtime smoke test passed
offline reviewed dataset -> hierarchical Hu training smoke test passed
```

Relevant test-only artifacts:

```text
outputs/human_ai_sessions/_validation_schema_v2_20260722_211635
outputs/hu_models/_validation_hierarchical_hu_20260727
```

Do not report the validation model as a successful experimental Hu. It was trained from only three Task samples and has no valid Coordination head evidence.

## 4. Current Gaps

### P0: repository safety

- Review the uncommitted batch, rerun tests, create a clean checkpoint commit, and push it.
- Ensure model/output artifacts remain ignored unless deliberately packaged.

### P1: real pilot data

- Collect real natural-language feedback under both Task and Coordination situations.
- Obtain valid opposite Coordination pairs under different conditions:
  - `YIELD > CONTINUE_CURRENT_SUBGOAL` when the human should pass.
  - `CONTINUE_CURRENT_SUBGOAL > YIELD` when the AI is near its task target.
- Review time, event, condition, decision level, and pair labels.
- Treat 8-12 reviewed Coordination pairs as a minimum pipeline pilot, not a formal dataset.

### P2: Hu evidence

- Train both real Hu heads with session/user-aware splits.
- Compare no-Hu, shadow, and apply on repeated probes.
- Show condition sensitivity rather than a global always-yield/always-continue bias.
- Verify task completion does not materially degrade.

### P3: attribution evaluation

- Implement a unified evaluation report over automatic attribution and separate gold review.
- Compute temporal IoU, boundary error, candidate recall, event top-1/top-2 accuracy, condition F1, label exact match, and clarification quality.
- Double-annotate a pilot subset and measure inter-rater agreement.

### P4: comparison and formal experiment

- Implement the localized Linguistic Feedback baseline on the same task backbone.
- Freeze schemas, prompts, candidate sets, model settings, probes, outcomes, and non-inferiority margin before formal data collection.
- Use pilot effect sizes for participant power analysis.
- Add human-facing Likert/NASA-TLX measures and objective feedback-burden metrics.

### P5: delivery

- Verify fresh-clone setup and all documented commands.
- Package representative sample data, model metadata, evaluation reports, and one-command workflows.
- Make repository state clean and reproducible.

## 5. Ordered Next Steps

Follow this order unless a newly discovered blocker changes it:

1. Audit and commit the current implementation batch.
2. Run 2-3 structured pilot sessions that deliberately exercise opposite coordination preferences.
3. Run offline attribution with LLM and review all pilot feedback.
4. Rebuild reviewed Hu datasets and inspect domain/sample counts.
5. Train hierarchical Hu and run offline probe scoring.
6. Run shadow sessions, then cautiously enable apply with small lambdas.
7. Build the unified evaluation report.
8. Implement/freeze the localized baseline and formal protocol.

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

