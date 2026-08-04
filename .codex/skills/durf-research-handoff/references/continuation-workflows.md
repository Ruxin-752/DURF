# Continuation Workflows

## Contents

1. Environment
2. Snapshot and tests
3. Play and collect
4. Attribute and review
5. Train and inspect Hu
6. Shadow and apply
7. Data quality checks
8. Git handoff

## 1. Environment

Use Python 3.10. The package declares `>=3.10,<3.11` and pins legacy Ray/TensorFlow versions.

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
```

If editable installation is missing:

```powershell
python -m pip install -e .
```

For DeepSeek calls:

```powershell
$env:DEEPSEEK_API_KEY="<key>"
```

Never store the key in source, logs, documentation, or commits.

## 2. Snapshot and Tests

Run from repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\.codex\skills\durf-research-handoff\scripts\project_snapshot.ps1
```

Run snapshot plus focused tests:

```powershell
powershell -ExecutionPolicy Bypass -File .\.codex\skills\durf-research-handoff\scripts\project_snapshot.ps1 -RunTests
```

Direct focused suite:

```powershell
python -m unittest `
  testing.feedback_attribution_test `
  testing.coordination_test `
  testing.baseline_task_logic_test `
  testing.review_replay_test
```

Also run `python -m py_compile` for touched modules and `git diff --check` before commit.

## 3. Play and Collect

Start the current main backbone without Hu:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --step-hz 2 `
  --start-delay 3
```

Use natural-language chat feedback. Do not use J/K as the main signal.

For the immediate Coordination pilot, deliberately sample both:

```text
human carrying dish through a constrained path -> feedback favoring YIELD
AI adjacent/near current task target -> feedback favoring CONTINUE
```

Record exact session directory printed by the runtime.

## 4. Attribute and Review

Run complete offline attribution:

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 30 `
  --use-llm
```

Open review:

```powershell
python -m durf.feedback_attribution.review_session `
  --session outputs\human_ai_sessions\<session_id>
```

Review in this order:

1. Time window.
2. Behavior event.
3. Conditions at the decision, not merely at feedback time.
4. Decision level.
5. Preferred/rejected candidates from the same domain.
6. Whether the record is suitable for Hu training.
7. Whether a schema proposal should be accepted, revised, mapped, or rejected.

After review, rerun the offline pipeline or dataset builder so approved decisions are consumed. Confirm `hu_dataset_summary.json` reports review records consumed and inspect pairwise records directly.

## 5. Train and Inspect Hu

Train:

```powershell
python -m durf.hu.train_subgoal_reranker `
  --dataset outputs\human_ai_sessions\<session_id> `
  --output-dir outputs\hu_models\pilot01_v1 `
  --epochs 200 `
  --learning-rate 0.05
```

Before trusting the model, inspect `metadata.json`:

```text
samples by Task and Coordination
train/validation pairwise accuracy by level
mean margin by level
top condition weights by level
```

A model with a missing Coordination head is not a complete hierarchical Hu. A high training accuracy on three records is only a smoke test.

Offline scoring:

```powershell
python -m durf.hu.score_subgoals `
  --model outputs\hu_models\pilot01_v1\hierarchical_hu.json `
  --decision-level coordination `
  --user-id PILOT01 `
  --condition-file <condition.json> `
  --candidate-subgoals YIELD CONTINUE_CURRENT_SUBGOAL
```

Use `--condition-file` in PowerShell to avoid JSON quoting errors.

## 6. Shadow and Apply

Shadow mode records Hu scores but does not alter choices:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --hu-model outputs\hu_models\pilot01_v1\hierarchical_hu.json `
  --hu-user-id PILOT01
```

After offline/shadow checks, enable small bounded influence:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --hu-model outputs\hu_models\pilot01_v1\hierarchical_hu.json `
  --hu-user-id PILOT01 `
  --hu-lambda 0.25 `
  --hu-coordination-lambda 0.25 `
  --hu-apply
```

Increase lambdas only after probe and task-safety evidence. Never choose a final lambda only because it produces the desired anecdote.

## 7. Data Quality Checks

For every pilot batch, verify:

```text
trajectory and Task decision completeness
feedback -> episode/step mapping
candidate recall against review gold
no event evidence after feedback timestep
event actor and valence
condition values and missingness
same-domain pair legality
review decisions consumed
Task/Coordination sample counts
probe hits and candidate availability
task reward/delivery outcomes
```

Keep automatic attribution files unchanged. Compare them with review decisions when computing accuracy.

## 8. Git Handoff

Before commit:

1. Run the snapshot and focused tests.
2. Inspect `git diff --stat`, `git diff --check`, and every staged path.
3. Confirm no API keys, raw participant identifiers, huge outputs, temporary validation models, or accidental checkpoints are staged.
4. Include source, tests, current docs, and this skill when they belong to the same coherent milestone.
5. State clearly whether packaged models are deliberate and license/privacy-safe.
6. Commit with a behavioral summary, then push the intended branch.
7. Update `references/current-progress.md` with commit ID and next blocker.

