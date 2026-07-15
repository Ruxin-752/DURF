# H0 Rule Executor V5

This directory contains the current recommended bundled agent artifact for H0 human-AI playtesting.

## Contents

- `executor.keras`: subgoal executor model trained from rule-teacher rollouts.
- `metadata.json`: training metadata, subgoal labels, label counts, and validation metrics.

## Recommended Command

Run from the repository root:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --subgoal-executor models\subgoal_executors\h0_rule_executor_v5\executor.keras `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --step-hz 2 `
  --start-delay 3 `
  --horizon 800
```

## Notes

This is not a pure PPO checkpoint. It is the current stable H0 playtest agent used with the planner/subgoal flow in `durf.group_a.play_with_baseline`.

The original training output was copied from:

`outputs/subgoal_executors/h0_p0_rule_executor_v5_fast_two_delivery_recovery/`
