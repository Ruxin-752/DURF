# DURF Project Guide

This repository contains four relevant layers:

1. The upstream Overcooked-AI environment under `src/overcooked_ai_py/`.
2. DURF research code under `durf/`.
3. Restored historical RLlib agent assets under `models/rllib_agents/`.
4. Archived legacy/reference material under `archive/`.

## Current trusted baseline

The pygame baseline uses the RLlib PPO agents restored from commit `b1e6c627`.
The default agent is `models/rllib_agents/RllibCrampedRoomSP`, which should be
used with the `cramped_room` layout.

The new layout `ring_tomato_onion_10x6` is present locally, but its matching
PPO agent is not committed yet. The intended local agent name is:

```text
models/rllib_agents/RllibRingTomatoOnion10x6SP
```

The human-AI pygame interface lives at:

```text
durf/group_a/play_with_baseline.py
```

It writes session CSV logs under:

```text
outputs/human_ai_sessions/
```

The active interface records language feedback plus step-level trajectory
snapshots. J/K scalar feedback has been archived and should not be used as the
main feedback mechanism for the new LLM-assisted attribution study.

## Code status

```text
durf/
|-- baseline/
|   |-- runtime.py                RLlib agent loading and pygame env helpers
|   |-- train_rllib_agent.py      Local self-play PPO training wrapper
|   |-- watch_baseline.py         Pygame RLlib PPO viewer
|   `-- evaluate_baseline.py      Fixed-seed RLlib PPO evaluation
|-- group_a/
|   `-- play_with_baseline.py     Human + RLlib PPO pygame interface
`-- group_b/
    `-- README.md                 Planned LLM-assisted feedback attribution
```

Legacy standalone scalar-feedback utilities were moved to:

```text
archive/legacy_scalar_feedback/
```

The original tutorial notebook was moved to:

```text
archive/legacy_notebooks/
```

The old Docker/browser demo under `src/overcooked_demo/` has been removed from
the active branch. The project now uses the pygame renderer from
`src/overcooked_ai_py/visualization/`.

## Next active layer

Immediate engineering priority:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6SP `
  --iterations 2 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite
```

This is a smoke training run. If it produces a loadable agent, increase
`--iterations` for a stronger policy.

New research work should start by creating:

```text
durf/feedback_attribution/
```

This package should turn natural-language feedback and recorded trajectories
into structured attribution samples for `H_u`:

```text
feedback text + trajectory facts + candidate events
-> LLM semantic attribution
-> structured sample for H_u
-> PPO task policy + H_u preference model
```

Programmatic code should handle factual state extraction and candidate event
detection. The LLM should only handle human-semantic interpretation that rules
cannot reliably resolve.

## Development rule

Use the outer repository only. Do not initialize another `.git` directory inside
a project subdirectory.

New research work should be developed on feature branches from this branch.
Generated logs, participant data, and new training checkpoints should stay
outside version control unless a checkpoint is intentionally promoted as a
shared model asset under `models/`.
