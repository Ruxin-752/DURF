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

The human-AI pygame interface lives at:

```text
durf/group_a/play_with_baseline.py
```

It writes session CSV logs under:

```text
outputs/human_ai_sessions/
```

The J/K keys are retained as lightweight scalar feedback logs for inspection.
They do not update PPO and are not the main feedback mechanism for the new
LLM-assisted attribution study.

## Code status

```text
durf/
|-- baseline/
|   |-- runtime.py                RLlib agent loading and pygame env helpers
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
