# DURF Project Guide

This repository contains three active layers:

1. The upstream Overcooked-AI environment under `src/overcooked_ai_py/`.
2. DURF research code under `durf/`.
3. Restored historical RLlib agent assets under `models/rllib_agents/`.

## Current trusted baseline

The pygame baseline uses the RLlib PPO agents restored from commit `b1e6c627`.
The default agent is `models/rllib_agents/RllibCrampedRoomSP`, which should be
used with the `cramped_room` layout.

The human feedback protocol in `durf/group_a/` is:

| Key | Signal |
| --- | ---: |
| J | +1 |
| K | -1 |
| Space | 0 |
| Q | Quit |

The human-AI pygame interface writes session CSV logs under
`outputs/human_ai_sessions/`. These runtime files are ignored by Git.

## Code status

```text
durf/
├── baseline/
│   ├── runtime.py                RLlib agent loading and pygame env helpers
│   ├── watch_baseline.py         Pygame RLlib PPO viewer
│   └── evaluate_baseline.py      Fixed-seed RLlib PPO evaluation
├── group_a/
│   ├── keyboard_listener.py      Trusted standalone listener
│   ├── play_with_baseline.py     Human + RLlib PPO pygame interface
│   └── examples/                 Sanitized output examples
└── group_b/
    └── README.md                 Reserved for later LLM integration
```

The previous `group_a_baseline/` implementation is legacy code and is not part
of the active branch. Its Git history remains available:

- Branch `archive/keyboard-stable` points to `1dbbd2c`.
- Branch `archive/experimental-demo` points to `875622d`.
- The nested repository metadata is backed up outside this repository.

The old Docker/browser demo under `src/overcooked_demo/` has been removed from
the active branch. The project now uses the pygame renderer from
`src/overcooked_ai_py/visualization/`.

## Development rule

Use the outer repository only. Do not initialize another `.git` directory
inside a project subdirectory.

New research work should be developed on feature branches from this reorganized
branch. Generated logs, participant data, and new training checkpoints should
stay outside version control unless a checkpoint is intentionally promoted as a
shared model asset under `models/`.
