# DURF Project Guide

This repository contains two distinct layers:

1. The upstream Overcooked-AI environment under `src/overcooked_ai_py/`.
2. DURF research code under `durf/`.

## Current trusted baseline

The trusted starting point is commit `1dbbd2c` (`update keyboard_listener.py`).
It provides a standalone feedback listener with this protocol:

| Key | Signal |
| --- | ---: |
| J | +1 |
| K | -1 |
| Space | 0 |
| Q | Quit |

The listener writes the latest signal to `reward_signal.txt` and appends events
to `feedback_log.csv`. These runtime files are now ignored by Git.

## Code status

```text
durf/
├── group_a/
│   ├── keyboard_listener.py      Trusted standalone listener
│   └── examples/                 Sanitized output examples
└── group_b/
    └── README.md                 Reserved for later LLM integration
```

The previous `group_a_baseline/` implementation is not considered a successful
end-to-end baseline. Its Git history and files have been preserved separately:

- Branch `archive/keyboard-stable` points to `1dbbd2c`.
- Branch `archive/experimental-demo` points to `875622d`.
- The nested repository metadata is backed up outside this repository.
- A local ignored copy is stored under `archive_local/`.

## Development rule

Use the outer repository only. Do not initialize another `.git` directory
inside a project subdirectory.

New research work should be developed on feature branches from this reorganized
branch. Generated logs, checkpoints, participant data, and model files must stay
outside version control.
