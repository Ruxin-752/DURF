# DURF Overcooked Pygame Baseline

This branch keeps the DURF pygame workflow and the trained RLlib PPO agents
that were committed in the project history by Ruxin. Docker and browser demo
code has been removed from the active path.

## What Is Kept

- `durf/`: project-specific pygame viewers, human-AI play, and feedback logging.
- `src/overcooked_ai_py/`: Overcooked environment, layouts, pygame renderer,
  graphics, and fonts.
- `src/human_aware_rl/`: the RLlib loader needed to restore historical PPO
  checkpoints.
- `models/rllib_agents/`: restored historical agents, including
  `RllibCrampedRoomSP`, `RllibCoordinationRingSP`,
  `RllibForcedCoordinationSP`, `RllibAsymmetricAdvantagesSP`, and
  `RllibCounterCircuit1OrderSP`.
- `archive/`: legacy scalar-feedback utilities and tutorial notebooks kept for
  reference, not active development.

## Default Agent

The pygame baseline defaults to:

```text
models/rllib_agents/RllibCrampedRoomSP
```

It was trained for the `cramped_room` layout. Other restored agents are kept as
model assets for future map-specific training and integration.

## Run Pygame Viewer

Run commands from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.watch_baseline --layout cramped_room
```

Useful smoke-test option:

```powershell
python -m durf.baseline.watch_baseline --layout cramped_room --max-steps 10
```

## Play With The PPO

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.group_a.play_with_baseline --layout cramped_room
```

Controls:

```text
WASD / Arrows = move
Space         = interact
P / Tab / F1  = pause or resume
R             = reset
J / K         = log scalar feedback only
Q / Esc       = quit
```

Session logs are written under `outputs/human_ai_sessions/`.

## Next Research Layer

The next active development target is an LLM-assisted feedback attribution
package under `durf/feedback_attribution/`. The pygame baseline should provide
trajectory logs and candidate events; the LLM should only handle human-semantic
interpretation that deterministic code cannot resolve reliably.
