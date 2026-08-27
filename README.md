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
  `RllibCounterCircuit1OrderSP`. The new `ring_tomato_onion_10x6` map is
  present, but its PPO agent must be trained locally first.
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

## Play And Collect Human Feedback

Use the complete `durf310` environment and the Route 2 learner:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2 `
  --teacher-id player_001 `
  --horizon 800
```

Controls:

- `WASD` / arrows: move
- `Space`: interact
- `暂停并反馈`: pause at the current trajectory step and focus the feedback editor
- `Enter`: submit feedback; `Shift+Enter`: newline; `Ctrl+V`: paste
- `继续游戏`: resume only after reviewing the semantic update
- `R`: reset; `Esc`: quit

The feedback editor uses a Chinese-capable font and Windows IME composition. It
occupies a dedicated right sidebar and never covers the game view. The current Route 2
training corpus is English-first, so use English for reliable online updates; Chinese
input is still recorded correctly. Session logs
are written under `outputs/human_ai_sessions/`.

The active research interface uses natural-language feedback for attribution.
J/K scalar feedback belongs to the archived legacy path and should not be used
as the main Group A signal.

## Train New Map Agent

To smoke-train the missing PPO self-play agent for `ring_tomato_onion_10x6`:

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

After the smoke run works, increase `--iterations` for a more useful policy.

## Next Research Layer

The next active development target is an LLM-assisted feedback attribution
package under `durf/feedback_attribution/`. The pygame baseline should provide
trajectory logs and candidate events; the LLM should only handle human-semantic
interpretation that deterministic code cannot resolve reliably.
