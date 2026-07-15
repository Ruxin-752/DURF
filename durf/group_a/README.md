# Group A: Human + H0/PPO Pygame Baseline

Run the bundled H0 planner + Keras subgoal executor as the blue agent on the
full 10x6 ring tomato/onion task, and control the green agent:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.group_a.play_with_baseline
```

The default command is equivalent to:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode subgoal_executor `
  --subgoal-executor models/subgoal_executors/h0_rule_executor_v5/executor.keras `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --horizon 800
```

This mode now reproduces the original `4ccb410` H0 playtest behavior exactly:
the rule teacher selects one of eight high-level subgoals and MotionPlanner
supplies the low-level action for every valid subgoal. The bundled Keras model
is loaded and contract-checked as in the source branch, but that branch's
runtime returns the planner action before `model.predict`. Cooperative
collision/yield and stale-object guards are also enabled. Layout hot switching
is disabled for the fixed H0 task.

H0 execution requires the project TensorFlow dependency (`tensorflow==2.10.1`).
Use the repository environment from `pyproject.toml`; the PPO-only mode loads
Ray lazily and is not required for H0.

Controls:

- `WASD` or arrow keys: move the green human agent
- `Space`: interact
- chat input: record natural-language feedback for attribution
- `P`, `Tab`, `F1`, or the on-screen button: pause or resume
- `R`: reset the episode
- `N` / `M`: switch among layouts compatible with a PPO agent
- `Q` or `Esc`: quit

Session data is written to `outputs/human_ai_sessions/<timestamp>/`.

The active Group A interface no longer uses J/K scalar feedback. It records
language feedback, step-level trajectory state snapshots, actions, rewards, and
candidate events for the LLM-assisted attribution pipeline.

The default interaction rate is two environment steps per second with a
three-second preparation countdown. To slow it further:

```powershell
python -m durf.group_a.play_with_baseline `
  --step-hz 1.5 --start-delay 5
```

The older standalone scalar keyboard listener has been archived under
`archive/legacy_scalar_feedback/`.

To run the previous large-ring late-stage PPO:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode ppo `
  --agent RllibRingHalfTaskStableTopLeft `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left
```

`RllibRingHalfTaskStableTopLeft` is a stable second-half cooking baseline: it
starts from the final onion stage, places the onion, picks up a dish, collects
ready soup, and serves it. It is not a full original-start policy for
`ring_tomato_onion_10x6`.

To run the older cramped-room PPO baseline:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode ppo `
  --layout cramped_room
```
