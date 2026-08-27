# Group A: Human + H0/PPO Pygame Baseline

Run the bundled H0 planner + Keras subgoal executor as the blue agent on the
full 10x6 ring tomato/onion task, and control the green agent:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline
```

The default command now opens the paper-aligned Route 2 feedback path:

```powershell
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2 `
  --teacher-id player_001 `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --horizon 800
```

The Route 2 agent chooses among feasible high-level subgoals with `w * phi`,
then uses the same H0 MotionPlanner for low-level actions. The legacy
`--ai-mode subgoal_executor` remains available for reproducing the original
`4ccb410` rule-teacher playtest. Layout hot switching is disabled for this fixed
H0 task.

H0 execution requires the project TensorFlow dependency (`tensorflow==2.10.1`).
Use the repository environment from `pyproject.toml`; the PPO-only mode loads
Ray lazily and is not required for H0.

For paper-aligned Route 2 human feedback collection:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2 `
  --teacher-id player_001 `
  --horizon 800
```

The feedback flow is deliberately modal: play first, click `暂停并反馈`, describe
the immediately preceding behavior, press `Enter` to record and apply the
semantic update, inspect the result, then click `继续游戏`. The dedicated right
sidebar supports Chinese IME composition, cursor editing, `Ctrl+V`, and
`Shift+Enter`; it does not cover the game view. The current Route 2 model is
English-first: Chinese is accepted and logged, but English should be used for
reliable online semantic updates.

The feedback panel displays the complete Evaluative / Imperative / Descriptive
score vector. The current TF-IDF logistic-regression scores are explicitly
marked `NOT CALIBRATED`; they are useful for ranking and low-score abstention,
but are not presented as measured correctness. If the model artifact is
missing, the syntax fallback shows a rule label without a made-up percentage.
These UI strategy scores are separate from the learner update trace.

The kitchen now uses the built-in pixel renderer with distinct stations,
players, held objects, soup state and cooking progress. It renders at integer
nearest-neighbour scale so resizing does not blur pixel edges.

The window and terminal show `ROUTE2｜在线学习已开启` in the default adaptive
mode. `--comfort-feedback-mode frozen` is a control condition: the UI shows
`FROZEN 对照模式｜只记录，不学习`, feedback is logged with `status=ignored`, and
no learner state can be resumed. Do not use `frozen` when testing online learning.
Route 2 uses the selected v5 ten-fold checkpoint, starts from a neutral reward, and
chooses subgoals only through `w * phi`; H0 resolves exact reward ties.
Feasibility is computed from live resource counts and geometry: an item already
staged for the next order is not immediately re-collected, and `YIELD_PATH` is
offered only when the human is visibly trying to enter the AI's occupied tile.
Neither behavior is keyed to feedback wording; all remaining feasible choices
are still ranked by `w * phi`. The chat reply reports the actual weight update
and current re-ranking, rather than promising a hard-coded future action.
Successful Route 2 comments are persisted to a privacy-preserving,
participant-specific path such as
`outputs/live/route2_state_v5_teacher_<hash>.json`. Continue the same human
preference profile with the same `--teacher-id`:

```powershell
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2 `
  --teacher-id player_001 `
  --resume-learner-state
```

Do not use `--resume-learner-state` when starting an independent participant.
Anonymous sessions receive session-specific state files and cannot implicitly
resume; use a stable pseudonymous `--teacher-id` for longitudinal play.

Audit a completed session with:

```powershell
python -m durf.group_a.summarize_session outputs/human_ai_sessions/<session_id>
```

The summary includes reward, WAIT streaks, feedback-update statuses, and
immediate `STASH -> GET_*` reversal counts.

Controls:

- `WASD` or arrow keys: move the green human agent
- `Space`: interact
- `暂停并反馈`: pause at the exact trajectory step and focus language input
- `Enter`: submit; `Shift+Enter`: newline; `Ctrl+V`: paste
- `继续游戏`, `P`, `Tab`, or `F1`: resume after reviewing the update
- `R`: reset the episode
- `N` / `M`: switch among layouts compatible with a PPO agent
- `Q` or `Esc`: quit

Session data is written to `outputs/human_ai_sessions/<timestamp>/`.
Summarize the latest session in one command (or append a session directory):

```powershell
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.summarize_session
```

The summary reports mode, online-learning state, reward, WAIT percentage and
longest WAIT streak, plus feedback update statuses.

The active Group A interface no longer uses J/K scalar feedback. It records
language feedback, step-level trajectory state snapshots, actions, rewards, and
candidate events for the LLM-assisted attribution pipeline.

The default interaction rate is two environment steps per second with a
three-second preparation countdown. To slow it further:

```powershell
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --step-hz 1.5 --start-delay 5
```

The older standalone scalar keyboard listener has been archived under
`archive/legacy_scalar_feedback/`.

To run the previous large-ring late-stage PPO:

```powershell
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
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
& "E:\miniconda\envs\durf310\python.exe" -m durf.group_a.play_with_baseline `
  --ai-mode ppo `
  --layout cramped_room
```
