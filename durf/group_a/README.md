# Group A: Human + PPO Pygame Baseline

Run the archived RLlib PPO as the blue agent and control the green agent:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.group_a.play_with_baseline --layout cramped_room
```

Controls:

- `WASD` or arrow keys: move the green human agent
- `Space`: interact
- chat input: record natural-language feedback for attribution
- `P`, `Tab`, `F1`, or the on-screen button: pause or resume
- `R`: reset the episode
- `N` / `M`: switch among layouts compatible with the selected agent
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

To play the new `ring_tomato_onion_10x6` map after its PPO agent has been
trained:

```powershell
python -m durf.group_a.play_with_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6
```
