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

## Real-time annotation mode

For recovery-data collection, you can let the game auto-pause every few steps
and type what the AI should have done while the map is still visible:

```powershell
python -m durf.group_a.play_with_baseline `
  --agent RllibRingCurrentBestPurePpo30Pct `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top `
  --annotation-mode `
  --annotation-interval 8
```

At each checkpoint, the annotation panel shows a visual mini-map replay of the
recent window.  Each thumbnail is the real rendered Overcooked state for that
timestep, with a label like `t-3 AI:interact YOU:north r=0`. Use a scope prefix
when it is easy:

- `now: AI should pick up the onion`
- `last3: do not keep interacting, move left to face the onion`
- `last8: this whole window did not make task progress`
- `event: you blocked my path, move down`

Then:

- press `Enter` to save and resume;
- press `Esc` to skip and resume.

The session folder will include:

- `trajectory.csv`: every environment step and state snapshot.
- `annotations.csv`: annotation text, scope hint, exact step, AI action,
  reward, state snapshot, and the recent replay shown to the player.
- `chat_messages.csv`: normal chat/LLM messages and a visible annotation trace.

Run the offline converter after the session to create JSONL files:

```powershell
python -m durf.feedback_attribution.session_converter `
  --session outputs\human_ai_sessions\<session_id>
```

The annotations become `human_annotation` records in `feedback_events.jsonl`.

## Human demonstration collection with a random collaborator

To collect BC demonstrations from a human expert, run the same pygame interface
with a random blue collaborator:

```powershell
python -m durf.group_a.play_with_baseline `
  --ai-mode random `
  --agent RllibRingCurrentBestPurePpo30Pct `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --horizon 800 `
  --step-hz 2 `
  --start-delay 3
```

You control the green player.  The blue player samples uniformly random actions.
The H0 layout starts from the full task state: empty pot, no pre-placed onion or
tomato, and one `tomato + tomato + onion` order.
The session folder still contains `trajectory.csv`, but now
`session_metadata.json` records that the collaborator was random and that the
human controlled player slot `1`.

After playing several episodes, convert only the successful episodes into BC
training data:

```powershell
python -m durf.baseline.build_human_demo_dataset `
  --session outputs\human_ai_sessions\<session_id> `
  --reference-agent RllibRingCurrentBestPurePpo30Pct `
  --policy-id ppo_1 `
  --human-player-index 1 `
  --success-threshold 1 `
  --drop-stay-fraction 0.5
```

The output `bc_dataset\human_demo_policy.npz` contains observations and human
action labels for behavior cloning.  `--drop-stay-fraction` is useful because
successful human episodes often contain many `STAY` labels; keeping all of them
can teach the model to wait too much.

To play the new `ring_tomato_onion_10x6` map after its PPO agent has been
trained:

```powershell
python -m durf.group_a.play_with_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6
```
