# Archived RLlib PPO Baseline

This module runs the trained RLlib PPO agents restored from commit
`b1e6c627`. The default pygame baseline uses:

```text
models/rllib_agents/RllibCrampedRoomSP
```

Use it only with `cramped_room`; other restored agents are map-specific assets.

Run commands from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.watch_baseline --layout cramped_room
```

Fixed-seed evaluation:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.evaluate_baseline `
  --episodes 20 `
  --output outputs\baseline_evaluation.json
```

The viewer keeps the original pygame pause/reset controls:

```text
Space = pause or resume
R     = reset
Q/Esc = quit
```

## Training a New Map Agent

The new map `ring_tomato_onion_10x6` is present locally, but its PPO agent is
not committed yet. Train a local self-play PPO agent with:

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

This installs the latest checkpoint under:

```text
models/rllib_agents/RllibRingTomatoOnion10x6SP/agent
```

After a smoke run works, increase `--iterations` for a stronger policy.

To continue improving the installed `ring_tomato_onion_10x6` agent instead of
restarting from scratch, use the stronger ring-map profile:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6SP `
  --profile paper-ring `
  --resume-installed `
  --iterations 300 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite
```

Then evaluate the installed checkpoint with:

```powershell
python -m durf.baseline.evaluate_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6 `
  --episodes 10 `
  --seed 200 `
  --output outputs\baseline_eval_ring_latest.json
```

If the sparse reward stays near zero on the target map, use curriculum training
so the policy first learns easier versions of the same 10x6 task before moving
back to the mixed tomato/onion order:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
.\scripts\train_ring_curriculum.ps1
```

The curriculum stages are now deliberately fine-grained. Each stage evaluates
the installed checkpoint before moving on, so a failed transfer is visible
early instead of after a long run:

0. `ring_tomato_onion_10x6_curriculum_micro`: same 10x6 observation size, tiny
   teaching kitchen, tomato-only soup, with temporary tomato pickup shaping.
1. `ring_tomato_onion_10x6_curriculum_micro_delivery`: tiny teaching kitchen
   with dish/server close to the pot, so the policy can learn the full
   potting-to-serving chain.
2. `ring_tomato_onion_10x6_curriculum_open_delivery`: bridge map where tomato
   moves away from the pot, while dish/server stay close enough to preserve
   the serving skill.
3. `ring_tomato_onion_10x6_curriculum_open`: open kitchen with the tomato
   dispenser close enough that the micro policy has a gentler transfer.
4. `ring_tomato_onion_10x6_curriculum_easy`: same 10x6 observation size, more
   open kitchen, tomato dispenser farther away.
5. `ring_tomato_onion_10x6_curriculum_corridor`: partial ring/corridor
   geometry, still tomato-only.
6. `ring_tomato_onion_10x6`: target ring geometry, target
   tomato-tomato-onion soup.

`ring_tomato_onion_10x6_curriculum_tomato` is kept as an older tomato-only
target-geometry diagnostic map, but it is no longer in the default curriculum.

The intended learning path is:

1. learn tomato pickup/potting;
2. learn dish/soup serving;
3. preserve those skills while increasing distance and corridor structure;
4. add onion only after the tomato-only chain survives transfer.

If you want a tomato-only target-geometry diagnostic after stage 4, run:

```powershell
python -m durf.baseline.evaluate_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6_curriculum_tomato `
  --episodes 10 `
  --seed 200 `
  --output outputs\baseline_eval_ring_tomato_diag.json
```

For a shorter smoke test, pass smaller iteration counts:

```powershell
.\scripts\train_ring_curriculum.ps1 `
  -Stage0Iterations 20 `
  -Stage1Iterations 20 `
  -Stage2Iterations 20 `
  -Stage3Iterations 20 `
  -Stage4Iterations 20 `
  -Stage5Iterations 20 `
  -EvalEpisodes 3
```

If a nested PowerShell loses the conda environment, pass the Python executable
explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_ring_curriculum.ps1 `
  -PythonExe "C:\Users\my185\Miniconda3\envs\durf310\python.exe"
```

## Current Ring Tomato-Onion Findings

For the newer counter-prefix curriculum map:

```text
ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top
```

the best pure PPO checkpoint so far is now frozen as:

```text
models/rllib_agents/RllibRingCurrentBestPurePpo30Pct/agent
```

Fixed-seed evaluation:

```powershell
python -m durf.baseline.evaluate_baseline `
  --agent RllibRingCurrentBestPurePpo30Pct `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top `
  --episodes 20 `
  --seed 2300 `
  --output outputs\baseline_eval_scan_stabilize20_checkpoint_000021_20eps.json `
  --skip-layout-check
```

Observed result: `6/20` successful deliveries, i.e. `30%` success. This improves
on the earlier `RllibRingSeparateCounterPrefixBest20Pct` checkpoint (`4/20`,
`20%`) and the previous `RllibRingSeparateCounterPrefixExactMicro` checkpoint
(`3/20`, `15%`).

The current `30%` checkpoint came from this sequence:

1. Start with the stable suffix model `RllibRingSeparateInitSmoke`.
2. Apply a mild supervised warm start for the four-action counter-prefix
   behavior, producing `RllibRingSeparateInitSmokeBCWarmStartMild`
   (`6/20`, `30%`).
3. Continue with a short, low-learning-rate PPO stabilization run and scan
   intermediate checkpoints. `checkpoint_000021` was the best scanned point
   (`5/10` in the quick scan, `6/20` in the fixed-seed re-evaluation), and is
   installed as `RllibRingCurrentBestPurePpo30Pct`.

Longer stochastic evaluations show that this checkpoint is still unstable, but
it remains the best pure PPO candidate overall.  For example, on seed range
`3201..3240` it reached `17/40` successful deliveries (`42.5%`), but with
`14` soup drops.  On seed range `3001..3040` it reached `13/40` (`32.5%`).

Other recent warm-start variants are informative but not better:

- `RllibRingSeparateCounterPrefixBCWarmStart`: strong full-model supervised
  warm start from the older `20%` PPO, `0/20`.
- `RllibRingSeparateInitSmokeBCWarmStart`: strong full-model supervised warm
  start from the stable suffix model, `3/20`.
- `RllibRingSeparateInitSmokeBCWarmStartLogits`: only trains the action-logits
  head (`ppo_0/dense_3`), `4/20`. This reduces soup drops compared with the
  full-model update, but does not beat the mild full-model checkpoint.
- `RllibRingSeparateInitSmokeBCMildPpoStabilize20`: final checkpoint after 20
  PPO iterations, `5/20`. Its intermediate `checkpoint_000021` is better and
  was frozen as `RllibRingCurrentBestPurePpo30Pct`.

## Subgoal Executor Backbone

Because plain PPO and direct `state -> action` imitation tended to memorize
local map positions instead of learning the task structure, the current H0
backbone path is:

```text
state -> rule/planner subgoal -> learned low-level executor action
```

The full-task H0 layout is:

```text
ring_tomato_onion_10x6_h0_full_task
```

It starts from an empty pot/counter and requires a full
`tomato + tomato + onion` soup.  The current p0 executor for repeated-delivery
backbone testing is:

```text
outputs/subgoal_executors/h0_p0_rule_executor_v5_fast_two_delivery_recovery/executor.keras
```

Training data:

```text
outputs/subgoal_executor_datasets/h0_p0_clean_jitter_recycle_two_delivery_v5/dataset.npz
```

This dataset combines clean one-delivery rollouts, successful recovery rollouts
after random start jitter, and two-delivery `safe_recycle` rollouts.  It
contains `8554` examples with these subgoals:

```text
GET_TOMATO, PUT_TOMATO_IN_POT, GET_ONION, PUT_ONION_IN_POT,
GET_DISH, PICKUP_SOUP, SERVE_SOUP, WAIT
```

Important implementation detail: `collect_rule_teacher_dataset.py` and
`evaluate_subgoal_executor.py` now seed the start-jitter RNG per episode using
`episode_seed`.  This makes failed or successful recovery starts reproducible
from a single seed.

Another important rule-teacher fix: the motion-planner labeler now rejects
plans that move into walls or pass through the other player.  If the planner
cannot provide a valid first step, a grid BFS fallback supplies a feasible
first action toward the relevant feature.  This fixed the repeated-delivery
failure where p0 delivered soup at the service window and then got stuck trying
to walk through the wall or through p1.

Current online checks for `h0_p0_rule_executor_v5_fast_two_delivery_recovery`:

- Two-delivery cycle with `partner_mode=dynamic_avoid`: `5/5`, seed `16700`,
  horizon `320`, success threshold `40`, exactly `185` steps per episode.
- Strong jitter recovery with `partner_mode=dynamic_avoid`:
  `8/8`, seed `16600`, horizon `300`, success threshold `20`,
  `start_jitter_steps=120`.

The older recovery-focused executor remains useful as a reference:

```text
outputs/subgoal_executors/h0_p0_rule_executor_v3_recovery60/executor.keras
```

It reached `10/10` on `start_jitter_steps=120` with `partner_mode=safe_corner`
before the repeated-delivery data was added.  The v5 model is better for
two-delivery cycles, while the v3 result shows that stronger one-delivery
recovery is possible.

`dynamic_avoid` is a conservative state-aware partner mode.  It stays still by
default, but if it occupies the teacher/executor's immediate next tile, it
moves to a nearby valid tile that clears the path.  This fixed the jitter
failure where p1 blocked the service exit at `(8, 4)` and p0 could not leave
the serving window.

These results mean the low-level executor can now complete repeated H0
deliveries when given rule-planner subgoals and a simple dynamic non-blocking
partner.  It does **not** yet mean we have solved p1 symmetry, arbitrary human
blocking, richer human-like partner behavior, or a learned high-level planner.
Those are the next stages.

Re-run the current two-delivery check with:

```powershell
python -m durf.baseline.evaluate_subgoal_executor `
  --executor outputs\subgoal_executors\h0_p0_rule_executor_v5_fast_two_delivery_recovery\executor.keras `
  --reference-agent RllibRingCurrentBestPurePpo30Pct `
  --layout ring_tomato_onion_10x6_h0_full_task `
  --policy-id ppo_0 `
  --teacher-player-index 0 `
  --partner-mode dynamic_avoid `
  --episodes 5 `
  --seed 16700 `
  --horizon 320 `
  --success-threshold 40 `
  --output outputs\subgoal_executors\h0_p0_rule_executor_v5_fast_two_delivery_recovery\online_eval_two_delivery_dynamic_avoid_seed16700_n5_h320.json
```

Other diagnostic attempts since then:

- Deterministic / argmax evaluation is not appropriate for these checkpoints.
  `RllibRingCurrentBestPurePpo30Pct` gets `0/20` with `--deterministic`: it
  reliably places the onion in the pot but then gets stuck before serving.
  The hybrid suffix model has the same issue under argmax.  These policies are
  genuinely stochastic policies, not argmax policies.
- Temperature sampling was added to `evaluate_baseline.py` and
  `evaluate_prefix_hybrid.py`.  The hybrid policy remains stable at lower
  temperatures (`0.35`, `0.5`, `0.75` all gave `10/10` in the quick scan), and
  lower temperature reduced unnecessary onion/dish drops.  For the pure PPO
  checkpoint, however, temperatures below `1.0` reduced success, so the current
  pure PPO evaluation should stay at default stochastic sampling.
- Hybrid trajectory distillation was tested with
  `durf.baseline.distill_hybrid_policy`.  A full-model distillation from the
  successful hybrid teacher destroyed the opening skill (`0/20`).  A very mild
  logits-only distillation preserved behavior but was effectively identical to
  the current best (`13/40` on the same seed range).  A stronger logits-only
  distillation returned to about `6/20`, so it is not better than the current
  checkpoint.
- Low-entropy PPO fine-tuning is available through `--entropy-start` and
  `--entropy-end` in `train_rllib_agent.py`.  A low-entropy fine-tune reduced
  soup drops (`3` vs `14` on seed range `3201..3240`) but did not improve
  success rate (`16/40` vs `17/40` for the current best).  Intermediate
  low-entropy checkpoints did not reveal a better candidate in the quick scan.
- A mid-entropy fine-tune (`--entropy-start 0.06 --entropy-end 0.03`) was also
  tested as a compromise between the default high-entropy policy and the
  low-entropy run.  It reached `14/40` on seed range `3401..3440`, compared
  with `13/40` for `RllibRingCurrentBestPurePpo30Pct` on the same seeds.
  This is only a marginal gain, and the checkpoint scan did not show a
  consistently stronger intermediate model, so the current best was not
  replaced.
- Reverse-prefix curriculum support was added to `train_rllib_agent.py` and
  `OvercookedMultiAgent`.  During training, `--reset-prefix-prior
  counter_onion_to_pot_top --reset-prefix-length N` can execute the first `N`
  prior actions after every reset, then hand control to PPO.  A four-stage run
  used lengths `3 -> 2 -> 1 -> 0`, with targeted shaping on the length-1 stage
  (`--onion-to-pot-distance-reward 0.2 --onion-drop-penalty -1.0`).  This
  strongly reduced useless onion drops in the final policy (`3` drops vs `34`
  for the current best on seed range `3601..3640`), but it also reduced success
  rate (`7/40`, `17.5%`, vs `9/40`, `22.5%`, for
  `RllibRingCurrentBestPurePpo30Pct` on the same seeds).  Do not promote
  `RllibRingReversePrefixCurriculum` to the main baseline yet.
- Mixed reverse-prefix curriculum was also added through
  `--reset-prefix-lengths`, which samples a prefix length uniformly at every
  reset.  A `0/1` mixed run from the length-1 checkpoint kept useless onion
  drops low (`3` drops on seed range `3601..3640`) but reached only `6/40`
  successes (`15%`).  This confirms the same diagnosis: the targeted curriculum
  can suppress bad object-drop behavior, but it still does not make the policy
  reliably initiate the full counter-onion prefix from the raw start state.
- Intermediate checkpoints from the reverse-prefix and mixed-prefix runs were
  scanned as temporary agents.  Several reached `5/20` on seed range
  `3601..3620`, but the best two candidates both fell to `8/40` (`20%`) on the
  full `3601..3640` seed range.  They therefore remain below
  `RllibRingCurrentBestPurePpo30Pct` (`9/40`, `22.5%`) despite lower onion-drop
  counts.
- Policy-id override diagnostics were added to `evaluate_baseline.py` through
  `--policy-id-0` and `--policy-id-1`.  For the current best checkpoint,
  forcing both players to use `ppo_0` gave only `1/20`, and forcing both to use
  `ppo_1` gave `0/20`.  The default asymmetric pairing (`ppo_0` for player 0,
  `ppo_1` for player 1) is still better, so the issue is not solved by simply
  sharing one policy across both player slots.
- A ppo_1-only assist fine-tune from the current best was tested as
  `RllibRingPpo1AssistFromBest`: `ppo_0` was frozen and only `ppo_1` was
  trained with mild object-drop penalties.  It matched the current best success
  rate (`9/40`, `22.5%`) but increased onion drops (`54` vs `34` on the same
  seed range), so it was not promoted.  Intermediate checkpoints from that run
  were also scanned and did not exceed `4/20`.
- A broader mixed-prefix run from the current best was tested as
  `RllibRingMixedPrefix0123FromBest`, using reset-prefix lengths `0/1/2/3` and
  joint `ppo_0`/`ppo_1` training.  The training-time metrics looked healthier
  (`sparse_reward_mean=12.08`, low onion-drop rate), but the installed policy
  reached only `5/40` (`12.5%`) when evaluated from the true raw start state.
  This indicates that the apparent improvement came from the assisted reset
  distribution rather than from a robust raw-start opening policy.

DAgger-style recovery diagnostics were added next.  The collection script lets
the current student policy visit its own states, then labels selected failed
episode states with either a teacher policy or a motion-planner task scaffold:

```powershell
python -m durf.baseline.collect_recovery_dataset `
  --student-agent RllibRingCurrentBestPurePpo30Pct `
  --teacher-agent RllibRingSeparateInitSmoke `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top `
  --episodes 24 `
  --seed 7400 `
  --max-examples 1200 `
  --stride 1 `
  --disagreement-only `
  --max-label-fraction 0.55 `
  --labeler task_rule `
  --output-dir outputs\recovery_datasets\recovery_task_rule_disagree_seed7400_n24
```

The resulting directory contains `dataset.npz` for supervised updates,
`dataset.jsonl` for inspecting labeled states, and `metadata.json` for settings
and summary counts.  Training uses:

```powershell
python -m durf.baseline.train_recovery_policy `
  --student-agent RllibRingCurrentBestPurePpo30Pct `
  --dataset outputs\recovery_datasets\recovery_task_rule_disagree_seed7400_n24\dataset.npz `
  --agent-name RllibRingRecoveryTaskRuleSeed7400Light `
  --epochs 10 `
  --learning-rate 5e-7 `
  --train-scope logits `
  --overwrite
```

Results so far:

- `prefix_then_teacher` labeler: a 192-example disagreement dataset trained
  with logits-only supervision reached `8/20` on seed range `7200..7219`,
  tying the current best on the same seeds.
- `task_rule` labeler: a 1002-example disagreement dataset produced
  `RllibRingRecoveryTaskRuleSeed7400Logits` at `8/20` on seed range
  `7500..7519`, while the current best reached `9/20` on the same seeds.
- A lighter logits-only update,
  `RllibRingRecoveryTaskRuleSeed7400Light`, reached `9/20` on the same seeds,
  effectively preserving the current best but not improving it.
- A very light full-model update,
  `RllibRingRecoveryTaskRuleSeed7400AllLight`, dropped to `8/20`, confirming
  that full-model supervised recovery can still damage a useful stochastic
  success path.

Current interpretation: the recovery-data pipeline now works, and it is safer
than earlier full hybrid distillation, but automatic scaffold labels are still
too coarse to improve raw-start success.  The current best pure PPO remains
`RllibRingCurrentBestPurePpo30Pct`.  The next useful step is likely a small
human or manually reviewed recovery dataset for ambiguous failure states,
rather than simply increasing supervised update strength.

For manual review, export a compact packet from the large recovery dataset:

```powershell
python -m durf.baseline.export_recovery_review_packet `
  --dataset-jsonl outputs\recovery_datasets\recovery_task_rule_disagree_seed7400_n24\dataset.jsonl `
  --output-dir outputs\recovery_review_packets\task_rule_seed7400_n24 `
  --max-examples 60 `
  --per-episode 6 `
  --min-step-gap 8
```

This creates:

- `human_review_packet.md`: a readable checklist of candidate failure states.
- `human_review_candidates.jsonl`: the same candidates with editable
  `human_review` fields.
- `review_packet_summary.json`: source path and action-count summary.

After filling `human_review.preferred_ai_action` for selected examples, compile
the reviewed labels back into a supervised dataset:

```powershell
python -m durf.baseline.build_reviewed_recovery_dataset `
  --source-jsonl outputs\recovery_datasets\recovery_task_rule_disagree_seed7400_n24\dataset.jsonl `
  --source-npz outputs\recovery_datasets\recovery_task_rule_disagree_seed7400_n24\dataset.npz `
  --reviewed-jsonl outputs\recovery_review_packets\task_rule_seed7400_n24\human_review_candidates.jsonl `
  --output-dir outputs\recovery_review_packets\task_rule_seed7400_n24\compiled_reviewed
```

Use `--include-reasonable-scaffold` if reviewed examples marked
`is_scaffold_label_reasonable: true` should keep the scaffold action even when
`preferred_ai_action` is blank.

Earlier PPO fine-tuning attempts regressed:

- `RllibRingSeparateCounterPrefixBest20PctSuffixStabilize`: ppo_1-only suffix
  stabilization with soup delivery/drop shaping, `0/20`.
- `RllibRingSeparateCounterPrefixBest20PctGentleFineTune`: very low learning
  rate joint fine-tune with mild shaping, `0/20`.

Do not treat the latest model directory as the best checkpoint. The current
evidence suggests that this map's hard part is not just reward scale tuning:
the opening counter-prefix behavior is too brittle for more naive PPO shaping.
The next promising direction is still a more targeted action-prior or
behavior-cloning warm start for the short prefix, followed by the already
stable PPO suffix. The latest evidence suggests that full PPO fine-tuning can
reduce soup drops, but it still does not make the opening counter-prefix skill
reliable enough for a stable pure PPO baseline.

A diagnostic hybrid baseline is available to show that the remaining hard part
is the short counter-prefix behavior. It uses the action prior
`counter_onion_to_pot_top`, which runs a four-action scripted prefix for agent 0:

```text
interact -> east -> north -> interact
```

then hands control to the stable PPO model `RllibRingSeparateInitSmoke`. This is
not a pure PPO agent, but it verifies that the PPO suffix policy can reliably
finish the task once the counter onion has been placed in the pot:

```powershell
python -m durf.baseline.evaluate_prefix_hybrid `
  --agent RllibRingSeparateInitSmoke `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top `
  --prior counter_onion_to_pot_top `
  --episodes 20 `
  --seed 1600 `
  --output outputs\baseline_eval_action_prior_init_smoke_counter_top.json
```

Observed result: `20/20` successful deliveries, with `20` soup pickups,
`20` soup deliveries, and `0` soup drops.

The same prior can be enabled in the pygame human-AI interface:

```powershell
python -m durf.group_a.play_with_baseline `
  --agent RllibRingSeparateInitSmoke `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top `
  --ai-action-prior counter_onion_to_pot_top
```

This is useful as a debugging and demo candidate, while pure PPO training should
continue to focus on learning the four-step counter-prefix skill without
destroying the already stable pot-to-serving suffix policy.
