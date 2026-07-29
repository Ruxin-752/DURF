# Adapted Overcooked Baseline B

This folder contains the Overcooked adaptation of Baseline B from *Learning Rewards from Linguistic Feedback*.

The first runnable version is intentionally offline and deterministic:

```text
linguistic feedback
-> feedback type classification
-> sentiment extraction (valence)
-> Overcooked feature grounding (reference vector)
-> Bayesian conjugate update of a Gaussian reward-weight belief
-> probe state evaluation (posterior mean + sampled policy)
```

It does not modify the PPO policy or the live pygame game loop. It faithfully
reproduces the paper's Bayesian reward learner: reward weights are a Gaussian
belief (`N(0, 25)` prior) updated by conjugate observations, mirroring
`science/agents/beliefs.py` and `MultivariateNormalLearner`. Both the paper's
`literal` and `pseudopragmatic` variants are available via `--mode`. See
`PAPER_STEP_MAPPING.md` for the step-by-step correspondence, and
`DIFFERENCES_FROM_PAPER.md` for every deviation and the explicit
"stop-before-training" boundary.

Sentiment/valence now uses NLTK **VADER** (`modified_vader_observation`),
utterances are split into per-phrase observations (`limited_punc_tokenization`),
reference vectors are normalized to sum 1, and the belief update uses the paper's
active `multiply()` factor product. The feedback corpus is **English-only**
(VADER is English-only). Requires `nltk` with the `vader_lexicon`, `punkt`,
`punkt_tab`, `wordnet`, `omw-1.4`, and `stopwords` data packages.

Two more entry points reproduce the paper's evaluation/model stages up to the
training boundary:

```powershell
# Online learning curve: multi-seed, 95% CI, literal vs pseudopragmatic, random baseline
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_learning_curve.py
# Route 2 neural inference network: assemble data + model, forward-check, STOP before training
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/train_route2_inference_network.py
```

The authored seed set currently contains 100 examples. The pipeline validates
IDs, feedback types, features, numeric values, target actions, and grounding
sources before learning. Probe reports include zero-weight, initial-weight,
learned-weight, and leave-one-probe-out results. Tied action scores are failures
rather than being resolved by JSON action order.

The action-probe benchmark has 14 probes across six categories
(`HumanComfortCoordination`, `RespectHumanIntent`, `RecipeCorrectness`,
`ServingReadiness`, `Safety`, `Efficiency`), including four `hard` tradeoff
probes where a tempting action shares the correct action's positive task
features but is net-negative due to coordination/recipe costs. The learned
reward solves all 14 (zero weights tie on every probe).

## Run

From the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_probe_states.py
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py
```

If the environment is already active, the scripts can also be run with plain `python`.

Choose the paper's learner variant with `--mode` (default `literal`):

```powershell
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py --mode pseudopragmatic
```

Paper-matched defaults: `--valence-scale 30 --precision-scale 2` (and
`--pragmatic-valence -30 --pragmatic-precision 2` for the pseudopragmatic mode).

## Subgoal re-ranking (H0 integration)

The learned reward is subgoal-compatible: it can re-rank H0's 8 subgoals via a
shared feature schema. To see the language feedback drive subgoal choice:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_subgoal_reranking.py --mode literal
```

`subgoal_featurizer.py` maps `(state, subgoal) -> phi`, and
`subgoal_reranker.choose_subgoal` picks the best of H0's task-valid subgoals by
`task_score + lambda_pref * comfort_score`. See `MIGRATION_NOTES.md` for the
full data flow.

## H0/subgoal trainable pipeline (no human data)

The reward learner is now trainable end to end without collecting a human
corpus. A hand-authored gold comfort weight vector `w*`
(`data/gold_comfort_weights.json`) acts as a rule teacher, and language is
synthesized (templates, optionally augmented by DeepSeek). Two things get
trained: (1) a Route 2 network mapping free-form language to a comfort reward,
and (2) a PPO policy shaped by that learned reward.

The reward-learning env (torch + nltk, e.g. `pantheonrl_env`):

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$B="baselines/baseline_b_linguistic_feedback/adapted_overcooked"
# 1. enumerate decision-point contexts (free, unlimited state supply)
python $B/scripts/enumerate_subgoal_contexts.py
# 2. synthesize the feedback corpus (templates; add DEEPSEEK_API_KEY for LLM augment)
python $B/scripts/generate_synthetic_feedback.py            # or --no-llm to force templates
# 3. train Route 2: language -> comfort reward (pure text -> reward by default)
python $B/scripts/train_route2.py --epochs 120
# 4. the metric that matters: held-out SUBGOAL accuracy + export frozen comfort weights
python $B/scripts/evaluate_route2_subgoal.py
```

Large synthetic-only runs use a frozen generator split, versioned cache, and
quality gates. Keep the API key in the process environment only:

```powershell
$env:DEEPSEEK_API_KEY="<set outside the repository>"
python $B/scripts/generate_synthetic_feedback.py `
  --llm-per-intent 10 --llm-max-intents 800 --target-size 10000 `
  --temperature 0.8 --top-p 0.95 --seed 17 `
  --llm-cache $B/outputs/synth/llm_cache_10k_v6.jsonl `
  --output $B/outputs/synth/synthetic_feedback.deepseek_10k.raw.json
python $B/scripts/generate_other_reference_feedback.py --per-topic 100 `
  --output $B/outputs/synth/deepseek_other_reference_v2.json
python $B/scripts/validate_feedback_corpus.py `
  --input $B/outputs/synth/synthetic_feedback.deepseek_10k.raw.json `
  --output $B/data/synthetic_feedback.deepseek_10k.validated.json `
  --report $B/outputs/synth/deepseek_10k.validation.json --enforce-quality
python $B/scripts/train_phrase_reference_classifier.py `
  --input $B/data/synthetic_feedback.deepseek_10k.validated.json `
  --augment $B/outputs/synth/deepseek_other_reference_v2.json
python $B/scripts/train_phrase_grounding.py `
  --input $B/data/synthetic_feedback.deepseek_10k.validated.json
python $B/scripts/train_route2_multiseed.py `
  --feedback $B/data/synthetic_feedback.deepseek_10k.validated.json
```

Generation labels always come from the rule teacher. DeepSeek supplies language
only; failed structure, sentiment, grounding, duplicate, and cross-split checks
are rejected before training.

Step 4 writes `outputs/route2/learned_comfort_weights.json`, the frozen comfort
weight vector handed to PPO. The PPO training env (`ray[rllib]` + tensorflow,
e.g. `durf310`):

```powershell
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6Comfort `
  --comfort-shaping-weights baselines/baseline_b_linguistic_feedback/adapted_overcooked/outputs/route2/learned_comfort_weights.json `
  --comfort-shaping-coeff 0.5 `
  --iterations 2 --num-workers 0 --ray-local-mode --overwrite
```

Omitting `--comfort-shaping-weights` reproduces the plain task PPO exactly, so
the comfort agent can be compared head to head with the task-only baseline.

Honesty note: the synthetic ground truth comes from the hand-authored rule
`w*`, so this validates *language generalization* (free text -> correct subgoal),
not the comfort rule itself. The rule is only sanity-checked against the small
real held-out set (the 5 real session examples + the hand-authored subgoal
probes), which are never used for training.

## Path A: live H0 + comfort subgoal agent (playable, no ray/PPO)

The learned reward now drives a playable agent directly, without PPO. At every
step H0 proposes its task-valid feasible subgoals, the learned comfort reward
re-ranks them (`plan_subgoal`), and H0 executes the chosen subgoal. Comfort thus
decides *which* subgoal is pursued; execution is unchanged (`execute_subgoal`
reproduces the frozen `rule_teacher_decision` motion exactly, enforced by
`durf/baseline/test_comfort_subgoal.py`).

Play as the human beside it (reward-learning env with NLTK; add torch only for
Route 2 live updates):

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
# Paper Route 1: classify -> type-specific grounding -> Bayesian update -> subgoal
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode route1-literal --route1-prior zero `
  --route1-lookback 25 --human-feedback-precision 4 `
  --max-consecutive-wait 3

# Paper PseudoPragmatic variant
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode route1-pseudopragmatic --route1-prior zero

# Route 2 neural online blend, or a non-adaptive frozen-weight control
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode frozen
```

Weights default to `outputs/route2/learned_comfort_weights.json`, falling back
to the gold teacher weights if Route 2 has not been exported yet. Route 1 uses
the paper's `N(0,25I)` prior by default; `--route1-prior frozen` warm-starts its
posterior mean from the exported weights. Every live update is written to
`feedback_updates.jsonl` with the speech act, five-class phrase reference type,
classification probability, effective precision, grounding source,
valence, posterior mean/variance changes, and the subgoal before/after update.
Trajectory references are grounded to features extracted from the latest
executed step window (default 25); if that window has no detectable event
features, the agent falls back to the selected-subgoal feature history.
Action references use the feasible subgoal/action library, while feature
references use text keywords. The learned posterior persists across episode resets, but the
trajectory window is cleared so a new episode cannot be credited to behavior
from the previous one.

To make the paper mechanism measurable rather than only visible in the UI, run
the grouped held-out type ablation:

```powershell
python $B/scripts/evaluate_route1_subgoal.py
```

It compares Route 1 Literal/PseudoPragmatic in `oracle` and fully `inferred`
modes, both with all feedback and with evaluative/imperative/descriptive
feedback separately. Gold annotations are retained only for scoring in inferred
mode; the learner cannot read them. The current 5-fold report
(`outputs/route1_subgoal_ablation_deepseek_strict.json`) makes the remaining
language bottleneck explicit: Literal reaches 97.8% with oracle interpretation
and 91.1% inferred (80.4% grounding coverage),
versus 95.6% for frozen Route 2. The former 57.3% number measured a separate
three-class speech act heuristic and was not comparable to the paper. The
paper-style five-class TF-IDF+LR reference classifier reaches 74.2% accuracy,
68.3% macro-F1, and 72.6% balanced accuracy on a fully scenario-disjoint
synthetic untouched test. This stricter number replaces the older 83.0%
intent-family split result; real human Overcooked reference labels remain the
required final benchmark.

The exact Route 1 posterior is atomically saved after accepted live feedback.
Use `--resume-learner-state` to restore it explicitly on a later run. Human
comments default to 4x source precision, then classification/grounding
confidence lowers that value when the reference is ambiguous.

Round-level comparison vs plain H0 rules (headless, judged by the gold `w*`
comfort partition, not the learned weights, so it is not circular):

```powershell
python durf/baseline/evaluate_comfort_subgoal.py --horizon 400
```

Headless rollouts drive both seats from H0 execution, which returns STAY when a
feature's single access tile is occupied by the partner. On the tight ring that
causes *static* mutual deadlock (a real person would just walk around). A small
symmetric `durf/baseline/coordination.StallBreaker` restores that walking-around
behavior for both seats, so the comparison reflects subgoal *choice* rather than
a low-level stand-off. (The pygame app already yields via
`cooperative_action_wrapper`; the StallBreaker is the headless analogue.)

### Simulated human partner (DeepSeek)

A mirror rule agent in the human seat deadlocks, so for a realistic comparison
`durf/baseline/play_deepseek_rounds.py` puts a DeepSeek "human" in the partner
seat. It reasons at the subgoal level -- picks one feasible subgoal per decision
and H0 executes it -- so it divides labor and yields like a person, at a few API
calls per episode. Requires `DEEPSEEK_API_KEY`.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"; $env:DEEPSEEK_API_KEY="sk-..."
python durf/baseline/play_deepseek_rounds.py --horizon 200 --episodes 1
```

Over 4 horizon-200 rounds the comfort agent was ~2x more considerate by the
gold judge (comfort/step +0.90 vs +0.40) while trading a little throughput
(soup 35 vs 40) at `lambda_pref=1.0`; single rounds are noisy but the
consistent signal is better complementary division of labor without deadlock.

Every step is logged to a JSONL training dataset (APPENDED across runs), so
repeated play accumulates a corpus of real DeepSeek language paired with state
context and subgoals -- the human data that was previously missing:

```powershell
python durf/baseline/play_deepseek_rounds.py --horizon 200 --episodes 4 --seed 100
#   -> data/deepseek_play/trajectories.jsonl   (use --no-dataset to disable)
```

Each record carries `recipe/pot_ingredients/pot_status`, both players'
`pos/held/subgoal/action`, the AI's active comfort features + gold comfort
score, and the human's utterance with `utterance_is_fresh` (true only at replan
steps) + `source`. The `(state, say, human.subgoal)` triples on fresh-utterance
rows are the language-training signal; `source == "deepseek"` marks genuine LLM
language (vs `rule fallback`).

## Tests

```powershell
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python -B -m unittest discover `
  -s baselines/baseline_b_linguistic_feedback/adapted_overcooked/tests `
  -v
```

## Convert an attributed session

First produce normalized trajectory, feedback, candidate-event, and attribution
JSONL files:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -B -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs/human_ai_sessions/<session_id>
```

Then convert attributed windows to provenance-rich Baseline B examples:

```powershell
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/convert_session_feedback.py `
  --session outputs/human_ai_sessions/<session_id>
```

The default output is
`baseline_b_feedback_examples.json` in the session directory. Ambiguous
attributions are excluded unless `--include-ambiguous` is passed. Converted
records retain the session, timestamp, step, layout, source, attribution
window, and confidence. They can be passed to the normal pipeline with
`--feedback`.

Current ring-layout session logs contain no human language feedback, so a ring
conversion can legitimately produce zero examples. This is a data-collection
gap, not a conversion failure.
