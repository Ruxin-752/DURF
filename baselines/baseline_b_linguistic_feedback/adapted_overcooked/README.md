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

The legacy scaffold below reproduces the paper model shape and deliberately
stops before training. The actively trained Route 2 pipeline is documented in
`ROUTE2_PAPER_ALIGNMENT.md`.

```powershell
# Online learning curve: multi-seed, 95% CI, literal vs pseudopragmatic, random baseline
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_learning_curve.py
# Legacy Route 2 shape check: forward-check, then stop before training
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
`subgoal_reranker.choose_subgoal` chooses the best H0-valid candidate by exact
`w · phi`. A non-unit comfort multiplier is rejected; H0 is used only as a
reward-tie fallback.

## H0/subgoal trainable pipeline (no human data)

Route 2 uses the paper's full-reward objective:

```text
(language, normalized trajectory features) -> complete 53-dim teacher reward w
```

The network remains `EmbeddingBag(30)` + trajectory features -> hidden 128 ->
reward vector. The paper's 15/9 dimensions become the Overcooked 53-feature
schema. The `paper_v5` corpus contains 83,592 reward-conditioned examples, 36
complete hidden rewards, and 12 stable synthetic authors whose identity is
independent of reward and feedback form. Of the 53 reward dimensions, 26 vary
across these configurations.

The frozen checkpoint is the paper-style ten-model ensemble at
`outputs/route2/paper_aligned_v5_seed137_selected/ensemble_manifest.json`. Fold `i` uses
validation fold `i`, test fold `(i+1)%10`, and the other eight folds for
training. Teacher IDs and complete reward configurations are both held out.
Candidate selection is per-fold and dev-only; its view physically omits test.
Checkpoint bytes and all manifests are hash-bound before the one-time test.
Deployment averages ten complete 53-dimensional predictions, while evaluation
scores each example only with its own held-out fold model.

The frozen seed-137 test contains 8,513 examples. Full-vector MSE is `0.012748`;
varying-dimension MSE is `0.025101`; cosine is `0.997620`; nearest reward-config
accuracy is `8512/8513 = 99.9883%`; preference-sensitive behavior majority is
`1194/1224 = 97.5490%`; and empty-hand cooking majority is `101/108 = 93.5185%`.
All three predeclared gates beat the train-mean and canonical constants.

This is only an identifiable synthetic full-reward result. Text-only remains
very strong (`0.037102` varying-dimension MSE, `99.9178%` nearest-config), while
trajectory-only fails (`8.108210`, `0%`). The separate handwritten local-human
probe is only `1/4 = 25%`; no human-language generalization claim is allowed.

A separate trajectory-required counterfactual benchmark closes the resulting
systems-test gap without reopening the formal seed-137 test. Version 2 was
regenerated after the live geometry/WAIT feature correction and crosses the
same feedback text with ten different trajectories/targets. On its frozen
60-row test, the three-seed full model reaches `0.105818` varying-dimension MSE
and `99.44%` nearest-target accuracy, versus `1.317562` and `10%` for text-only;
the full model improves MSE by `91.97%`. A 40-round synthetic online-adaptation
diagnostic improves preference satisfaction from `10.0%` to `77.5%`, lowers
regret from `23.6` to `4.6`, and lowers safety violations from `14` to `3`.
Both results are synthetic systems diagnostics, not human-preference evidence.
See `outputs/route2/trajectory_required_v2/` and the earlier simulation in
`outputs/route2/trajectory_required_v1/`.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$B="baselines/baseline_b_linguistic_feedback/adapted_overcooked"
python $B/scripts/generate_route2_teacher_corpus.py `
  --output $B/data/route2_teacher_feedback.paper_v5.synthetic.json
python $B/scripts/select_route2_candidate_on_dev.py `
  --split-seed 137 --output-dir <new_output_dir>
python $B/scripts/finalize_route2_selection.py --model-dir <new_output_dir>
python $B/scripts/evaluate_route2_paper_crossval.py --model-dir <new_output_dir>
```

The checked-in formal directory already has a completed one-time receipt and
must not be evaluated again. See [ROUTE2_AUDIT_REPORT.md](ROUTE2_AUDIT_REPORT.md)
and [ROUTE2_PAPER_ALIGNMENT.md](ROUTE2_PAPER_ALIGNMENT.md) for exact metrics and
limitations. The remaining substantive gap is held-out real human teacher data.

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
  --route1-lookback 25 --human-feedback-precision 4

# Paper PseudoPragmatic variant
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode route1-pseudopragmatic --route1-prior zero

# Route 2 ten-model neural ensemble, or a non-adaptive frozen-weight control
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode route2 `
  --teacher-id player_001
python -m durf.group_a.play_with_baseline --ai-mode comfort_subgoal `
  --comfort-feedback-mode frozen
```

Route 2 starts from a neutral zero reward, loads the paper-style ten-model
ensemble, applies the paper's precision-2 Gaussian belief update, records
ensemble disagreement, and saves the full model-bound posterior state to
`outputs/live/route2_state_v5_teacher_<hash>.json` (or a session-specific file
for anonymous play). Route 1 uses
the paper's `N(0,25I)` prior by default; `--route1-prior frozen` warm-starts it
from `--comfort-weights`. Every live update is written to
`feedback_updates.jsonl` with the speech act, five-class phrase reference type,
classification probability, effective precision, grounding source,
valence, posterior mean/variance changes, and the subgoal before/after update.
For Route 1, the phrase-level three-way `f_G` prediction shown in the UI is the
same object consumed by the learner: `Evaluative` selects trajectory credit,
`Imperative` selects a commanded action, and `Descriptive` selects named
features. Its confidence scales observation precision and an abstained phrase
rejects the atomic message update. The five-way reference prediction is only a
within-form subtype; a cross-form prediction is logged and projected back to
the three-way branch, never allowed to redirect credit assignment.
Trajectory references are grounded to features extracted from the latest
executed step window (default 25), while event boundaries are preserved:
`trajectory` uses the causal window, `action_behavioral` may refine that window
to a repeated event pattern, and an Imperative `action_spatial` subtype must
resolve to a feasible commanded subgoal. Future events are
rejected and the selected step/span is written to the trace. If that window has
no detectable event features, the agent falls back to the selected-subgoal feature history.
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
five-class TF-IDF+LR implementation reproduces the original paper's exact
982-row protocol at `87.162%` accuracy and `75.252%` macro-F1 (paper: about
`87%`/`75%`). That random-row protocol contains repeated text and task IDs, so
it is reported only as a protocol reproduction. The v8 classifier is selected
only on train/dev and reaches `87.02%` accuracy / `87.08%` macro-F1 on its
601-row combined dev split. Its fixed task_uuid- and normalized-text-disjoint
paper-v1 regression is `77.36%` / `72.52%`; this split is leakage-free but has
already been inspected and is no longer sealed. The 861-row legacy synthetic
regression is `92.68%` / `92.40%`. A manually labeled, teacher/session-disjoint
human Overcooked set remains the required final test. Full provenance is in
`REFERENCE_CLASSIFIER_REPORT.md`.

Phrase grounding now uses raw word `(1,2)` plus character `char_wb (3,5)`
TF-IDF with one-vs-rest LogisticRegression. Configuration and thresholds were
chosen on dev only. The exposed synthetic regression improves from
`72.65%/55.91%` to `75.59%/58.99%` micro/macro-F1. Per-feature support, F1,
thresholds, and prediction coverage are recorded in
`outputs/phrase_grounding/model.joblib.report.json`. Of all 53 reward
dimensions, 30 are proven to vary between live candidates, six are observable
but common to every candidate, and 17 are not supported by the current live
adapter. The latter 23 are explicitly decision-null and are masked/rejected in
Route 1 rather than producing a silent posterior update.

The exact Route 1 posterior is atomically saved after accepted live feedback.
Use `--resume-learner-state` to restore it explicitly on a later run. Human
comments default to 4x source precision, then three-way form, within-form
reference, grounding, and valence confidence lower that value when the
interpretation is ambiguous.

### Add human paper feedback-strategy labels (minimal two-field JSONL)

The three paper-facing strategy labels are exactly `Evaluative`, `Imperative`,
and `Descriptive`. For synthetic supervision they are fixed collapses of the
five reference classes: `trajectory -> Evaluative`, `feature -> Descriptive`,
and `action_spatial -> Imperative`. `action_behavioral` and `other` are excluded,
matching the original notebook's three-way analysis. Add one JSON object per line to
`data/human_feedback_form_annotations.jsonl`. The public schema accepts exactly
two fields and the label spelling/case is strict:

```json
{"language":"Please fetch an onion next.","classification_label":"Imperative"}
```

Copy `data/human_feedback_form_annotations.template.jsonl` for a blank-form
example or inspect `data/human_feedback_form_annotations.example.jsonl` for all
three labels. Then prepare template-family-disjoint partitions, derive
train-only variants, generate the paper-mapped hard contrasts, and train the
three-class TF-IDF + LogisticRegression model:

See `HUMAN_FEEDBACK_FORM_REVIEW.md` for the few current rows that require the
annotator's semantic confirmation; they were not silently relabelled.

```powershell
python -B scripts/prepare_human_feedback_form_annotations.py
python -B scripts/augment_human_feedback_form_train.py
python -B scripts/generate_feedback_form_hard_train.py
python -B scripts/train_feedback_form_classifier.py
```

The current input has 50 manually labelled rows. Automatic template-family
partitioning produces 44 train, 6 diagnostic dev, and 0 test rows. The 44 train
rows produce 85 traceable label-preserving variants; the raw text is not
overwritten. A separate balanced hard set contributes 144 train-only minimal
contrasts. The final training set has 3,131 rows and dev has 308 rows.

The frozen model reaches `95.13%` accuracy / `95.26%` macro-F1 on dev. On the
once-opened, normalized-text/group-disjoint synthetic test it reaches `90.91%`
accuracy / `90.85%` macro-F1 over 550 rows. The main v7 source is `93.27%`, but
the 45-row hard cross-template subset is only `64.44%`; therefore the aggregate
passes 87%, while robust hard-template generalization does not. Accuracy on
fresh, player-authored language remains unknown. Full counts and hashes are in
`outputs/feedback_form_classifier/model.report.json` and
`outputs/feedback_form_classifier/synthetic_test.report.json`.

For broader language coverage, 89 assistant-authored utterances were blindly
labelled/confirmed by the user and sealed without entering train/dev. The human
labels contain 30 Evaluative, 31 Imperative, and 28 Descriptive examples, all in
different surface families. The frozen model scores `64.04%` accuracy / `63.06%`
macro-F1; most errors are between Imperative and Descriptive. This is correctly
reported as human-confirmed labels on AI-candidate language, not as accuracy on
player-authored language. See `outputs/human_feedback_form_holdout.evaluation.json`.

To create a real held-out pilot, fill the blank labels in
`data/feedback_form_candidates.blind_labeling.template.jsonl` without opening
the suggested-label file, then save the reviewed two-field rows to
`data/human_feedback_form_holdout.jsonl`. Seal and evaluate them with:

```powershell
python -B scripts/prepare_human_feedback_form_holdout.py --text-origin ai_candidate_human_confirmed --confirm-human-labels
python -B scripts/evaluate_human_feedback_form_test.py --test data/human_feedback_form_holdout_test.json --sidecar outputs/human_feedback_form_holdout.sidecar.json
```

Use `--text-origin human_authored` instead when the utterances themselves were
written by a player after the model was frozen. The holdout gate requires at
least 10 independent families per label; otherwise it reports insufficient
data and does not reveal model metrics.

The deployed model is loaded automatically by `feedback_form_classifier.py`;
its public Route 1 API still returns lowercase `evaluative`, `imperative`, or
`descriptive`. If the model artifact is absent, live play uses the transparent
legacy rules. If a loaded model's maximum class probability is below `0.55`,
the learned prediction is retained and marked
`tfidf_logistic_regression_low_confidence` with `abstained=true`; the rule result
is logged only as an audit field and cannot overwrite the model. Evaluate the frozen model on the held-out human text
partition only after training and model selection are complete:

```powershell
python -B scripts/evaluate_human_feedback_form_test.py
python -B scripts/evaluate_synthetic_feedback_form_test.py
```

For synthetic rows, only `reference_type` and the fixed mapping above define the
target. The generator's cross-product `expected_feedback_type` is explicitly
ignored. Human targets come only from the explicit `classification_label`;
role strings and model predictions never become labels. Because the requested
public schema has no player/session IDs, the human split is a stable 80/10/10
hash of an automatically derived, label-independent template family. It
supports a template-family-disjoint claim, not a teacher- or session-held-out
claim. The current Evaluative annotations form only two automatically derived
families and both use the same basic `That move was ...` construction, so more
varied Evaluative phrasings are required before a human test can be released.

This three-class artifact owns Route 1's coarse credit-assignment branch. The
five-class phrase reference classifier below remains as a constrained subtype
and safety signal; it cannot override the three-class branch.

### Optional research-only five-class reference labels

This is **not** the two-field player JSON requested above. It is retained only
for a future fine-grained five-class credit-assignment benchmark, which needs
extra teacher/session provenance and independent adjudication. Ordinary player
annotations should use `human_feedback_form_annotations.jsonl` instead.

Copy the one-line schema from `data/human_reference_annotations.template.jsonl`
into `data/human_reference_annotations.jsonl`, then add one JSON object per
line.  The five labels and complete examples are in
`data/human_reference_annotations.example.jsonl`.  Keep the session's stable
`feedback_id`, `teacher_id`, and `session_id`, and provide either `scenario_id`
or `group_id`.  A first label uses `single_annotated`; an independent reviewer
may append the same `feedback_id` with an incremented `annotation_revision`,
status `adjudicated`, and their `adjudicator_id`.

Use `trajectory` for the complete executed sequence, `feature` for an
object/state/result property, `action_spatial` for one action at a particular
place or time, `action_behavioral` for a repeated habit, and `other` when the
utterance does not evaluate game behavior. Label from the utterance and its
recorded context, never from the current classifier prediction.

```powershell
python -B scripts/prepare_human_reference_annotations.py
```

The split is a fixed hash of teacher + session, so later appends never reshuffle
old groups. Single-person labels can enter train only; if their fixed split is
dev/test they remain sealed until adjudication. The generated gold benchmark
contains adjudicated dev/test rows only. Missing or empty input safely produces
empty outputs. Classifier predictions are never used as human gold labels.

Evaluate the frozen classifier only after adjudicated held-out rows exist:

```powershell
python -B scripts/evaluate_human_reference_gold.py
```

The evaluator fails on normalized-text or teacher/session exposure and reports
dev/test separately. With no human gold, accuracy remains `null`/unknown.

After train candidates exist, include only that generated train file in a new
classifier run:

```powershell
python -B scripts/train_phrase_reference_classifier.py `
  --augment data/human_reference_train_candidates.json
```

Never pass `human_reference_gold_benchmark.json` to `--augment`; it is reserved
for evaluation after the model has been frozen.

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

For a session whose online Route 2 update failed after logging (for example,
because its Pygame environment lacked the offline NLTK package), replay the
saved comments against their preceding trajectory windows:

```powershell
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/recover_route2_session_feedback.py `
  --session outputs/human_ai_sessions/<session_id>
```

This writes recovered update traces, personalized weights, a resumable Route 2
state, and an audit report inside the session. These are personalized inference
artifacts, not global supervised training labels: a comment does not reveal the
human's complete 53-dimensional reward vector.
