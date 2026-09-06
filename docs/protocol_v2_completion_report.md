# DURF Protocol v2 Simulation Completion Report

**Date**: 2026-08-11
**Model**: Hierarchical Hu (3-level pairwise linear ranking)
**Status**: Simulation pipeline complete; human-participant validation pending

> **Addendum (2026-09-03)**: the `user_bias` finding in this report ("bias-only accuracy
> stays at exactly Hu_general's baseline") was measured on the **coordination** domain and
> still holds unchanged. A follow-up check found the **task** domain tells a different
> story: across all four personas, `user_bias_task` stays within +/-0.06 (essentially
> noise) at every data level, while `user_bias_coord` learns a real, persona-varying signal
> (+/-0.05 to +/-1.74). `PerUserAdapter` now defaults to `enable_task_bias=False`: the task
> head drops its constant per-user offset entirely, while the coordination head keeps its
> full three-level decomposition unchanged. This is a reversible flag, not an architecture
> deletion -- see `durf/hu/subgoal_reranker.py` and `docs/hierarchical_hu_runtime.md` §10.

---

## 1. System Architecture

```
Hu_user(condition, subgoal)
  = Hu_general(condition, subgoal)      // frozen shared prior (3806 invariant sim samples)
  + user_bias(subgoal)                   // unconditional per-user offset
  + condition_delta(condition, subgoal)  // condition-sensitive per-user weights
```

- **Hu_general** is frozen after training on persona-invariant simulated feedback. Never updated during per-user adaptation.
- **user_bias** is a fixed subgoal bias vector per user. Learned but cannot express condition-dependent preferences.
- **condition_delta** is a sparse weight matrix (28 conditions x 11 task subgoals + 14 conditions x 2 coordination subgoals). Only active conditions observed in a feedback sample contribute to gradient updates.

---

## 2. Phase Status

| Phase | Task | Status | Key Result |
|---|---|---|---|
| P0 | Schema freeze, condition manifest, directory structure | Complete | 28 frozen conditions, 8 provenance fields |
| P1 | Cross-persona consistency audit | Complete | 5 conflicting groups (29 samples), all in coordination |
| P2 | Train and freeze Hu_general | Complete | Held-out test: task 100%, coord 99.0% |
| P3 | Implement PerUserAdapter (3-level architecture) | Complete | general + user_bias + condition_delta |
| P4 | Sim data closed-loop pipeline | Complete | 4 personas fully processed |
| P5 | 4-persona PerUserAdapter training + evaluation | Complete | Selfish coord acc: 0.713 -> 1.000 |
| P6 | Multi-persona comparison analysis | Complete (script) | Report saved to `outputs/hu_evaluation/` |
| P7 | Probe-based evaluation | Partial | Results available; unified comparison report not yet finalized |
| P8 | Learning curve analysis (selfish persona) | Complete (sim) | Real human data efficiency not yet verified |

---

## 3. Data Pipeline

### 3.1 Simulation Sessions

| Metric | Value |
|---|---|
| Total sim sessions | 127 |
| Per-persona breakdown | cooperative (37), selfish (42), polite (24), lenient (24) |
| Pilot sessions (real human) | 21 |
| Total pairwise samples extracted | 4,543 |
| Total probe hits | 67,014 |

### 3.2 Cross-Persona Audit (P1, corrected)

The audit groups pairwise preference samples by `(decision_level, active_condition_signature, frozenset(subgoal_pair))`. The pair grouping is order-invariant so that `A > B` and `B > A` are compared in the same group.

| Metric | Value |
|---|---|
| Unique (condition, pair) groups | 319 |
| Invariant groups (all personas agree) | 148 (3,806 samples) |
| Conflicting groups (personas disagree) | **5** (29 samples) |
| Single-persona groups (insufficient for comparison) | 166 (680 samples) |

All 5 conflict groups are in the coordination layer (CONTINUE_CURRENT_SUBGOAL vs YIELD) and typically involve conditions like `ai_on_human_path=True`, `narrow_corridor=True`, `human_trying_to_pass=True`. These are the exact scenarios where persona differences are expected — cooperative and lenient personas tend to prefer CONTINUE, while polite and selfish personas sometimes prefer YIELD.

**Important caveat**: This audit groups by the `condition_signature` (active non-zero conditions only). Samples with different raw condition values but the same active-condition frozenset are compared. This may overestimate agreement for conditions with continuous or multi-level values. The "5 conflicting groups" should be interpreted as a lower bound; finer-grained grouping might reveal additional disagreements.

### 3.3 Hu_general Training Set (from invariant samples)

| Dataset | Task Samples | Coordination Samples | Total |
|---|---|---|---|
| Train (70%) | 2,200 | 464 | 2,664 |
| Validation (15%) | 471 | 99 | 570 |
| Test (15%) | 472 | 100 | 572 |

Split is per-domain (stratified by decision_level) to protect the rarer coordination samples.

---

## 4. Hu_general (Frozen Prior v2.1)

### 4.1 Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Learning rate | 0.05 |
| L2 regularization | 1e-4 |
| Seed | 42 |

### 4.2 Performance

| Split | Task Accuracy | Coord Accuracy | Task Margin | Coord Margin |
|---|---|---|---|---|
| Train (2,195) | 1.000 | 1.000 | 10.82 | 7.69 |
| Validation (469) | 1.000 | 0.988 | 10.79 | 7.53 |
| **Test (305)** | **1.000** | **0.990** | **10.79** | **8.19** |

These are **hold-out test metrics on invariant sim data** — not generalization to real human preferences.

### 4.3 Freeze Record

| Field | Value |
|---|---|
| File | `outputs/hu_general/hierarchical_hu.json` |
| Version | v2.1 |
| SHA256 | `19b0a1b79d96cb1ef4b5858cf1373d0aa0e1b0278a0ea8b46cd336df7870576c` |
| Status | Frozen, read-only |
| Audit notes | 5 conflicting groups excluded from invariant set |

### 4.4 Top Coordination Weights

| Condition | Subgoal | Weight | Direction |
|---|---|---|---|
| `ai_adjacent_to_current_subgoal_target` | CONTINUE_CURRENT_SUBGOAL | +3.512 | Promotes |
| `ai_adjacent_to_current_subgoal_target` | YIELD | -3.510 | Discourages |
| `human_trying_to_pass` | CONTINUE_CURRENT_SUBGOAL | +1.866 | Promotes |
| `human_trying_to_pass` | YIELD | -1.860 | Discourages |
| `ai_on_human_path` | YIELD | -1.858 | Discourages |
| `ai_on_human_path` | CONTINUE_CURRENT_SUBGOAL | +1.851 | Promotes |
| `human_has_onion` | YIELD | -0.549 | Discourages |
| `human_has_onion` | CONTINUE_CURRENT_SUBGOAL | +0.544 | Promotes |

**Interpretation**: Hu_general encodes a general default-continue coordination strategy. When the AI is adjacent to its subgoal target, the human is trying to pass, or the AI is on the human's path, the model promotes CONTINUE and discourages YIELD. This likely reflects the fact that the training set was dominated by cooperative and lenient persona labels, which tend to prioritize task progress over explicit yielding. The weaker signals — `pot_cooking_or_ready` promoting YIELD (+0.217) and `narrow_corridor` promoting YIELD (+0.483) — suggest that the model does learn to yield under very specific conditions, but with much lower weight.

**This differs from the v2.0 report** which incorrectly described these weights as "AI should yield." The weights consistently promote CONTINUE in coordination scenarios. See Section 9.1 for an analysis of this discrepancy.

### 4.5 Top Task Weights

| Condition | Subgoal | Weight | Direction |
|---|---|---|---|
| `pot_empty` | GET_DISH | -0.253 | Discourages |
| `human_has_tomato` | PICKUP_SOUP | -0.250 | Discourages |
| `human_has_onion` | GET_DISH | -0.250 | Discourages |
| `ai_empty_handed` | PICKUP_SOUP | +0.249 | Promotes |
| `pot_empty` | PICKUP_SOUP | -0.249 | Discourages |
| `pot_partially_filled` | GET_DISH | -0.249 | Discourages |
| `human_trying_to_pass` | PICKUP_SOUP | -0.248 | Discourages |
| `human_has_onion` | PICKUP_SOUP | -0.248 | Discourages |

All task weights are within a narrow range (~±0.25), suggesting the model learns a consistent "what NOT to do" signal — discourage redundant or unproductive subgoals — rather than strongly promoting any specific task action. The dominant task subgoals affected are PICKUP_SOUP and GET_DISH, consistent with the Overcooked domain where these are the most frequent and consequential task decisions.

---

## 5. Per-User Adapter Results (P5, v2.1)

Each persona: 12 train sessions (seeds 0-11), 4 test sessions (seeds 20-23). Training uses Hu_general v2.1 as the frozen base.

### 5.1 Results Table

| Persona | Train Samples | Hu_general Coord Acc | Hu_user Coord Acc | Coord Delta |
|---|---|---|---|---|
| selfish | 666 | **0.713** | **1.000** | **+0.287** |
| cooperative | 628 | 1.000 | 1.000 | +0.000 |
| polite | 628 | 1.000 | 1.000 | +0.000 |
| lenient | 630 | 0.951 | 0.951 | +0.000 |

Task accuracy is 1.000 for all personas, both Hu_general and Hu_user. Task-level preferences do not meaningfully vary by persona in the Overcooked domain.

### 5.2 Probe Ranking Sensitivity

| Persona | Task Hits | Task Ranking Changes | Coord Hits | Coord Ranking Changes |
|---|---|---|---|---|
| selfish | 3,037 | 126 (4.2%) | 112 | 43 (38.4%) |
| cooperative | 3,059 | 82 (2.7%) | 14 | 0 (0%) |
| polite | 3,059 | 82 (2.7%) | 14 | 12 (85.7%) |
| lenient | 3,059 | 82 (2.7%) | 14 | 0 (0%) |

Probe ranking changes indicate that `condition_delta` alters the model's internal preference ordering even when test accuracy does not change. This is most pronounced for polite (85.7% of coordination probes changed rank despite 0% accuracy delta), suggesting `condition_delta` learns to amplify margins in a direction that happens to align with existing test labels.

### 5.3 Personality Characterization (Simulated Personas)

| Persona | Learned Coordination Pattern |
|---|---|
| **selfish** | Strongest YIELD preference. Delta promotes YIELD in all coordination contexts. This is the only persona where Hu_general genuinely underperforms (71.3%). |
| **cooperative** | Aligns closely with Hu_general's default-continue strategy. Delta adds minimal refinement. |
| **polite** | Yields in narrow corridors but continues when holding items. Delta amplifies margins significantly (7.85 -> 15.34). |
| **lenient** | Most tolerant. Hu_general and Hu_user have identical accuracy. Delta improves margin only. |

**Important caveat**: These "personality" characterizations describe how the condition_delta weights recover the pre-designed sim persona behavior patterns. They do not constitute evidence about real human personality differences. The persona labels (selfish, cooperative, etc.) are simulation design parameters, not validated psychological constructs.

---

## 6. Cross-Persona Condition Divergence (P6)

### 6.1 Most Divergent Coordination Conditions

Top conditions where persona-specific `condition_delta` weights diverge most:

| Condition | Subgoal | Selfish | Cooperative | Polite | Lenient | Max Spread |
|---|---|---|---|---|---|---|
| `ai_adjacent_to_subgoal_target` | CONTINUE | -1.061 | +0.215 | -0.699 | +0.135 | 1.276 |
| `ai_adjacent_to_subgoal_target` | YIELD | +1.061 | -0.215 | +0.699 | -0.135 | 1.276 |
| `pot_cooking_or_ready` | CONTINUE | +0.607 | -0.046 | +0.699 | -0.440 | 1.139 |
| `pot_cooking_or_ready` | YIELD | -0.607 | +0.046 | -0.699 | +0.440 | 1.139 |
| `ai_has_dish` | YIELD | -0.755 | -0.031 | -0.486 | +0.234 | 0.989 |
| `human_has_dish` | CONTINUE | +0.715 | +0.034 | +0.362 | -0.189 | 0.904 |
| `narrow_corridor` | CONTINUE | -0.859 | -0.107 | -0.486 | +0.020 | 0.879 |

Selfish is consistently the largest outlier across most coordination conditions, with cooperative/lenient clustering toward the center. This validates the PerUserAdapter design: `condition_delta` can recover the pre-designed persona preference differences.

**Caveat**: The condition features used for grouping include only conditions that are explicitly True/False (+1/-1). Features with `None` or 0 values are excluded from the signature. This means the condition signature is an approximation of the full game state.

---

## 7. Learning Curve Analysis (P8, Selfish Persona Only)

### 7.1 Configuration
- Persona: selfish (most divergent from Hu_general)
- Data fractions: 5%, 10%, 20%, 35%, 50%, 75%, 100%
- Trials per fraction: 5 (random subsampling)
- Ablation: full adapter (general + bias + delta) vs bias-only (general + bias)

### 7.2 Coordination Accuracy vs Sample Size

| Fraction | N Samples | Full Adapter Acc | Bias-only Acc | Gain over Hu_general |
|---|---|---|---|---|
| 5% | 33 | 0.963 +/- 0.005 | 0.713 +/- 0.000 | +0.251 |
| 10% | 66 | 0.979 +/- 0.017 | 0.713 +/- 0.000 | +0.267 |
| 20% | 133 | 0.993 +/- 0.014 | 0.713 +/- 0.000 | +0.280 |
| 50% | 333 | 1.000 +/- 0.000 | 0.713 +/- 0.000 | +0.287 |
| 100% | 666 | 1.000 +/- 0.000 | 0.713 +/- 0.000 | +0.287 |

Hu_general baseline coordination accuracy: **0.713**

### 7.3 Task Accuracy

Task accuracy is 1.000 for all fractions and both variants. Hu_general already saturates task preferences.

### 7.4 Interpretation (With Limitations)

1. **`user_bias` alone is insufficient**: At every data level, bias-only accuracy remains at exactly Hu_general's baseline (0.713). No fixed per-subgoal offset can capture persona-specific coordination preferences because those preferences are fundamentally condition-sensitive.

2. **`condition_delta` shows rapid improvement**: 33 pairwise samples (5% of training data) achieve 96.3% coordination accuracy — a +25.1 percentage point improvement. At 133 samples (20%), accuracy reaches 99.3%.

3. **Sample efficiency in context**: These results are obtained on **simulated persona data** where training and test samples are drawn from the same simulation pipeline with different random seeds. The efficiency should be interpreted as: within the current sim persona and task distribution, the model architecture can adapt quickly. Whether real human participants produce similarly learnable preference patterns at comparable sample sizes is an open question that requires separate empirical validation.

---

## 8. Limitations and Caveats

### 8.1 Sim-to-Real Gap

All results in this report are obtained from simulated personas with pre-designed preference patterns. The following are **not yet established**:

- Whether real human preferences are as learnable as simulated ones
- Whether natural language feedback produces pairwise labels of comparable quality to direct sim behavior labels
- Whether the condition schema (28 task conditions, 14 coordination conditions) captures the features that real humans use when deciding preferences

### 8.2 Audit Granularity

The cross-persona audit groups samples by `condition_signature` — only active non-zero conditions. Samples with the same active-condition frozenset but different raw condition values are considered equivalent, which may overestimate agreement. The "5 conflicting groups" should be treated as a lower bound.

### 8.3 Persona Labels

The persona characterizations (selfish, cooperative, polite, lenient) are simulation design parameters, not validated psychological constructs. The results show that the PerUserAdapter can recover these designed patterns, but this should not be interpreted as evidence that the system can classify or identify real human personality traits.

### 8.4 Probe Sensitivity

Probe ranking changes indicate that `condition_delta` alters internal preference ordering, but the biological validity of these probe states as evaluation benchmarks has not been externally validated. Probe results should be interpreted as an internal consistency check, not as an external measure of model quality.

### 8.5 Learning Curve Generalizability

The learning curve analysis was performed on only the selfish persona. Results may differ for other personas or for real human participants.

---

## 9. Issues Identified and Corrected After v2.0

### 9.1 Conflict Audit Bug (Corrected)

**v2.0**: Reported "0 conflicting groups in 326 groups."
**v2.1**: After fixing the grouping key to be order-invariant (using `frozenset` of the subgoal pair instead of ordered `(preferred, rejected)`), the audit correctly detects 5 conflicting groups (29 samples), all in the coordination layer. The original "0 conflicts" was an artifact of the grouping logic, not evidence of actual agreement.

### 9.2 Missing Held-out Test Set (Corrected)

**v2.0**: Hu_general was evaluated only on train/validation splits (no held-out test data).
**v2.1**: A proper 70/15/15 train/val/test split is now used. Held-out test metrics: task 1.000, coordination 0.990. All reported metrics now distinguish train, validation, and test performance.

### 9.3 Coordination Weight Interpretation (Corrected)

**v2.0 report**: Described `human_trying_to_pass` and `ai_on_human_path` conditions as "AI should yield."
**v2.1**: The actual weights are:
- `human_trying_to_pass` -> CONTINUE_CURRENT_SUBGOAL: **+1.866** (promotes)
- `human_trying_to_pass` -> YIELD: **-1.860** (discourages)
- `ai_on_human_path` -> CONTINUE_CURRENT_SUBGOAL: **+1.851** (promotes)

Hu_general encodes a default-continue strategy, not a yield-when-blocking strategy. The report text now reflects the actual weight directions.

### 9.4 Task Weight Values (Corrected)

**v2.0 report**: Reported `pot_cooking_or_ready -> PICKUP_SOUP = +0.504`.
**v2.1**: Actual weight from frozen model (SHA256 `19b0a1b7`): `pot_cooking_or_ready -> PICKUP_SOUP = +0.245`. The v2.0 report appears to have read weights from a different model version. All weight tables in this report are generated directly from the frozen v2.1 model.

### 9.5 Missing P6 Report File (Corrected)

**v2.0**: `report_p6_comparison.py` existed but only printed to stdout; no saved report artifact.
**v2.1**: Multi-persona comparison results are now saved to `outputs/hu_evaluation/comparison_report.json`.

### 9.6 Overstated Status (Corrected)

**v2.0**: "Status: All phases complete."
**v2.1**: "Status: Simulation pipeline complete; human-participant validation pending." P4 (human data closed-loop) and P7 (unified probe comparison report) are explicitly noted as pending or partial.

---

## 10. Artifact Inventory

### 10.1 Key Files

```
outputs/hu_general/
  hierarchical_hu.json                            -- Frozen Hu_general model (SHA256: 19b0a1b79d96cb...)
  frozen_hash.txt                                 -- SHA256 record
  metadata.json                                   -- Full training config and metrics (v2.1)

outputs/hu_general/filtered_training/
  hu_general_invariant.jsonl                      -- 3,806 invariant pairwise samples (audit v2.1)
  hu_general_train.jsonl                          -- 2,664 training samples (70%)
  hu_general_val.jsonl                            -- 570 validation samples (15%)
  hu_general_test.jsonl                           -- 572 test samples (15%)
  sim_general_audit.json                          -- Full audit results (5 conflicts)
  sim_persona_conflicts.jsonl                     -- 29 conflicting samples from 5 groups

outputs/hu_users/{selfish,cooperative,polite,lenient}/
  hu_user.json                                    -- Trained PerUserAdapter (v2.1 Hu_general base)
  metadata.json                                   -- Training/eval/probe metrics

outputs/hu_evaluation/
  comparison_report.json                          -- Multi-persona comparison data (P6)
  comparison_report.txt                           -- Full comparison report output

docs/
  condition_manifest_v2.md                        -- 28 frozen condition feature definitions
  research_protocol_freeze_v2.md                  -- Governing experimental protocol
  protocol_v2_completion_report.md                -- This report
```

### 10.2 Code Files

| File | Purpose | Changes in v2.1 |
|---|---|---|
| `durf/feedback_attribution/schemas.py` | Added provenance fields | — |
| `durf/feedback_attribution/hu_dataset_builder.py` | User isolation check | — |
| `durf/hu/subgoal_reranker.py` | PerUserAdapter class | Fixed per-level evaluate |
| `durf/hu/audit_sim_personas.py` | Cross-persona audit | **Fixed**: order-invariant pair grouping |
| `durf/hu/build_general_prior_dataset.py` | Hu_general training set filter | Backward compat for old sessions |
| `durf/hu/train_subgoal_reranker.py` | Training script | Used --test-dataset for held-out test |
| `durf/hu/run_p5_closed_loop.py` | Single-persona pipeline | Regenerated with v2.1 Hu_general |
| `durf/hu/report_p6_comparison.py` | Multi-persona comparison | — |
| `durf/hu/run_p8_learning_curve.py` | Data efficiency analysis | — |
| `scripts/split_hu_general_data.py` | 70/15/15 split | **New** |

---

## 11. Next Steps

### 11.1 Before Real Human Experiments

1. **Probe evaluation**: Finalize a unified probe comparison report across all cuatro personas (P7) — currently each persona has separate probe results but no cross-persona analysis of which probes are most discriminative.

2. **Condition granularity**: Investigate whether the `condition_signature` grouping (active conditions only) is appropriate for the audit, or whether finer-grained (e.g., including 0-value conditions as "known absent") grouping would reveal additional conflicts.

3. **Coordination weight interpretation**: Investigate why Hu_general promotes CONTINUE in `human_trying_to_pass` and `ai_on_human_path` scenarios. Possible explanations: (a) the invariant training set is dominated by cooperative persona labels that prefer CONTINUE; (b) the sim label generation logic may bias toward CONTINUE; (c) the pairwise label construction from feedback events may systematically favor CONTINUE over YIELD.

### 11.2 Real Human Participant Pipeline

1. Recruit participants to play 3-5 Overcooked sessions with the AI agent.
2. Collect natural language feedback during gameplay (current UI supports this).
3. Run the attribution pipeline to convert feedback into pairwise preferences.
4. Train `delta_user` on top of frozen Hu_general (v2.1).
5. Evaluate on held-out participant sessions and probe states.
6. Compare human delta weights against simulated persona delta weights.
7. Assess whether 30-50 pairwise preference samples per participant yield meaningful personalization, and whether the condition schema captures relevant features for real human preferences.

---

## 12. Core Conclusions

Within the scope of the current Overcooked task and simulated persona setup, the following have been demonstrated:

1. **The protocol and engineering pipeline are functional**: Raw sim sessions can be converted through feedback attribution into pairwise preference labels, audited for cross-persona consistency, and used to train a hierarchical preference model.

2. **A frozen shared prior (Hu_general) can serve as a baseline**: The model achieves 100% task accuracy and 99.0% coordination accuracy on held-out invariant sim data, providing a stable reference point for per-user adaptation.

3. **The three-level PerUserAdapter (general + bias + delta) can recover pre-designed persona differences**: The selfish persona's coordination accuracy improves from 71.3% to 100% through `condition_delta` alone. `user_bias` contributes nothing — all persona-specific signal is condition-sensitive.

4. **The architecture shows promising sample efficiency on simulated data**: 33 pairwise samples (equivalent to ~3 Overcooked sessions with feedback) achieve 96.3% coordination accuracy for the selfish persona.

These results demonstrate the feasibility of the protocol and architecture design. They do **not** yet constitute evidence that the system can learn real human preferences from natural language feedback, nor that similar sample efficiency will hold for real participants. These questions require human-participant validation.
