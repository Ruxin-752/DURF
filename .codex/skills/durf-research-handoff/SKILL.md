---
name: durf-research-handoff
description: Continue, audit, explain, or hand off the DURF Overcooked human-AI natural-language feedback attribution research project. Use when an AI needs to assess current progress, locate important files, modify the feedback/event/condition/Hu pipeline, run pilots, train or evaluate the hierarchical Hu model, prepare a meeting update, or decide the next engineering or experimental step without reconstructing the project history from scratch.
---

# DURF Research Handoff

Treat this skill as the project continuity contract for the DURF repository.

## Start Every Task

1. Locate the repository root containing `pyproject.toml`, `durf/`, `docs/`, and `src/`.
2. Run `scripts/project_snapshot.ps1` before making claims about branch, dirty files, recent sessions, or models.
3. Read the relevant references:
   - Research question or architecture: `references/experiment-overview.md`.
   - Current status and priorities: `references/current-progress.md`.
   - File ownership and data artifacts: `references/file-map.md`.
   - Commands, tests, and continuation procedure: `references/continuation-workflows.md`.
4. Read the live source files before editing. References explain intent but do not override newer code or Git state.
5. Preserve existing user changes. The handoff snapshot records a large uncommitted implementation batch; never reset or discard it.

## Non-Negotiable Research Contract

Maintain these boundaries unless the user explicitly changes the research design:

```text
program extracts state facts and candidate events
LLM resolves language semantics among evidence-backed candidates
human review creates pilot gold labels and approves schema changes
Hu learns condition-aware pairwise preferences
task backbone retains responsibility for completing the recipe
```

Do not:

- Reintroduce J/K scalar feedback as the primary method; it is archived legacy/TAMER baseline material.
- Describe the current best main agent as pure PPO. It is a task scorer/planner plus learned low-level subgoal executor.
- Describe Hu as PPO, DPO, an LLM, a reward model, or a neural policy. Current Hu is a hierarchical linear pairwise reranker.
- Mix Task labels with Coordination labels. `GET_TOMATO > YIELD` is invalid.
- Put environmental conditions into event names. Event = behavior; condition = state facts; preference = ranking label.
- Let LLM output silently mutate the event dictionary. Require review for schema additions or revisions.
- Count human-reviewed labels as automatic attribution accuracy. Preserve automatic output and review output separately.
- Use feedback-future trajectory frames during attribution.

## Current Architecture

Use the two-stage decision interface:

```text
state
-> Task candidates + task_score + Hu_task score
-> selected task subgoal
-> Recovery
-> Coordination candidates + prior + Hu_coord score
-> Safety
-> low-level action
```

Task answers what to do. Coordination answers how to execute the current task during human conflict. The Coordination vocabulary is exactly `CONTINUE_CURRENT_SUBGOAL` and `YIELD`; `HOLD_POSITION` was removed (same stay action as YIELD) and `REROUTE` is a task-layer replan decision that would need its own workflow.

## Work Procedure

1. Classify the request as architecture, code, data collection, attribution review, Hu training, evaluation, Git handoff, or reporting.
2. Inspect the mapped files and relevant session artifacts.
3. State whether findings are facts, inferences, hypotheses, or unknowns.
4. Make narrowly scoped edits using existing project patterns.
5. Run focused tests and compilation checks from `references/continuation-workflows.md`.
6. For data changes, verify the full provenance chain rather than only final output counts.
7. Update `references/current-progress.md` after a meaningful milestone, especially after commits, real pilot collection, model training, schema freeze, or experiment completion.
8. Report what changed, what was verified, what remains unproven, and whether changes are committed.

## Current Highest-Priority Goal

Do not add broad new architecture first. Complete the empirical minimum loop:

```text
commit current implementation checkpoint
-> collect real Task and Coordination language feedback
-> review time/event/condition/pair labels
-> train both Hu heads
-> compare no-Hu, shadow, and apply on probe states
-> verify preference gains without material task degradation
```

The immediate data bottleneck is valid Coordination feedback. Collect both directions under different conditions: `YIELD > CONTINUE_CURRENT_SUBGOAL` and `CONTINUE_CURRENT_SUBGOAL > YIELD`.

## Completion Standard

Do not call the experiment delivered until all are true:

- Current implementation is committed and reproducible from a fresh clone.
- Automatic attribution is evaluated against separate human gold labels.
- Task and Coordination Hu heads both have real reviewed data.
- Probe evaluation demonstrates condition-sensitive ranking changes.
- Task completion is non-inferior to the chosen baseline under a predeclared margin.
- The localized Linguistic Feedback baseline and ours share a fair task backbone and input evidence boundary.
- Evaluation scripts, frozen schemas, model metadata, sample sessions, and run instructions are included.

