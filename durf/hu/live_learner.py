"""Learn from a participant's feedback DURING the session.

Until 2026-09-05 "Ours" loaded its preference model once, before the first
step, and never touched it again; the labels came out of an offline pass after
the session.  The comparison arm updates its weights live, so a participant of
ours could give twenty pieces of feedback and see nothing change -- which
breaks the within-session experience the experiment is about, and makes any
"did the behaviour change after this feedback" measure zero by construction
(work_plan_v1.md item 5; open_issues_v2.md #1).

This module closes the loop inside the session:

    step record ──► observe_step()           (rolling in-memory trajectory)
    feedback    ──► on_feedback()
                      ├─ detect candidate events on the recent window
                      ├─ attribute (LLM, keyword baseline as labelled stand-in)
                      ├─ provenance ► pairwise labels ► protocol-v2 schema gate
                      ├─ retrain the per-user adapter IN PLACE on all labels so far
                      ├─ archive a checkpoint at F0 / F5 / F10 / F15 accepted feedbacks
                      └─ append provenance / labels / audit / latency to the session dir

It reuses the offline pipeline's functions on in-memory records, so a live
session and an offline re-run of the same logs produce the same labels.  The
files it writes are the same files the offline tools read, so nothing
downstream changes.

Two honesty points that are measured, not assumed:

* Latency.  An LLM round-trip sits between the feedback and the update.  Every
  call records ``update_latency_ms`` (this is M1-3's producer, which did not
  exist), and the runtime decides whether to block on it or apply it when it
  arrives (sim: block; pygame: worker thread).
* Failure.  If the LLM call fails, the keyword baseline stands in, LABELLED
  ``rule_fallback_after_llm_error``; those labels still update the model (the
  participant should not be punished for an outage) but are counted separately
  and excluded from any "LLM converted this feedback" statistic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from durf.feedback_attribution.event_detectors import detect_candidate_events
from durf.feedback_attribution.hu_dataset_builder import (
    build_provenance_record,
    build_training_samples,
)
from durf.feedback_attribution.label_validator import (
    filter_training_samples,
    rejection_summary,
)
from durf.feedback_attribution.llm_attributor import run_llm_attribution
from durf.feedback_attribution.sample_builder import build_preview_attribution
from durf.feedback_attribution.schemas import (
    ATTRIBUTOR_RULE_BASELINE,
    ATTRIBUTOR_RULE_FALLBACK,
)
from durf.hu.subgoal_reranker import (
    DEFAULT_ENABLE_TASK_BIAS,
    HierarchicalHu,
    PairwiseSample,
    PerUserAdapter,
    pairwise_sample_from_record,
)

DEFAULT_CHECKPOINTS = (0, 5, 10, 15)
DEFAULT_LOOKBACK_STEPS = 30


@dataclass
class FeedbackUpdate:
    """What one piece of feedback did to the model."""

    feedback_event_id: str | None
    attributor: str
    labels_added: int
    labels_rejected: int
    rejection_reasons: dict[str, int]
    attribution_flags: list[str]
    update_latency_ms: float
    accepted_feedback_count: int
    checkpoint_written: str | None
    error: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {"record_type": "live_feedback_update", **self.__dict__}


@dataclass
class LiveLearner:
    user_id: str
    session_dir: Path
    hu_general: HierarchicalHu | None = None
    layout: str | None = None
    use_llm: bool = True
    lookback_steps: int = DEFAULT_LOOKBACK_STEPS
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    enable_task_bias: bool = DEFAULT_ENABLE_TASK_BIAS
    train_epochs: int = 200
    seed: int = 0
    # Injection points so tests can run without a network and the pygame UI
    # can hand in its own threaded caller.
    attribute_fn: Callable[..., tuple[dict, dict]] | None = None
    clock: Callable[[], float] = time.perf_counter

    trajectory: list[dict] = field(default_factory=list, repr=False)
    samples: list[PairwiseSample] = field(default_factory=list, repr=False)
    accepted_feedback_count: int = 0
    updates: list[FeedbackUpdate] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        prior = self.hu_general or HierarchicalHu(task_head=None, coordination_head=None)
        self.adapter = PerUserAdapter(
            hu_general=prior, user_id=self.user_id, enable_task_bias=self.enable_task_bias
        )
        self.session_dir = Path(self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = self.session_dir / "checkpoints"
        self._write_checkpoint_if_due()

    # ------------------------------------------------------------ model view
    @property
    def model(self) -> PerUserAdapter:
        """The object the runtime scores with.  Identity never changes."""
        return self.adapter

    # ------------------------------------------------------------ observing
    def observe_step(self, step_record: dict) -> None:
        """Feed one trajectory_step-shaped record (same shape as trajectory.jsonl)."""
        self.trajectory.append(step_record)

    # ------------------------------------------------------------ learning
    def on_feedback(self, feedback: dict) -> FeedbackUpdate:
        """Attribute one feedback event and fold its labels into the model.

        Synchronous convenience: ``apply(attribute(feedback))``.  The pygame
        runtime splits the two -- ``attribute`` (slow: LLM round-trip) runs in
        a worker thread and never touches the model; ``apply`` (fast: a small
        retrain) runs on the game loop's thread, so the loop never scores with
        a half-written weight matrix.
        """
        return self.apply(self.attribute(feedback))

    def attribute(self, feedback: dict) -> dict[str, Any]:
        """Slow half: events + attribution.  Pure w.r.t. the model."""
        started = self.clock()
        window = self.trajectory[-(self.lookback_steps * 3):] if self.trajectory else []
        candidate_events = detect_candidate_events(window) if window else []

        baseline = build_preview_attribution(
            feedback=feedback,
            trajectory=window,
            candidate_events=candidate_events,
            lookback_steps=self.lookback_steps,
        )
        attribution = baseline
        audit: dict[str, Any] | None = None
        error: str | None = None
        if self.use_llm:
            attribute = self.attribute_fn or run_llm_attribution
            try:
                attribution, audit = attribute(
                    feedback=feedback,
                    trajectory=window,
                    candidate_events=baseline.get("candidate_events") or [],
                    baseline_attribution=baseline,
                    lookback_steps=self.lookback_steps,
                )
            except Exception as exc:  # noqa: BLE001 -- any failure is a stand-in, never a crash
                error = str(exc)
                attribution = dict(baseline)
                attribution["attributor"] = ATTRIBUTOR_RULE_FALLBACK
                attribution["attribution_error"] = error
                audit = {
                    "feedback_event_id": baseline.get("feedback_event_id"),
                    "status": "fallback_to_baseline",
                    "attributor": ATTRIBUTOR_RULE_FALLBACK,
                    "error": error,
                }
        else:
            attribution.setdefault("attributor", ATTRIBUTOR_RULE_BASELINE)
        return {
            "feedback": feedback,
            "window": window,
            "attribution": attribution,
            "audit": audit,
            "error": error,
            "started": started,
        }

    def apply(self, pending: dict[str, Any]) -> FeedbackUpdate:
        """Fast half: provenance, schema gate, retrain in place, checkpoint, log."""
        feedback = pending["feedback"]
        attribution = pending["attribution"]
        provenance = build_provenance_record(
            attribution=attribution,
            feedback=feedback,
            trajectory=pending["window"],
            user_id=self.user_id,
            review_decision=None,
        )
        raw_samples = build_training_samples([provenance])
        kept, rejected = filter_training_samples(raw_samples)
        new_samples = [
            sample
            for sample in (pairwise_sample_from_record(record) for record in kept)
            if sample is not None
        ]
        self.samples.extend(new_samples)
        if new_samples:
            self._retrain()

        # Protocol: a feedback counts toward the checkpoint schedule when it
        # was ACCEPTED -- it produced at least one admissible label.  Chatter
        # that yields nothing does not advance F5/F10/F15.
        checkpoint_written = None
        if new_samples:
            self.accepted_feedback_count += 1
            checkpoint_written = self._write_checkpoint_if_due()

        latency_ms = (self.clock() - pending["started"]) * 1000.0
        update = FeedbackUpdate(
            feedback_event_id=provenance.get("feedback_event_id"),
            attributor=str(attribution.get("attributor") or ATTRIBUTOR_RULE_BASELINE),
            labels_added=len(new_samples),
            labels_rejected=len(rejected),
            rejection_reasons=rejection_summary(rejected),
            attribution_flags=list(provenance.get("attribution_flags") or []),
            update_latency_ms=round(latency_ms, 1),
            accepted_feedback_count=self.accepted_feedback_count,
            checkpoint_written=checkpoint_written,
            error=pending["error"],
        )
        self.updates.append(update)
        self._append_session_records(
            attribution=attribution,
            audit=pending["audit"],
            provenance=provenance,
            kept=kept,
            rejected=rejected,
            update=update,
        )
        return update

    # ------------------------------------------------------------ internals
    def _retrain(self) -> None:
        # Canonical order before training: SGD with a fixed seed still depends
        # on which sample sits at which index, so two sessions that received
        # the same labels in a different order would otherwise end up with
        # weights that differ in the 4th decimal.  Sorting makes the model a
        # pure function of the label SET.
        ordered = sorted(
            self.samples,
            key=lambda s: (
                str(s.source_feedback_id or ""),
                s.decision_level,
                s.preferred_subgoal,
                s.rejected_subgoal,
                json.dumps(s.condition_features, sort_keys=True, default=str),
            ),
        )
        self.adapter.reset()
        self.adapter.train(ordered, epochs=self.train_epochs, seed=self.seed)

    def _write_checkpoint_if_due(self) -> str | None:
        count = self.accepted_feedback_count
        if count not in self.checkpoints:
            return None
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"F{count}.json"
        if path.exists():
            return None
        self.adapter.save(path)
        return str(path)

    def _append_session_records(
        self,
        *,
        attribution: dict,
        audit: dict | None,
        provenance: dict,
        kept: list[dict],
        rejected: list[tuple[dict, list[str]]],
        update: FeedbackUpdate,
    ) -> None:
        def append(name: str, rows: list[dict]) -> None:
            if not rows:
                return
            with (self.session_dir / name).open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        append("attribution_preview.jsonl", [attribution])
        append("hu_attribution_provenance.jsonl", [provenance])
        append("hu_subgoal_preferences.jsonl", kept)
        append(
            "hu_subgoal_preferences.rejected.jsonl",
            [{**sample, "schema_violations": problems} for sample, problems in rejected],
        )
        if audit is not None:
            append("llm_attribution_audit.jsonl", [audit])
        append("live_feedback_updates.jsonl", [update.as_record()])

    # ------------------------------------------------------------ reporting
    def summary(self) -> dict[str, Any]:
        by_attributor: dict[str, int] = {}
        for update in self.updates:
            by_attributor[update.attributor] = by_attributor.get(update.attributor, 0) + 1
        latencies = [u.update_latency_ms for u in self.updates]
        return {
            "feedback_events": len(self.updates),
            "accepted_feedback_count": self.accepted_feedback_count,
            "labels": len(self.samples),
            "by_attributor": by_attributor,
            "llm_fallbacks": sum(1 for u in self.updates if u.attributor == ATTRIBUTOR_RULE_FALLBACK),
            "median_update_latency_ms": (sorted(latencies)[len(latencies) // 2] if latencies else None),
            "checkpoints": sorted(p.name for p in self._checkpoint_dir.glob("F*.json"))
            if self._checkpoint_dir.exists() else [],
        }
