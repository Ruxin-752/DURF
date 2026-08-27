"""Evaluate paper-style Route 2 folds against honest constant baselines.

Each held-out example is scored only by its own fold model.  Synthetic
complete-reward inference and the separate local-language deployment probe
are reported independently; neither may be presented as evidence for the
other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from src.evaluation_splits import canonical_sha256  # noqa: E402
from src.feature_schema import load_features, write_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    collate_batch,
    load_checkpoint,
    load_predictor,
    predict_reward_distribution,
)
from src.observations import reference_vector  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_planner import enumerate_feasible_subgoals  # noqa: E402
from src.subgoal_reranker import choose_subgoal  # noqa: E402


DEFAULT_FEEDBACK = ROOT / "data" / "route2_teacher_feedback.paper_v5.synthetic.json"
DEFAULT_MODEL_DIR = ROOT / "outputs" / "route2" / "paper_aligned_v5_seed137_selected"
RECEIPT_SCHEMA_VERSION = "route2-one-time-test-receipt-v2"
TIE_VOTE = "__UNRESOLVED_REWARD_TIE__"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_hash(payload: dict, *, label: str) -> str:
    supplied = payload.get("manifest_sha256")
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if supplied != expected:
        raise ValueError(f"{label} manifest hash mismatch")
    return str(supplied)


def _write_json_exclusive(path: Path, value: dict) -> None:
    """Atomically reserve the one-time test before any test row is decoded."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _checkpoint_path(model_dir: Path, relative: str) -> Path:
    candidate = (model_dir / relative).resolve()
    try:
        candidate.relative_to(model_dir.resolve())
    except ValueError as error:
        raise ValueError(f"checkpoint escapes model directory: {relative!r}") from error
    return candidate


def _validate_frozen_members(
    *,
    model_dir: Path,
    cv_manifest: dict,
    selection_manifest: dict,
    ensemble_manifest: dict,
) -> tuple[dict[int, dict], list[dict]]:
    """Bind every CV fold to the exact dev-selected checkpoint bytes."""

    cv_rows = cv_manifest.get("folds") or []
    cv_folds = {int(row["fold"]) for row in cv_rows}
    expected_folds = set(range(int(cv_manifest.get("n_folds", 0))))
    if cv_folds != expected_folds or len(cv_rows) != len(expected_folds):
        raise ValueError(
            f"cross-validation folds must be exactly {sorted(expected_folds)}"
        )
    selection_folds = {}
    for row in selection_manifest.get("folds") or []:
        fold = int(row["fold"])
        if fold in selection_folds:
            raise ValueError(f"selection repeats fold {fold}")
        selection_folds[fold] = row
    members: dict[int, dict] = {}
    for member in ensemble_manifest.get("members") or []:
        fold = int(member["fold"])
        if fold in members:
            raise ValueError(f"ensemble repeats fold {fold}")
        members[fold] = member
    if cv_folds != set(selection_folds) or cv_folds != set(members):
        raise ValueError(
            "CV, selection, and ensemble fold sets must match exactly: "
            f"cv={sorted(cv_folds)}, selection={sorted(selection_folds)}, "
            f"ensemble={sorted(members)}"
        )

    verified = []
    for fold in sorted(cv_folds):
        winner = selection_folds[fold].get("winner") or {}
        member = members[fold]
        relative = str(member.get("checkpoint") or "")
        if not relative or relative != str(winner.get("checkpoint") or ""):
            raise ValueError(f"fold {fold}: checkpoint path is not bound to selection")
        expected_sha = str(member.get("checkpoint_sha256") or "")
        if not expected_sha or expected_sha != str(winner.get("checkpoint_sha256") or ""):
            raise ValueError(f"fold {fold}: checkpoint SHA256 is not bound to selection")
        selected_candidate = str(member.get("selected_candidate") or "")
        if selected_candidate != str(winner.get("candidate_id") or ""):
            raise ValueError(f"fold {fold}: selected candidate differs across manifests")
        path = _checkpoint_path(model_dir, relative)
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"fold {fold}: frozen checkpoint hash mismatch")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        extra = checkpoint.get("extra") or {}
        expected_extra = {
            "fold": fold,
            "selection_partition": "dev",
            "test_accessed": False,
            "selected_candidate": selected_candidate,
            "candidate_registry_sha256": selection_manifest.get(
                "candidate_registry_sha256"
            ),
            "cross_validation_manifest_sha256": cv_manifest.get("manifest_sha256"),
            "corpus_sha256": selection_manifest.get("corpus_sha256"),
        }
        mismatches = {
            key: {"expected": value, "actual": extra.get(key)}
            for key, value in expected_extra.items()
            if extra.get(key) != value
        }
        if mismatches:
            raise ValueError(f"fold {fold}: checkpoint metadata mismatch: {mismatches}")
        verified.append(
            {
                "fold": fold,
                "checkpoint": relative,
                "checkpoint_sha256": actual_sha,
                "selected_candidate": selected_candidate,
            }
        )
    return members, verified


def _reward_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dimension_mask: torch.Tensor | None = None,
) -> dict:
    if dimension_mask is not None:
        predictions = predictions[:, dimension_mask]
        targets = targets[:, dimension_mask]
    if predictions.shape[1] == 0:
        raise ValueError("reward metric needs at least one selected dimension")
    active = torch.abs(targets) > 1e-8
    sign_correct = (torch.sign(predictions) == torch.sign(targets)) & active
    cosine = torch.nn.functional.cosine_similarity(predictions, targets, dim=1, eps=1e-8)
    return {
        "mse": float(torch.mean((predictions - targets) ** 2)),
        "mean_cosine_similarity": float(torch.mean(cosine)),
        "active_sign_accuracy": (
            float(sign_correct.sum() / active.sum()) if int(active.sum()) else 0.0
        ),
        "active_target_count": int(active.sum()),
        "dimensions": int(predictions.shape[1]),
        "examples": int(targets.shape[0]),
    }


def _reward_catalog(examples: list[dict]) -> dict[str, torch.Tensor]:
    catalog: dict[str, torch.Tensor] = {}
    for example in examples:
        reward_id = str(example.get("reward_config_id") or "")
        if not reward_id:
            raise ValueError("full-reward evaluation requires reward_config_id")
        vector = torch.tensor(example["target_reward"], dtype=torch.float32)
        previous = catalog.get(reward_id)
        if previous is not None and not torch.allclose(previous, vector, atol=1e-8, rtol=0.0):
            raise ValueError(f"reward_config_id {reward_id!r} has inconsistent vectors")
        catalog[reward_id] = vector
    return dict(sorted(catalog.items()))


def _varying_dimension_mask(catalog: dict[str, torch.Tensor]) -> torch.Tensor:
    matrix = torch.stack(list(catalog.values()))
    return (torch.max(matrix, dim=0).values - torch.min(matrix, dim=0).values) > 1e-8


def _nearest_config(
    predictions: torch.Tensor,
    examples: list[dict],
    catalog: dict[str, torch.Tensor],
    varying_mask: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    ids = list(catalog)
    matrix = torch.stack([catalog[reward_id] for reward_id in ids])
    selected_predictions = predictions[:, varying_mask]
    selected_matrix = matrix[:, varying_mask]
    distances = torch.sum(
        (selected_predictions[:, None, :] - selected_matrix[None, :, :]) ** 2,
        dim=2,
    )
    nearest_indices = torch.argmin(distances, dim=1)
    predicted_ids = [ids[int(index)] for index in nearest_indices]
    correct = sum(
        predicted == str(example["reward_config_id"])
        for predicted, example in zip(predicted_ids, examples)
    )
    projected = matrix[nearest_indices]
    return (
        {
            "correct": correct,
            "total": len(examples),
            "accuracy": correct / len(examples) if examples else 0.0,
            "distance_dimensions": int(varying_mask.sum()),
        },
        projected,
    )


def _fold_behavior(examples: list[dict], predictions: torch.Tensor, features: list[str]) -> dict:
    utterance_correct = 0
    unresolved_tie_votes = 0
    scenario_votes: dict[tuple[str, str], Counter] = {}
    scenario_targets: dict[tuple[str, str], set[str]] = {}
    for example, vector in zip(examples, predictions):
        weights = {feature: float(value) for feature, value in zip(features, vector.tolist())}
        decision = choose_subgoal(weights, example["context"], example["feasible_subgoals"])
        acceptable = set(example.get("acceptable_subgoals") or [example.get("expected_subgoal")])
        correct = (not decision["is_tie"]) and decision["chosen_subgoal"] in acceptable
        utterance_correct += int(correct)
        key = (str(example["reward_config_id"]), str(example["group_id"]))
        vote = TIE_VOTE if decision["is_tie"] else decision["chosen_subgoal"]
        unresolved_tie_votes += int(decision["is_tie"])
        scenario_votes.setdefault(key, Counter())[vote] += 1
        previous_targets = scenario_targets.setdefault(key, acceptable)
        if previous_targets != acceptable:
            raise ValueError(
                f"inconsistent acceptable_subgoals within reward/context group {key}"
            )

    scenario_correct = 0
    scenario_vote_ties = 0
    for key, votes in scenario_votes.items():
        maximum = max(votes.values())
        winners = [candidate for candidate, count in votes.items() if count == maximum]
        if len(winners) != 1:
            scenario_vote_ties += 1
            continue
        chosen = winners[0]
        scenario_correct += int(chosen != TIE_VOTE and chosen in scenario_targets[key])
    return {
        "per_utterance": {
            "correct": utterance_correct,
            "total": len(examples),
            "accuracy": utterance_correct / len(examples) if examples else 0.0,
        },
        "per_reward_context_majority": {
            "correct": scenario_correct,
            "total": len(scenario_votes),
            "accuracy": scenario_correct / len(scenario_votes) if scenario_votes else 0.0,
        },
        "unresolved_reward_tie_votes": unresolved_tie_votes,
        "majority_vote_ties": scenario_vote_ties,
    }


def _preference_sensitive_groups(examples: list[dict]) -> set[str]:
    choices: dict[str, set[str]] = {}
    for example in examples:
        group = str(example.get("group_id"))
        choices.setdefault(group, set()).add(str(example.get("expected_subgoal")))
    return {group for group, expected in choices.items() if len(expected) > 1}


def _behavior_suite(
    examples: list[dict],
    predictions: torch.Tensor,
    features: list[str],
    sensitive_groups: set[str],
) -> dict:
    selected = [
        index
        for index, example in enumerate(examples)
        if str(example.get("group_id")) in sensitive_groups
    ]
    cooking = [
        index
        for index in selected
        if (examples[index].get("context") or {}).get("pot_status") == "cooking"
        and (examples[index].get("context") or {}).get("agent_holding") is None
    ]

    def evaluate(indices: list[int]) -> dict:
        subset_examples = [examples[index] for index in indices]
        subset_predictions = predictions[indices] if indices else predictions[:0]
        return _fold_behavior(subset_examples, subset_predictions, features)

    return {
        "all": _fold_behavior(examples, predictions, features),
        "preference_sensitive": evaluate(selected),
        "preference_sensitive_empty_hand_cooking": evaluate(cooking),
    }


def _evaluation_suite(
    examples: list[dict],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    features: list[str],
    catalog: dict[str, torch.Tensor],
    varying_mask: torch.Tensor,
    sensitive_groups: set[str],
) -> tuple[dict, torch.Tensor]:
    nearest, projected = _nearest_config(predictions, examples, catalog, varying_mask)
    return (
        {
            "reward_metrics": _reward_metrics(predictions, targets),
            "varying_dimension_reward_metrics": _reward_metrics(
                predictions, targets, varying_mask
            ),
            "nearest_reward_config": nearest,
            "behavior_metrics": _behavior_suite(
                examples, predictions, features, sensitive_groups
            ),
        },
        projected,
    )


def _deployment_language_probe(predictor: dict) -> dict:
    context = {
        "recipe": ["tomato", "tomato", "onion"],
        "pot_ingredients": ["tomato", "tomato", "onion"],
        "pot_status": "cooking",
        "agent_holding": None,
    }
    cases = (
        (
            "positive_specific",
            "Please prepare an onion for the next round while this soup cooks.",
            "WAIT",
            {"GET_ONION"},
        ),
        (
            "positive_paraphrase",
            "Use the cooking time to get an ingredient ready for our next soup.",
            "WAIT",
            {"GET_ONION", "GET_TOMATO"},
        ),
        (
            "negative_specific",
            "Do not grab an onion this early; keep your hands free while the soup cooks.",
            "GET_ONION",
            {"WAIT"},
        ),
        (
            "negative_paraphrase",
            "Wait until this soup is done; do not fill your hands yet.",
            "GET_ONION",
            {"WAIT"},
        ),
    )
    features = predictor["features"]
    candidates = enumerate_feasible_subgoals(context)
    rows = []
    for case_id, text, grounding_subgoal, acceptable_subgoals in cases:
        counts = reference_vector(featurize_subgoal(context, grounding_subgoal), features).tolist()
        prediction = predict_reward_distribution(predictor, text, feature_counts=counts)
        decision = choose_subgoal(prediction["weights"], context, candidates, tie_fallback="WAIT")
        chosen = decision["chosen_subgoal"]
        correct = (not decision["is_tie"]) and chosen in acceptable_subgoals
        disagreement = prediction["uncertainty"]
        rows.append(
            {
                "case_id": case_id,
                "text": text,
                "acceptable_subgoals": sorted(acceptable_subgoals),
                "chosen_subgoal": chosen,
                "correct": correct,
                "reward_margin": decision["reward_margin"],
                "mean_model_disagreement": sum(disagreement.values()) / len(disagreement),
                "max_model_disagreement": max(disagreement.values()),
            }
        )
    correct = sum(row["correct"] for row in rows)
    return {
        "status": "separate_handwritten_local_language_probe_not_cv_estimate",
        "claim_scope": "behavioral response only; not complete-reward accuracy",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "cases": rows,
    }


def _validity_gate(model: dict, constants: dict[str, dict]) -> dict:
    behavior_key = "preference_sensitive_empty_hand_cooking"
    behavior_total = model["behavior_metrics"][behavior_key]["per_reward_context_majority"]["total"]
    if not behavior_total:
        behavior_key = "preference_sensitive"
        behavior_total = model["behavior_metrics"][behavior_key]["per_reward_context_majority"]["total"]
    best_constant_mse = min(
        row["varying_dimension_reward_metrics"]["mse"] for row in constants.values()
    )
    best_constant_nearest = max(
        row["nearest_reward_config"]["accuracy"] for row in constants.values()
    )
    best_constant_behavior = max(
        row["behavior_metrics"][behavior_key]["per_reward_context_majority"]["accuracy"]
        for row in constants.values()
    )
    checks = {
        "varying_dimension_mse": (
            model["varying_dimension_reward_metrics"]["mse"] < best_constant_mse
        ),
        "nearest_reward_config_accuracy": (
            model["nearest_reward_config"]["accuracy"] > best_constant_nearest
        ),
        "preference_sensitive_behavior": (
            behavior_total > 0
            and model["behavior_metrics"][behavior_key]["per_reward_context_majority"]["accuracy"]
            > best_constant_behavior
        ),
    }
    valid = all(checks.values())
    status = (
        "inconclusive_no_preference_sensitive_examples"
        if not behavior_total
        else "valid_better_than_constant_baselines"
        if valid
        else "invalid_not_better_than_constant_baselines"
    )
    return {
        "status": status,
        "claim_allowed": valid,
        "required_checks": checks,
        "behavior_subset": behavior_key,
        "constant_thresholds": {
            "best_varying_dimension_mse": best_constant_mse,
            "best_nearest_reward_config_accuracy": best_constant_nearest,
            "best_preference_sensitive_behavior_accuracy": best_constant_behavior,
        },
        "rule": (
            "Do not claim successful complete-reward inference unless the model "
            "beats both constant baselines on every required preference metric."
        ),
    }


def _compact_suite_metrics(suite: dict) -> dict[str, float]:
    behavior = suite["behavior_metrics"]
    return {
        "full_reward_mse": float(suite["reward_metrics"]["mse"]),
        "varying_dimension_mse": float(
            suite["varying_dimension_reward_metrics"]["mse"]
        ),
        "mean_cosine_similarity": float(
            suite["reward_metrics"]["mean_cosine_similarity"]
        ),
        "active_sign_accuracy": float(
            suite["reward_metrics"]["active_sign_accuracy"]
        ),
        "nearest_reward_config_accuracy": float(
            suite["nearest_reward_config"]["accuracy"]
        ),
        "all_behavior_majority_accuracy": float(
            behavior["all"]["per_reward_context_majority"]["accuracy"]
        ),
        "preference_sensitive_behavior_majority_accuracy": float(
            behavior["preference_sensitive"]["per_reward_context_majority"][
                "accuracy"
            ]
        ),
        "preference_sensitive_cooking_majority_accuracy": float(
            behavior["preference_sensitive_empty_hand_cooking"][
                "per_reward_context_majority"
            ]["accuracy"]
        ),
    }


def _fold_macro_metrics(fold_reports: list[dict]) -> dict:
    if not fold_reports:
        return {}
    system_names = list(fold_reports[0]["systems"])
    output = {}
    for system in system_names:
        rows = [_compact_suite_metrics(fold["systems"][system]) for fold in fold_reports]
        output[system] = {
            "folds": len(rows),
            **{
                key: sum(row[key] for row in rows) / len(rows)
                for key in rows[0]
            },
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--allow-legacy-unfrozen-diagnostic",
        action="store_true",
        help=(
            "Allow old model directories without a frozen dev-selection manifest. "
            "Results from this diagnostic mode are not one-time test evidence."
        ),
    )
    args = parser.parse_args()

    manifest = json.loads(
        (args.model_dir / "cross_validation_manifest.json").read_text(encoding="utf-8")
    )
    ensemble_manifest = json.loads(
        (args.model_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
    )
    cv_manifest_sha256 = _verify_manifest_hash(manifest, label="cross-validation")
    ensemble_manifest_sha256 = _verify_manifest_hash(
        ensemble_manifest, label="ensemble"
    )
    selection_path = args.model_dir / "selection_manifest.json"
    selection_manifest = None
    receipt_path = args.model_dir / "test_evaluation_receipt.json"
    if not selection_path.exists() and not args.allow_legacy_unfrozen_diagnostic:
        raise FileNotFoundError(
            "strict test evaluation requires selection_manifest.json; run the "
            "dev-only selector and finalize its checkpoint hashes first"
        )

    frozen_members: dict[int, dict] | None = None
    verified_checkpoints: list[dict] = []
    selection_manifest_sha256 = None
    if selection_path.exists():
        selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_manifest_sha256 = _verify_manifest_hash(
            selection_manifest, label="selection"
        )
        if selection_manifest.get("test_accessed") is not False:
            raise ValueError("selection manifest was not frozen before test")
        if int(selection_manifest.get("test_examples_evaluated", -1)) != 0:
            raise ValueError("selection manifest already reports test evaluation")
        if ensemble_manifest.get("test_accessed") is not False:
            raise ValueError("ensemble manifest was not frozen before test")
        if canonical_sha256(selection_manifest.get("candidate_registry")) != str(
            selection_manifest.get("candidate_registry_sha256") or ""
        ):
            raise ValueError("selection candidate registry hash mismatch")
        if (
            ensemble_manifest.get("selection_manifest_sha256")
            != selection_manifest_sha256
        ):
            raise ValueError("ensemble does not match the frozen selection manifest")
        if (
            selection_manifest.get("cross_validation_manifest_sha256")
            != cv_manifest_sha256
        ):
            raise ValueError("selection and cross-validation manifests do not match")
        if (
            selection_manifest.get("corpus_sha256")
            != ensemble_manifest.get("corpus_sha256")
        ):
            raise ValueError("selection and ensemble corpus hashes do not match")
        frozen_members, verified_checkpoints = _validate_frozen_members(
            model_dir=args.model_dir,
            cv_manifest=manifest,
            selection_manifest=selection_manifest,
            ensemble_manifest=ensemble_manifest,
        )
    if ensemble_manifest.get("cross_validation_manifest_sha256") != cv_manifest_sha256:
        raise ValueError("ensemble and cross-validation manifest do not match")
    if receipt_path.exists():
        raise RuntimeError(
            "this frozen selection has already reserved or completed its one-time "
            "test; refusing to rerun"
        )

    if selection_manifest is not None:
        started_receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "started",
            "test_accessed": True,
            "partition": "test",
            "selection_manifest_sha256": selection_manifest_sha256,
            "ensemble_manifest_sha256": ensemble_manifest_sha256,
            "cross_validation_manifest_sha256": cv_manifest_sha256,
            "corpus_sha256": selection_manifest["corpus_sha256"],
            "verified_checkpoints": verified_checkpoints,
        }
        started_receipt["receipt_sha256"] = canonical_sha256(started_receipt)
        _write_json_exclusive(receipt_path, started_receipt)

    # The one-time receipt is now reserved.  Only from here may the corpus be
    # decoded and held-out target rows be materialized.
    raw = args.feedback.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if ensemble_manifest.get("corpus_sha256") != raw_sha256:
        raise ValueError("evaluation corpus SHA256 does not match the trained ensemble")
    dataset = build_dataset(
        json.loads(raw.decode("utf-8")),
        load_probe_states(args.probe_states),
        load_features(),
        paper_cv_mode=True,
        source_corpus_sha256=raw_sha256,
    )

    catalog = _reward_catalog(dataset["examples"])
    varying_mask = _varying_dimension_mask(catalog)
    sensitive_groups = _preference_sensitive_groups(dataset["examples"])
    canonical = catalog.get("reward_00_gold")
    if canonical is None:
        raise ValueError("canonical baseline requires reward_00_gold")

    fold_reports = []
    all_examples: list[dict] = []
    prediction_rows: dict[str, list[torch.Tensor]] = {
        "model": [],
        "text_only": [],
        "trajectory_only": [],
        "train_mean": [],
        "canonical": [],
    }
    target_rows: list[torch.Tensor] = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        checkpoint_path = (
            _checkpoint_path(
                args.model_dir, str(frozen_members[fold_id]["checkpoint"])
            )
            if frozen_members is not None
            else args.model_dir / f"fold_{fold_id:02d}" / "model.pt"
        )
        model, vocab, features, use_feature_counts = load_checkpoint(
            checkpoint_path
        )
        if features != dataset["features"]:
            raise ValueError(f"fold {fold_id} feature order differs from evaluation corpus")
        examples = [dataset["examples"][index] for index in fold["test_indices"]]
        batch = collate_batch(examples, vocab)
        normal_counts = batch["feature_counts"] if use_feature_counts else torch.zeros_like(batch["feature_counts"])
        trajectory_only_examples = [dict(example, tokens=[]) for example in examples]
        trajectory_only_batch = collate_batch(trajectory_only_examples, vocab)
        with torch.no_grad():
            normal = model(batch["tokens"], batch["offsets"], normal_counts)
            text_only = model(
                batch["tokens"], batch["offsets"], torch.zeros_like(batch["feature_counts"])
            )
            trajectory_only = model(
                trajectory_only_batch["tokens"],
                trajectory_only_batch["offsets"],
                trajectory_only_batch["feature_counts"]
                if use_feature_counts
                else torch.zeros_like(trajectory_only_batch["feature_counts"]),
            )
        train_targets = torch.tensor(
            [dataset["examples"][index]["target_reward"] for index in fold["train_indices"]],
            dtype=torch.float32,
        )
        train_mean = torch.mean(train_targets, dim=0).repeat(len(examples), 1)
        canonical_rows = canonical.repeat(len(examples), 1)
        fold_system_predictions = {
            "model": normal,
            "text_only": text_only,
            "trajectory_only": trajectory_only,
            "train_mean": train_mean,
            "canonical": canonical_rows,
        }
        fold_suites = {}
        for system, fold_predictions in fold_system_predictions.items():
            fold_suites[system], _ = _evaluation_suite(
                examples,
                fold_predictions,
                batch["targets"],
                features,
                catalog,
                varying_mask,
                sensitive_groups,
            )
        fold_reports.append(
            {
                "fold": fold_id,
                "test_examples": len(examples),
                "model": fold_suites["model"],
                "systems": fold_suites,
                "coverage": fold.get("coverage"),
                "heldout_overlap_audit": fold.get("heldout_overlap_audit"),
            }
        )
        all_examples.extend(examples)
        target_rows.append(batch["targets"])
        prediction_rows["model"].append(normal)
        prediction_rows["text_only"].append(text_only)
        prediction_rows["trajectory_only"].append(trajectory_only)
        prediction_rows["train_mean"].append(train_mean)
        prediction_rows["canonical"].append(canonical_rows)

    targets = torch.cat(target_rows)
    suites: dict[str, dict] = {}
    projected_model: torch.Tensor | None = None
    for name, rows in prediction_rows.items():
        suite, projected = _evaluation_suite(
            all_examples,
            torch.cat(rows),
            targets,
            dataset["features"],
            catalog,
            varying_mask,
            sensitive_groups,
        )
        suites[name] = suite
        if name == "model":
            projected_model = projected
    assert projected_model is not None
    projected_suite, _ = _evaluation_suite(
        all_examples,
        projected_model,
        targets,
        dataset["features"],
        catalog,
        varying_mask,
        sensitive_groups,
    )
    constants = {"train_mean": suites["train_mean"], "canonical": suites["canonical"]}
    validity = _validity_gate(suites["model"], constants)

    predictor = load_predictor(args.model_dir / "ensemble_manifest.json")
    report = {
        "protocol": "paper_teacher_and_reward_holdout",
        "partition": "test",
        "selection_manifest_sha256": (
            selection_manifest.get("manifest_sha256")
            if selection_manifest is not None
            else None
        ),
        "target_scope": (
            "identifiable synthetic reward-conditioned language -> complete teacher "
            "reward; an upper-bound systems check, not a human local-language estimate"
        ),
        "leakage_rule": "Each CV test example is scored only by its own fold model.",
        "corpus_sha256": raw_sha256,
        "cross_validation_coverage": manifest.get("coverage"),
        "reward_dimensions": {
            "total": len(dataset["features"]),
            "varying": int(varying_mask.sum()),
            "constant": int((~varying_mask).sum()),
        },
        "preference_sensitive_contexts": len(sensitive_groups),
        "folds": fold_reports,
        "aggregate_held_out": {
            "model": suites["model"],
            "ablations": {
                "text_only_input": suites["text_only"],
                "trajectory_only_input": suites["trajectory_only"],
            },
            "constant_baselines": constants,
            "model_projected_to_nearest_reward_config": projected_suite,
            # Compatibility aliases for older report readers.
            "reward_metrics": suites["model"]["reward_metrics"],
            "behavior_metrics": suites["model"]["behavior_metrics"]["all"],
            "fold_macro_unweighted": _fold_macro_metrics(fold_reports),
        },
        "validity_gate": validity,
        "ensemble_deployment_language_probe": _deployment_language_probe(predictor),
    }
    output = args.report_out or args.model_dir / "evaluation_report.json"
    write_json(output, report)
    if selection_manifest is not None:
        completed_receipt = {
            key: value
            for key, value in started_receipt.items()
            if key != "receipt_sha256"
        }
        completed_receipt.update(
            {
                "status": "completed",
                "evaluation_report": str(output),
                "evaluation_report_sha256": canonical_sha256(report),
                "evaluation_report_file_sha256": _file_sha256(output),
            }
        )
        completed_receipt["receipt_sha256"] = canonical_sha256(completed_receipt)
        _write_json_atomic(receipt_path, completed_receipt)
    print(json.dumps(report["aggregate_held_out"], indent=2))
    print(json.dumps(report["validity_gate"], indent=2))
    print(json.dumps(report["ensemble_deployment_language_probe"], indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
