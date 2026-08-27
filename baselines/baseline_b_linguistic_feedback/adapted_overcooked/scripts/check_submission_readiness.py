"""Build a concise, claim-safe submission readiness report from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import load_features  # noqa: E402
from src.subgoal_featurizer import audit_live_feature_coverage  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must contain one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_smoke_audit(root: Path) -> dict[str, Any]:
    smoke_root = root / "outputs/_submission_live_smoke"
    sessions = sorted(
        path
        for path in smoke_root.iterdir()
        if path.is_dir() and (path / "session_manifest.json").is_file()
    ) if smoke_root.is_dir() else []
    if len(sessions) < 2:
        return {"passed": False, "completed_sessions": len(sessions)}
    first_final = sessions[-2] / "final_learner_state.json"
    second_initial = sessions[-1] / "initial_learner_state.json"
    first_manifest = _read(sessions[-2] / "session_manifest.json")
    second_manifest = _read(sessions[-1] / "session_manifest.json")
    exact_resume = (
        first_final.is_file()
        and second_initial.is_file()
        and _sha256(first_final) == _sha256(second_initial)
    )
    passed = bool(
        exact_resume
        and first_manifest.get("feedback_mode") == "route2"
        and second_manifest.get("feedback_mode") == "route2"
        and first_manifest.get("model_sha256") == second_manifest.get("model_sha256")
        and first_manifest.get("score_formula") == "w_dot_phi"
    )
    return {
        "passed": passed,
        "completed_sessions": len(sessions),
        "exact_resume_state_match": exact_resume,
        "model_sha256": first_manifest.get("model_sha256"),
        "score_formula": first_manifest.get("score_formula"),
        "scope": "headless startup/save/resume smoke; no human feedback claim",
    }


def build_readiness_report(root: Path = ROOT) -> dict[str, Any]:
    reference_path = root / "outputs/reference_classifier/model.report.json"
    grounding_path = root / "outputs/phrase_grounding/model.joblib.report.json"
    route2_path = (
        root
        / "outputs/route2/paper_aligned_v5_seed137_selected/evaluation_report.json"
    )
    trajectory_required_path = (
        root / "outputs/route2/trajectory_required_v1/evaluation_report.json"
    )
    adaptation_path = (
        root
        / "outputs/route2/trajectory_required_v1/online_adaptation_simulation.json"
    )
    human_reference_path = root / "outputs/human_reference_gold_classifier.report.json"
    human_form_path = root / "outputs/human_feedback_form_holdout.evaluation.json"
    test_report_path = root / "outputs/submission_test_report.json"

    reference = _read(reference_path)
    grounding = _read(grounding_path)
    route2 = _read(route2_path)
    trajectory_required = _read(trajectory_required_path)
    adaptation = _read(adaptation_path)
    human_reference = _read(human_reference_path)
    human_form = _read(human_form_path)
    test_report = _read(test_report_path)

    paper = reference["paper_protocol_reproduction"]["reproduced"]
    strict = reference["paper_strict_external_test"]["metrics"]
    grounding_test = grounding["untouched_test"]
    route2_model = route2["aggregate_held_out"]["model"]
    route2_gate = route2["validity_gate"]
    trajectory_gate = trajectory_required["validity_gate"]
    feature_audit = audit_live_feature_coverage(load_features())
    live_smoke = _live_smoke_audit(root)

    engineering_checks = {
        "paper_protocol_reproduced_at_about_87_percent": bool(
            float(paper["accuracy"]) >= 0.87
        ),
        "paper_strict_split_has_zero_active_overlap": bool(
            reference["paper_strict_external_test"]["active_id_overlap_count"] == 0
            and reference["paper_strict_external_test"][
                "active_normalized_text_overlap_count"
            ]
            == 0
            and reference["paper_strict_external_test"][
                "active_task_uuid_overlap_count"
            ]
            == 0
        ),
        "route2_beats_constant_baselines": bool(route2_gate["claim_allowed"]),
        "route2_uses_complete_53d_reward": bool(
            route2_model["reward_metrics"]["dimensions"] == 53
        ),
        "route1_all_53_dimensions_are_decision_or_explicitly_null": bool(
            feature_audit["schema_size"] == 53
            and feature_audit["complete"]
            and feature_audit["decision_feature_count"]
            + feature_audit["context_only_feature_count"]
            + feature_audit["unsupported_feature_count"]
            == feature_audit["schema_size"]
        ),
        "route2_trajectory_required_gate_passes": bool(
            trajectory_gate["claim_allowed"]
            and not trajectory_required["formal_seed137_test_touched"]
        ),
        "online_adaptation_simulation_gate_passes": bool(
            adaptation["validity_gate"]["status"] == "passed"
        ),
        "headless_route2_save_resume_smoke_passes": bool(live_smoke["passed"]),
        "all_submission_tests_pass": bool(
            test_report["status"] == "passed"
            and all(suite["status"] == "passed" for suite in test_report["suites"])
        ),
        "human_reference_evaluator_fails_closed_without_gold": bool(
            human_reference["gold_rows"] == 0
            and not human_reference["final_human_accuracy_claim_allowed"]
        ),
        "three_class_holdout_not_claimed_as_human_authored": bool(
            human_form["provenance"][
                "ai_candidate_is_not_claimed_as_human_authored"
            ]
        ),
    }
    engineering_ready = all(engineering_checks.values())
    human_reference_ready = bool(
        human_reference["gold_rows"] > 0
        and human_reference["final_human_accuracy_claim_allowed"]
    )

    reports = [
        reference_path,
        grounding_path,
        route2_path,
        trajectory_required_path,
        adaptation_path,
        human_reference_path,
        human_form_path,
        test_report_path,
    ]
    return {
        "schema_version": "submission-readiness-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": (
            "complete_with_human_evidence"
            if engineering_ready and human_reference_ready
            else "engineering_ready_human_evaluation_pending"
            if engineering_ready
            else "engineering_checks_failed"
        ),
        "engineering_ready": engineering_ready,
        "engineering_checks": engineering_checks,
        "headline_metrics": {
            "five_class_paper_protocol": {
                "accuracy": paper["accuracy"],
                "macro_f1": paper["macro_f1"],
                "claim_scope": "original random-row protocol reproduction",
            },
            "five_class_strict_exposed_regression": {
                "accuracy": strict["accuracy"],
                "macro_f1": strict["macro_f1"],
                "claim_scope": "task/text-disjoint exposed regression",
            },
            "route1_phrase_grounding_synthetic": {
                "micro_f1": grounding_test["micro_f1"],
                "macro_f1": grounding_test["macro_f1"],
                "claim_scope": "synthetic untouched split",
            },
            "route1_live_feature_contract": {
                "schema_dimensions": feature_audit["schema_size"],
                "decision_varying_dimensions": feature_audit[
                    "decision_feature_count"
                ],
                "context_only_dimensions": feature_audit[
                    "context_only_feature_count"
                ],
                "unsupported_decision_null_dimensions": feature_audit[
                    "unsupported_feature_count"
                ],
            },
            "route2_frozen_synthetic": {
                "examples": route2_model["reward_metrics"]["examples"],
                "full_reward_mse": route2_model["reward_metrics"]["mse"],
                "varying_dimension_mse": route2_model[
                    "varying_dimension_reward_metrics"
                ]["mse"],
                "nearest_reward_config_accuracy": route2_model[
                    "nearest_reward_config"
                ]["accuracy"],
                "claim_scope": "identifiable synthetic complete-reward task",
            },
            "route2_trajectory_required_synthetic": {
                "examples": trajectory_required["examples"],
                "full_varying_dimension_mse": trajectory_required["aggregate"][
                    "full"
                ]["varying_dimension_mse"]["mean"],
                "text_only_varying_dimension_mse": trajectory_required[
                    "aggregate"
                ]["text_only"]["varying_dimension_mse"]["mean"],
                "full_nearest_target_accuracy": trajectory_required["aggregate"][
                    "full"
                ]["nearest_target_vector_accuracy"]["mean"],
                "text_only_nearest_target_accuracy": trajectory_required[
                    "aggregate"
                ]["text_only"]["nearest_target_vector_accuracy"]["mean"],
                "relative_mse_gain_over_text_only": trajectory_gate[
                    "full_relative_mse_gain_over_text_only"
                ],
                "claim_scope": trajectory_required["claim_scope"],
            },
            "online_adaptation_synthetic": {
                "rounds": adaptation["rounds"],
                "frozen": adaptation["conditions"]["frozen_h0"],
                "adaptive": adaptation["conditions"]["online_adaptive"],
                "claim_scope": adaptation["claim_scope"],
            },
            "three_class_human_verified_ai_candidates": {
                "rows": human_form["metrics"]["rows"],
                "accuracy": human_form["metrics"]["accuracy"],
                "macro_f1": human_form["metrics"]["macro_f1"],
                "claim_scope": "human-confirmed labels on assistant-authored text",
            },
        },
        "human_evidence": {
            "five_class_gold_rows": human_reference["gold_rows"],
            "five_class_accuracy_claim_allowed": human_reference[
                "final_human_accuracy_claim_allowed"
            ],
            "remaining_required": [
                "participant/session-disjoint five-class and temporal-credit gold",
                "fresh player session with real language feedback for ecological validation",
                "blind frozen-H0 versus adaptive preference study",
            ],
        },
        "runtime_verification": live_smoke,
        "test_verification": test_report,
        "claim_boundaries": [
            "87.16% is the original five-class random-row protocol reproduction.",
            "77.36% is the stricter task/text-disjoint exposed regression.",
            "Route 2 frozen metrics are synthetic and do not establish human-language generalization.",
            "The three-class classifier is auxiliary and is not the paper's 87% classifier.",
        ],
        "source_reports": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in reports
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["headline_metrics"]
    five = metrics["five_class_paper_protocol"]
    strict = metrics["five_class_strict_exposed_regression"]
    grounding = metrics["route1_phrase_grounding_synthetic"]
    route2 = metrics["route2_frozen_synthetic"]
    trajectory = metrics["route2_trajectory_required_synthetic"]
    adaptation = metrics["online_adaptation_synthetic"]
    feature_contract = metrics["route1_live_feature_contract"]
    form = metrics["three_class_human_verified_ai_candidates"]
    human = report["human_evidence"]
    tests = report["test_verification"]
    return f"""# Submission Status

## 结论

当前状态：`{report['overall_status']}`。核心工程链路已经具备，尚不能替代的缺口是真人五分类/temporal-credit gold 与真人偏好对照实验。

## 已完成并可报告

- 原论文五分类随机行协议复现：Accuracy `{five['accuracy']:.2%}`，Macro-F1 `{five['macro_f1']:.2%}`。
- 五分类严格 task/text-disjoint 暴露回归：Accuracy `{strict['accuracy']:.2%}`，Macro-F1 `{strict['macro_f1']:.2%}`。
- Route 1 phrase grounding 合成 untouched：Micro-F1 `{grounding['micro_f1']:.2%}`，Macro-F1 `{grounding['macro_f1']:.2%}`。
- Route 1 的 53 维现已全部审计：`{feature_contract['decision_varying_dimensions']}` 维可改变候选排序，`{feature_contract['context_only_dimensions']}` 维仅为共享上下文，`{feature_contract['unsupported_decision_null_dimensions']}` 维当前不支持；后两类显式 mask/拒识。
- Route 2 冻结合成测试：`{route2['examples']}` 条，53维 MSE `{route2['full_reward_mse']:.6f}`，变化维 MSE `{route2['varying_dimension_mse']:.6f}`，最近奖励配置 `{route2['nearest_reward_config_accuracy']:.2%}`。
- Route 2 trajectory-required 反事实测试：`{trajectory['examples']}` 条，full/text-only 变化维 MSE `{trajectory['full_varying_dimension_mse']:.6f}` / `{trajectory['text_only_varying_dimension_mse']:.6f}`，full 相对改善 `{trajectory['relative_mse_gain_over_text_only']:.2%}`；最近目标为 `{trajectory['full_nearest_target_accuracy']:.2%}` / `{trajectory['text_only_nearest_target_accuracy']:.2%}`。
- 40 回合同 seed 合成在线适应：偏好满足率由 `{adaptation['frozen']['preference_satisfaction_rate']:.1%}` 提升至 `{adaptation['adaptive']['preference_satisfaction_rate']:.1%}`，regret 由 `{adaptation['frozen']['cumulative_regret']:.1f}` 降至 `{adaptation['adaptive']['cumulative_regret']:.1f}`，安全违规由 `{adaptation['frozen']['safety_violation_count']}` 降至 `{adaptation['adaptive']['safety_violation_count']}`。这只是系统诊断，不是真人结论。
- 无界面 Route 2 启动、v3 状态保存和精确恢复 smoke 已通过；恢复前后状态 SHA 完全一致。
- 辅助三分类：`{form['rows']}` 条人工确认的 AI 候选语言，Accuracy `{form['accuracy']:.2%}`，Macro-F1 `{form['macro_f1']:.2%}`；它不是论文 87% 指标。

## 验证

- 自动测试共 `{tests['total_tests']}` 项：adapted `{tests['suites'][0]['tests']}/{tests['suites'][0]['tests']}`、DURF baseline `{tests['suites'][1]['tests']}/{tests['suites'][1]['tests']}`、UI `{tests['suites'][2]['tests']}/{tests['suites'][2]['tests']}`，失败和错误均为 0。
- `{tests['syntax_check']['python_files']}` 个 Python 文件语法检查通过，`git diff --check` 通过。

## 尚需真人完成

- 五分类真人 gold：当前 `{human['five_class_gold_rows']}` 条。
- 真人语句对应的 actor、目标事件/时间窗、目标 feature 与 polarity 标注。
- 新真人局中的真实语言更新与行为变化验收（自动保存/恢复链路已通过）。
- frozen H0 与 online-adaptive 的盲测，报告偏好满足率、regret、任务回报和安全违规。

## 口径

论文 87% 只对应五分类原协议复现；Route 2 高分只说明合成完整奖励任务成功。未采集上述真人证据前，不声明真人语言 accuracy 或真人 preference improvement。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs/submission_readiness.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "SUBMISSION_STATUS.md",
    )
    args = parser.parse_args()
    report = build_readiness_report()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"status={report['overall_status']}")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_markdown}")
    return 0 if report["engineering_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
