"""Tkinter review window for feedback attribution sessions.

The GUI reads one game session, exports review_items.jsonl, and lets a human
reviewer write review_decisions.jsonl.  It intentionally does not mutate the
automatic attribution files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .review_io import (
    build_review_items,
    export_review_items,
    parse_json_field,
    upsert_review_decision,
)
from .review_replay import ReplayWindow, select_replay_steps
from .subgoal_preferences import COORDINATION_SUBGOALS, HU_SUBGOALS


DECISION_OPTIONS = [
    "accept",
    "revise",
    "reject",
    "provenance_only",
    "map_to_existing",
    "needs_revision",
    "needs_more_context",
]

SCHEMA_ACTION_OPTIONS = [
    "none",
    "accept_new_event",
    "revise_new_event",
    "map_to_existing",
    "add_condition_only",
    "reject",
]

SUBGOAL_DESCRIPTIONS = {
    "GET_TOMATO": "去取得番茄",
    "PUT_TOMATO_IN_POT": "把番茄放入锅",
    "GET_ONION": "去取得洋葱",
    "PUT_ONION_IN_POT": "把洋葱放入锅",
    "GET_DISH": "去取得盘子",
    "PICKUP_SOUP": "用盘子取起成品汤",
    "SERVE_SOUP": "把汤送到出餐口",
    "WAIT": "等待、不推进当前任务",
    "YIELD": "让路",
    "WAIT_NEAR_POT": "在锅附近等待",
    "PUT_DOWN_OBJECT": "放下手中物品",
    "GET_USEFUL_INGREDIENT": "获取当前有用的原料",
    "CONTINUE_CURRENT_SUBGOAL": "继续当前 subgoal",
}

NON_TRAINING_DECISIONS = {
    "reject",
    "provenance_only",
    "needs_revision",
    "needs_more_context",
}


def to_pretty_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def set_text(widget: tk.Text, value: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert("1.0", value)


def read_text(widget: tk.Text) -> str:
    return widget.get("1.0", tk.END).strip()


def short_text(text: str | None, limit: int = 42) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def object_name(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        return str(value.get("name") or value.get("type") or "-")
    return str(value)


def format_list(values: list[Any] | None) -> str:
    return ", ".join(str(value) for value in values or []) or "-"


def set_subgoal_selection(widget: tk.Listbox, values: list[str] | None) -> None:
    selected = set(values or [])
    widget.selection_clear(0, tk.END)
    for index, subgoal in enumerate(HU_SUBGOALS):
        if subgoal in selected:
            widget.selection_set(index)


def selected_subgoals(widget: tk.Listbox) -> list[str]:
    return [HU_SUBGOALS[int(index)] for index in widget.curselection()]


def trajectory_candidate_subgoals(item: dict[str, Any]) -> list[str]:
    attribution = item.get("attribution") or {}
    decision = item.get("existing_decision") or item.get("default_decision") or {}
    window = (
        decision.get("approved_time_window")
        or attribution.get("target_time_window")
        or []
    )
    start = window[0] if len(window) == 2 else None
    end = window[1] if len(window) == 2 else None
    observed = []
    for step in item.get("recent_steps") or []:
        timestep = step.get("total_step")
        if start is not None and end is not None:
            if timestep is None or not int(start) <= int(timestep) <= int(end):
                continue
        for candidate in step.get("candidate_subgoals") or []:
            subgoal = candidate.get("subgoal")
            if subgoal in HU_SUBGOALS and subgoal not in observed:
                observed.append(subgoal)
        subgoal = step.get("ai_subgoal")
        if subgoal in HU_SUBGOALS and subgoal not in observed:
            observed.append(subgoal)
    return observed


def active_conditions(conditions: dict[str, Any] | None) -> list[str]:
    """Return a compact, human-readable subset of conditions worth reviewing."""
    conditions = conditions or {}
    priority_keys = [
        "needed_ingredient",
        "human_inferred_subgoal",
        "ai_empty_handed",
        "ai_has_dish",
        "ai_has_onion",
        "ai_has_tomato",
        "human_has_dish",
        "human_has_onion",
        "human_has_tomato",
        "human_holding_last_needed_ingredient",
        "pot_empty",
        "pot_partially_filled",
        "pot_cooking_or_ready",
        "recipe_needs_onion",
        "recipe_needs_tomato",
        "human_trying_to_pass",
        "ai_on_human_path",
        "human_waiting_near_pot",
        "narrow_corridor",
        "useful_counter_object_available",
        "useful_counter_object_type",
        "useful_counter_object_closer_than_dispenser",
        "useful_counter_object_lower_task_cost_than_dispenser",
    ]
    lines = []
    for key in priority_keys:
        value = conditions.get(key)
        if value is True or (value not in (None, False, "", [], {})):
            lines.append(f"{key}={value}")
    return lines


def matching_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    attribution = item.get("attribution") or {}
    target_event = attribution.get("target_event")
    candidates = item.get("nearby_candidate_events") or []
    matches = [
        event
        for event in candidates
        if event.get("event_type") == target_event
    ]
    if not matches:
        return None
    window = attribution.get("target_time_window") or []
    if len(window) == 2:
        exact = [
            event
            for event in matches
            if event.get("start_timestep") == window[0]
            and event.get("end_timestep") == window[1]
        ]
        if exact:
            return exact[0]
    return matches[0]


def render_review_summary(item: dict[str, Any]) -> str:
    attribution = item.get("attribution") or {}
    default = item.get("existing_decision") or item.get("default_decision") or {}
    preference = default.get("approved_preference") or {}
    conditions = (
        default.get("approved_condition_overrides")
        or item.get("condition_at_feedback")
        or {}
    )
    candidate = matching_candidate(item)
    evidence = (candidate or {}).get("evidence") or {}
    schema_reviews = item.get("schema_reviews") or []

    warnings = []
    if attribution.get("needs_clarification"):
        warnings.append("自动归因认为语义需要进一步澄清")
    if not attribution.get("target_event"):
        warnings.append("尚未锁定 target event")
    if not preference.get("preferred_subgoals"):
        warnings.append("缺少 preferred subgoal")
    if not preference.get("rejected_subgoals"):
        warnings.append("缺少 rejected subgoal")
    if attribution.get("target_event") and candidate is None:
        warnings.append("目标 event 没有匹配到附近的程序检测证据")
    if schema_reviews:
        warnings.append(f"LLM 提出了 {len(schema_reviews)} 条 schema 建议，需单独判断")

    lines = [
        "你只需要确认下面 4 件事",
        "=" * 46,
        "",
        "玩家原话",
        f'  “{item.get("feedback_text") or "-"}”',
        f"  发生时间：t={item.get('feedback_total_step')}  "
        f"episode={item.get('episode')}  step={item.get('episode_step')}",
        "",
        "1. 时间：这句话评价的是哪段行为？",
        f"  自动建议：{attribution.get('target_time_window') or '-'}",
        "",
        "2. 行为：玩家说的是哪个 AI event？",
        f"  自动建议：{attribution.get('target_event') or '-'}",
        f"  自动置信度：{attribution.get('confidence', '-')}",
    ]
    if candidate:
        lines.extend(
            [
                f"  检测证据：t={candidate.get('start_timestep')}.."
                f"{candidate.get('end_timestep')} | "
                f"{candidate.get('event_valence', '-')}",
                f"  规则说明：{evidence.get('reason') or '-'}",
            ]
        )
    lines.extend(
        [
            f"  附近共有 {len(item.get('nearby_candidate_events') or [])} 个候选，"
            "可在“候选事件”页比较",
            "",
            "3. 条件：这个偏好在什么状态下成立？",
        ]
    )
    condition_lines = active_conditions(conditions)
    if condition_lines:
        lines.extend(f"  - {line}" for line in condition_lines)
    else:
        lines.append("  - 未提取到明显条件")
    lines.extend(
        [
            "",
            "4. 偏好：在上述条件下，Hu 应如何相对排序？",
            f"  更希望 AI 选择：{format_list(preference.get('preferred_subgoals'))}",
            f"  更不希望 AI 选择：{format_list(preference.get('rejected_subgoals'))}",
            f"  该时间段实际出现过：{format_list(trajectory_candidate_subgoals(item))}",
            "",
            "审核动作",
            "  全部正确：Decision=accept，确认训练勾选后保存并下一条",
            "  有一项错误：Decision=revise，在右侧多选并修正",
            "  证据不足：Decision=needs_more_context，取消训练勾选",
        ]
    )
    if warnings:
        lines.extend(["", "需要特别注意"])
        lines.extend(f"  ! {warning}" for warning in warnings)
    return "\n".join(lines)


def render_raw_context(item: dict[str, Any]) -> str:
    context = {
        "feedback": {
            "text": item.get("feedback_text"),
            "value": item.get("feedback_value"),
            "role": item.get("feedback_role"),
            "total_step": item.get("feedback_total_step"),
            "episode": item.get("episode"),
            "episode_step": item.get("episode_step"),
        },
        "automatic_attribution": item.get("attribution"),
        "hu_provenance": item.get("provenance"),
        "schema_reviews": item.get("schema_reviews"),
        "condition_at_feedback": item.get("condition_at_feedback"),
        "nearby_candidate_events": item.get("nearby_candidate_events"),
        "nearby_probe_hits": item.get("nearby_probe_hits"),
        "recent_steps": item.get("recent_steps"),
    }
    return to_pretty_json(context)


class ReviewApp:
    def __init__(self, root: tk.Tk, session_dir: Path, lookback_steps: int) -> None:
        self.root = root
        self.session_dir = session_dir
        self.lookback_steps = lookback_steps
        self.items = build_review_items(session_dir, lookback_steps=lookback_steps)
        self.current_index = 0

        self.root.title(f"DURF attribution review - {session_dir.name}")
        self.root.geometry("1450x860")

        self._build_widgets()
        self._refresh_list()
        if self.items:
            self._select_index(0)
        else:
            messagebox.showinfo("No feedback", "No feedback events were found in this session.")

    def _build_widgets(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(1, weight=1)

        self.progress_var = tk.StringVar()
        ttk.Label(left, textvariable=self.progress_var).grid(row=0, column=0, sticky="w")
        self.item_list = tk.Listbox(left, width=44, exportselection=False)
        self.item_list.grid(row=1, column=0, sticky="ns")
        self.item_list.bind("<<ListboxSelect>>", self._on_list_select)

        button_bar = ttk.Frame(left)
        button_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_bar, text="上一条", command=self.prev_item).grid(row=0, column=0)
        ttk.Button(button_bar, text="下一条", command=self.next_item).grid(
            row=0,
            column=1,
            padx=4,
        )
        ttk.Button(button_bar, text="重新读取", command=self.reload_items).grid(
            row=0,
            column=2,
        )

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)

        self.header_var = tk.StringVar()
        ttk.Label(right, textvariable=self.header_var, font=("Segoe UI", 12, "bold")).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
        )

        evidence = ttk.Frame(right)
        evidence.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        evidence.columnconfigure(0, weight=1)
        evidence.rowconfigure(1, weight=1)

        ttk.Label(
            evidence,
            text="审核证据：先看摘要；只有拿不准时再看候选事件、轨迹或完整 JSON。",
            foreground="#7A4E00",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.evidence_tabs = ttk.Notebook(evidence)
        self.evidence_tabs.grid(row=1, column=0, sticky="nsew")

        summary_tab = ttk.Frame(self.evidence_tabs, padding=6)
        summary_tab.columnconfigure(0, weight=1)
        summary_tab.rowconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_tab, wrap="word")
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.configure(font=("Segoe UI", 10), state="disabled")
        self.evidence_tabs.add(summary_tab, text="审核摘要")

        candidate_tab = ttk.Frame(self.evidence_tabs, padding=6)
        candidate_tab.columnconfigure(0, weight=1)
        candidate_tab.rowconfigure(1, weight=1)
        ttk.Label(
            candidate_tab,
            text="双击候选，或选中后点击按钮，将其填入右侧 event 和时间窗口。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        candidate_columns = ("time", "event", "valence", "confidence", "subgoal")
        self.candidate_tree = ttk.Treeview(
            candidate_tab,
            columns=candidate_columns,
            show="headings",
            height=12,
        )
        candidate_headings = {
            "time": "时间",
            "event": "事件",
            "valence": "类型",
            "confidence": "置信度",
            "subgoal": "相关 subgoal",
        }
        candidate_widths = {
            "time": 80,
            "event": 260,
            "valence": 120,
            "confidence": 70,
            "subgoal": 120,
        }
        for column in candidate_columns:
            self.candidate_tree.heading(column, text=candidate_headings[column])
            self.candidate_tree.column(column, width=candidate_widths[column], anchor="w")
        self.candidate_tree.grid(row=1, column=0, sticky="nsew")
        self.candidate_tree.bind("<<TreeviewSelect>>", self._show_candidate_detail)
        self.candidate_tree.bind("<Double-1>", self._use_selected_candidate)
        ttk.Button(
            candidate_tab,
            text="采用选中候选",
            command=self._use_selected_candidate,
        ).grid(row=2, column=0, sticky="w", pady=4)
        self.candidate_detail_text = tk.Text(candidate_tab, height=9, wrap="word")
        self.candidate_detail_text.grid(row=3, column=0, sticky="ew")
        self.candidate_detail_text.configure(font=("Consolas", 9), state="disabled")
        self.evidence_tabs.add(candidate_tab, text="候选事件")

        trajectory_tab = ttk.Frame(self.evidence_tabs, padding=6)
        trajectory_tab.columnconfigure(0, weight=1)
        trajectory_tab.rowconfigure(1, weight=1)
        ttk.Label(
            trajectory_tab,
            text="每行是一帧紧凑轨迹；可打开地图回放确认人物、物品和动作顺序。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        trajectory_columns = (
            "t",
            "ai_action",
            "ai_subgoal",
            "ai_held",
            "human_action",
            "human_held",
            "pot",
        )
        self.trajectory_tree = ttk.Treeview(
            trajectory_tab,
            columns=trajectory_columns,
            show="headings",
            height=20,
        )
        trajectory_headings = {
            "t": "t",
            "ai_action": "AI 动作",
            "ai_subgoal": "AI subgoal",
            "ai_held": "AI 手持",
            "human_action": "人类动作",
            "human_held": "人类手持",
            "pot": "锅状态",
        }
        trajectory_widths = {
            "t": 45,
            "ai_action": 80,
            "ai_subgoal": 145,
            "ai_held": 75,
            "human_action": 80,
            "human_held": 75,
            "pot": 190,
        }
        for column in trajectory_columns:
            self.trajectory_tree.heading(column, text=trajectory_headings[column])
            self.trajectory_tree.column(column, width=trajectory_widths[column], anchor="w")
        self.trajectory_tree.grid(row=1, column=0, sticky="nsew")
        ttk.Button(
            trajectory_tab,
            text="▶ 在地图上回放当前评价时段",
            command=self._open_map_replay,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.evidence_tabs.add(trajectory_tab, text="近期轨迹")

        raw_tab = ttk.Frame(self.evidence_tabs, padding=6)
        raw_tab.columnconfigure(0, weight=1)
        raw_tab.rowconfigure(0, weight=1)
        self.raw_context_text = tk.Text(raw_tab, wrap="none")
        self.raw_context_text.grid(row=0, column=0, sticky="nsew")
        self.raw_context_text.configure(font=("Consolas", 9), state="disabled")
        self.evidence_tabs.add(raw_tab, text="完整 JSON")

        edit = ttk.Frame(right)
        edit.grid(row=1, column=1, sticky="nsew")
        edit.columnconfigure(1, weight=1)
        edit.rowconfigure(8, weight=1)
        edit.rowconfigure(9, weight=1)
        edit.rowconfigure(10, weight=1)

        self.decision_var = tk.StringVar()
        self.schema_action_var = tk.StringVar()
        self.use_for_training_var = tk.BooleanVar()
        self.time_window_var = tk.StringVar()
        self.event_var = tk.StringVar()
        self.candidate_subgoal_hint_var = tk.StringVar()

        ttk.Label(edit, text="最终判断（Decision）").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            edit,
            textvariable=self.decision_var,
            values=DECISION_OPTIONS,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="评价时间段 JSON").grid(row=1, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.time_window_var).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="认可的 event").grid(row=2, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.event_var).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(
            edit,
            textvariable=self.candidate_subgoal_hint_var,
            foreground="#5A5A5A",
            wraplength=520,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 2))

        ttk.Label(edit, text="Hu 相对排序").grid(row=4, column=0, sticky="nw")
        preference_frame = ttk.Frame(edit)
        preference_frame.grid(row=4, column=1, sticky="nsew", pady=2)
        preference_frame.columnconfigure(0, weight=1)
        preference_frame.columnconfigure(1, weight=1)

        ttk.Label(
            preference_frame,
            text="更希望 AI 选择（可多选）",
            foreground="#176B2C",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            preference_frame,
            text="更不希望 AI 选择（可多选）",
            foreground="#9B2C2C",
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        self.preferred_listbox = tk.Listbox(
            preference_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=len(HU_SUBGOALS),
        )
        self.rejected_listbox = tk.Listbox(
            preference_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=len(HU_SUBGOALS),
        )
        for subgoal in HU_SUBGOALS:
            label = f"{subgoal} | {SUBGOAL_DESCRIPTIONS.get(subgoal, '')}"
            self.preferred_listbox.insert(tk.END, label)
            self.rejected_listbox.insert(tk.END, label)
        self.preferred_listbox.grid(row=1, column=0, sticky="nsew")
        self.rejected_listbox.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        ttk.Label(
            preference_frame,
            text="直接点击可切换多个选项；两侧不能选择同一个 subgoal。",
            foreground="#5A5A5A",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        ttk.Checkbutton(
            edit,
            text="这条可用于 Hu 训练",
            variable=self.use_for_training_var,
        ).grid(row=5, column=1, sticky="w", pady=2)

        ttk.Label(edit, text="事件库处理").grid(row=6, column=0, sticky="w")
        ttk.Combobox(
            edit,
            textvariable=self.schema_action_var,
            values=SCHEMA_ACTION_OPTIONS,
            state="readonly",
        ).grid(row=6, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="认可的 conditions JSON").grid(row=7, column=0, sticky="nw")
        self.condition_text = tk.Text(edit, height=8, wrap="none")
        self.condition_text.grid(row=7, column=1, sticky="nsew", pady=2)
        self.condition_text.configure(font=("Consolas", 9))

        ttk.Label(edit, text="新事件定义 JSON").grid(row=8, column=0, sticky="nw")
        self.schema_update_text = tk.Text(edit, height=8, wrap="none")
        self.schema_update_text.grid(row=8, column=1, sticky="nsew", pady=2)
        self.schema_update_text.configure(font=("Consolas", 9))

        ttk.Label(edit, text="审核备注").grid(row=9, column=0, sticky="nw")
        self.notes_text = tk.Text(edit, height=6, wrap="word")
        self.notes_text.grid(row=9, column=1, sticky="nsew", pady=2)

        save_bar = ttk.Frame(edit)
        save_bar.grid(row=10, column=1, sticky="sew", pady=(8, 0))
        ttk.Button(save_bar, text="保存", command=self.save_current).grid(row=0, column=0)
        ttk.Button(save_bar, text="保存并下一条", command=self.save_and_next).grid(
            row=0,
            column=1,
            padx=4,
        )
        ttk.Button(save_bar, text="重新导出审核包", command=self.export_items).grid(
            row=0,
            column=2,
        )

    def _item_label(self, index: int, item: dict[str, Any]) -> str:
        reviewed = bool(item.get("existing_decision"))
        decision = item.get("existing_decision") or item.get("default_decision") or {}
        status = decision.get("decision") or "new"
        source = "REVIEWED" if reviewed else "AUTO"
        step = item.get("feedback_total_step")
        text = short_text(item.get("feedback_text"))
        return f"{index + 1:02d}. t={step} [{source}: {status}] {text}"

    def _refresh_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for index, item in enumerate(self.items):
            self.item_list.insert(tk.END, self._item_label(index, item))
        reviewed = sum(bool(item.get("existing_decision")) for item in self.items)
        self.progress_var.set(f"反馈审核：已保存 {reviewed}/{len(self.items)}")

    def _on_list_select(self, _event: tk.Event) -> None:
        selected = self.item_list.curselection()
        if selected:
            self._select_index(int(selected[0]))

    def _select_index(self, index: int) -> None:
        if not self.items:
            return
        self.current_index = max(0, min(index, len(self.items) - 1))
        self.item_list.selection_clear(0, tk.END)
        self.item_list.selection_set(self.current_index)
        self.item_list.see(self.current_index)
        self._load_item(self.items[self.current_index])

    def _load_item(self, item: dict[str, Any]) -> None:
        decision = item.get("existing_decision") or item.get("default_decision") or {}
        preference = decision.get("approved_preference") or {}

        self.header_var.set(
            f"Feedback {self.current_index + 1}/{len(self.items)} | "
            f"id={item.get('feedback_event_id')} | t={item.get('feedback_total_step')}"
        )
        self.decision_var.set(decision.get("decision") or "needs_revision")
        self.time_window_var.set(to_pretty_json(decision.get("approved_time_window")).replace("\n", " "))
        self.event_var.set(decision.get("approved_event") or "")
        set_subgoal_selection(
            self.preferred_listbox,
            preference.get("preferred_subgoals") or [],
        )
        set_subgoal_selection(
            self.rejected_listbox,
            preference.get("rejected_subgoals") or [],
        )
        observed_subgoals = trajectory_candidate_subgoals(item)
        self.candidate_subgoal_hint_var.set(
            "该归因时间段实际出现过的运行时候选："
            + format_list(observed_subgoals)
            + "。下面仍列出全部合法 Hu subgoal。"
        )
        self.use_for_training_var.set(bool(decision.get("use_for_hu_training")))
        self.schema_action_var.set(decision.get("schema_action") or "none")
        set_text(self.condition_text, to_pretty_json(decision.get("approved_condition_overrides") or {}))
        set_text(self.schema_update_text, to_pretty_json(decision.get("approved_schema_update")))
        set_text(self.notes_text, decision.get("review_notes") or "")
        set_text(self.summary_text, render_review_summary(item))
        self.summary_text.configure(state="disabled")
        set_text(self.raw_context_text, render_raw_context(item))
        self.raw_context_text.configure(state="disabled")
        self._load_candidate_table(item)
        self._load_trajectory_table(item)

    def _load_candidate_table(self, item: dict[str, Any]) -> None:
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        candidates = item.get("nearby_candidate_events") or []
        target_event = (item.get("attribution") or {}).get("target_event")
        target_item = None
        for index, event in enumerate(candidates):
            start = event.get("start_timestep")
            end = event.get("end_timestep")
            row_id = str(index)
            self.candidate_tree.insert(
                "",
                "end",
                iid=row_id,
                values=(
                    f"{start}..{end}",
                    event.get("event_type") or "-",
                    event.get("event_valence") or "-",
                    event.get("confidence", "-"),
                    event.get("related_subgoal") or "-",
                ),
            )
            if target_item is None and event.get("event_type") == target_event:
                target_item = row_id
        if target_item is not None:
            self.candidate_tree.selection_set(target_item)
            self.candidate_tree.see(target_item)
            self._show_candidate_detail()
        else:
            set_text(self.candidate_detail_text, "没有与自动 target_event 匹配的候选。")
            self.candidate_detail_text.configure(state="disabled")

    def _selected_candidate(self) -> dict[str, Any] | None:
        selected = self.candidate_tree.selection()
        if not selected:
            return None
        index = int(selected[0])
        candidates = self.current_item().get("nearby_candidate_events") or []
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    def _show_candidate_detail(self, _event: tk.Event | None = None) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            return
        evidence = candidate.get("evidence") or {}
        conditions = active_conditions(candidate.get("condition_features"))
        detail = [
            f"event: {candidate.get('event_type')}",
            f"actor / valence: {candidate.get('actor')} / {candidate.get('event_valence')}",
            f"time: {candidate.get('start_timestep')}..{candidate.get('end_timestep')}",
            f"reason: {evidence.get('reason') or '-'}",
            f"related_subgoal: {candidate.get('related_subgoal') or '-'}",
            f"alternative_subgoals: {format_list(candidate.get('alternative_subgoals'))}",
            "active conditions:",
        ]
        detail.extend(f"  {line}" for line in conditions or ["-"])
        set_text(self.candidate_detail_text, "\n".join(detail))
        self.candidate_detail_text.configure(state="disabled")

    def _use_selected_candidate(self, _event: tk.Event | None = None) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            messagebox.showinfo("未选择候选", "请先在表格中选择一个候选事件。")
            return
        self.event_var.set(candidate.get("event_type") or "")
        self.time_window_var.set(
            to_pretty_json(
                [candidate.get("start_timestep"), candidate.get("end_timestep")]
            ).replace("\n", " ")
        )
        self.decision_var.set("revise")

    def _load_trajectory_table(self, item: dict[str, Any]) -> None:
        self.trajectory_tree.delete(*self.trajectory_tree.get_children())
        for step in item.get("recent_steps") or []:
            self.trajectory_tree.insert(
                "",
                "end",
                values=(
                    step.get("total_step"),
                    step.get("ai_action") or "-",
                    step.get("ai_subgoal") or "-",
                    object_name(step.get("ai_held_object")),
                    step.get("human_action") or "-",
                    object_name(step.get("human_held_object")),
                    short_text(to_pretty_json(step.get("pot_states")), 45),
                ),
            )

    def _open_map_replay(self) -> None:
        item = self.current_item()
        try:
            time_window = parse_json_field(self.time_window_var.get(), None)
        except json.JSONDecodeError as exc:
            messagebox.showerror("时间段无效", f"评价时间段不是合法 JSON：{exc}")
            return
        steps = select_replay_steps(
            item.get("recent_steps") or [],
            time_window,
            context_steps=3,
        )
        if not steps:
            messagebox.showinfo(
                "没有可回放轨迹",
                "当前评价时段附近没有保存可用的轨迹帧。",
            )
            return
        ReplayWindow(
            self.root,
            steps=steps,
            time_window=time_window,
            feedback_step=item.get("feedback_total_step"),
            feedback_text=item.get("feedback_text") or "",
            candidate_events=item.get("nearby_candidate_events") or [],
        )

    def current_item(self) -> dict[str, Any]:
        return self.items[self.current_index]

    def build_current_decision(self) -> dict[str, Any]:
        item = self.current_item()
        default = item.get("default_decision") or {}
        try:
            time_window = parse_json_field(self.time_window_var.get(), None)
            condition_overrides = parse_json_field(read_text(self.condition_text), {})
            schema_update = parse_json_field(read_text(self.schema_update_text), None)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON field is invalid: {exc}") from exc

        preferred_subgoals = selected_subgoals(self.preferred_listbox)
        rejected_subgoals = selected_subgoals(self.rejected_listbox)
        overlap = sorted(set(preferred_subgoals) & set(rejected_subgoals))
        if overlap:
            raise ValueError(
                "同一个 subgoal 不能同时出现在“更希望”和“更不希望”中："
                + ", ".join(overlap)
            )
        labeled_subgoals = {*preferred_subgoals, *rejected_subgoals}
        coordination_labels = labeled_subgoals & set(COORDINATION_SUBGOALS)
        task_labels = labeled_subgoals - set(COORDINATION_SUBGOALS)
        if coordination_labels and task_labels:
            raise ValueError(
                "一条 Hu pairwise 标签不能混合 Task 与 Coordination 决策域。"
                f" Task={sorted(task_labels)}；"
                f" Coordination={sorted(coordination_labels)}。"
            )

        decision_name = self.decision_var.get() or "needs_revision"
        use_for_training = bool(self.use_for_training_var.get())
        approved_event = self.event_var.get().strip() or None
        schema_action = self.schema_action_var.get() or "none"
        if schema_action in {"accept_new_event", "revise_new_event"}:
            if not isinstance(schema_update, dict) or not schema_update:
                raise ValueError(
                    f"事件库处理={schema_action} 时，必须填写完整的新事件定义 JSON。"
                )
        if schema_action == "map_to_existing" and not approved_event:
            raise ValueError(
                "事件库处理=map_to_existing 时，必须填写要映射到的 approved event。"
            )
        if schema_action == "add_condition_only":
            if not isinstance(condition_overrides, dict) or not condition_overrides:
                raise ValueError(
                    "事件库处理=add_condition_only 时，必须填写至少一个认可的 condition。"
                )
        if use_for_training:
            if decision_name in NON_TRAINING_DECISIONS:
                raise ValueError(
                    f"Decision={decision_name} 不能同时勾选“用于 Hu 训练”。"
                )
            if not approved_event:
                raise ValueError("用于 Hu 训练时必须确认 approved event。")
            if not preferred_subgoals or not rejected_subgoals:
                raise ValueError(
                    "用于 Hu 训练时，必须至少选择一个 preferred subgoal "
                    "和一个 rejected subgoal。"
                )
            if not isinstance(time_window, list) or len(time_window) != 2:
                raise ValueError("用于 Hu 训练时，时间段必须是 [start, end]。")
            if not isinstance(condition_overrides, dict):
                raise ValueError("Conditions 必须是一个 JSON object。")

        return {
            "record_type": "review_decision",
            "feedback_event_id": item.get("feedback_event_id"),
            "feedback_total_step": item.get("feedback_total_step"),
            "feedback_text": item.get("feedback_text"),
            "decision": decision_name,
            "approved_time_window": time_window,
            "approved_event": approved_event,
            "approved_condition_overrides": condition_overrides,
            "approved_preference": {
                "preferred_subgoals": preferred_subgoals,
                "rejected_subgoals": rejected_subgoals,
            },
            "use_for_hu_training": use_for_training,
            "schema_action": schema_action,
            "approved_schema_update": schema_update,
            "review_notes": read_text(self.notes_text),
            "default_decision_snapshot": default,
        }

    def save_current(self) -> bool:
        try:
            decision = self.build_current_decision()
        except ValueError as exc:
            messagebox.showerror("Cannot save", str(exc))
            return False
        upsert_review_decision(self.session_dir, decision)
        self.items[self.current_index]["existing_decision"] = decision
        self._refresh_list()
        self._select_index(self.current_index)
        return True

    def save_and_next(self) -> None:
        if self.save_current():
            self.next_item()

    def prev_item(self) -> None:
        self._select_index(self.current_index - 1)

    def next_item(self) -> None:
        self._select_index(self.current_index + 1)

    def reload_items(self) -> None:
        self.items = build_review_items(self.session_dir, lookback_steps=self.lookback_steps)
        self._refresh_list()
        if self.items:
            self._select_index(min(self.current_index, len(self.items) - 1))

    def export_items(self) -> None:
        counts = export_review_items(self.session_dir, lookback_steps=self.lookback_steps)
        messagebox.showinfo(
            "Exported",
            f"review_items={counts['review_items']}, existing_decisions={counts['existing_decisions']}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review feedback attribution for one session.")
    parser.add_argument("--session", required=True, type=Path, help="Session directory.")
    parser.add_argument("--lookback-steps", type=int, default=50)
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only write review_items.jsonl; do not open the GUI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.session
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    counts = export_review_items(session_dir, lookback_steps=args.lookback_steps)
    print(
        f"Exported {counts['review_items']} review items "
        f"({counts['existing_decisions']} existing decisions)."
    )

    if args.export_only:
        return 0

    root = tk.Tk()
    ReviewApp(root, session_dir=session_dir, lookback_steps=args.lookback_steps)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
