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
    parse_csv_field,
    parse_json_field,
    upsert_review_decision,
)


DECISION_OPTIONS = [
    "accept",
    "revise",
    "reject",
    "provenance_only",
    "map_to_existing",
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


class ReviewApp:
    def __init__(self, root: tk.Tk, session_dir: Path, lookback_steps: int) -> None:
        self.root = root
        self.session_dir = session_dir
        self.lookback_steps = lookback_steps
        self.items = build_review_items(session_dir, lookback_steps=lookback_steps)
        self.current_index = 0

        self.root.title(f"DURF attribution review - {session_dir.name}")
        self.root.geometry("1280x820")

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

        ttk.Label(left, text="Feedback items").grid(row=0, column=0, sticky="w")
        self.item_list = tk.Listbox(left, width=44, exportselection=False)
        self.item_list.grid(row=1, column=0, sticky="ns")
        self.item_list.bind("<<ListboxSelect>>", self._on_list_select)

        button_bar = ttk.Frame(left)
        button_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_bar, text="Prev", command=self.prev_item).grid(row=0, column=0)
        ttk.Button(button_bar, text="Next", command=self.next_item).grid(row=0, column=1, padx=4)
        ttk.Button(button_bar, text="Reload", command=self.reload_items).grid(row=0, column=2)

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

        self.context_text = tk.Text(right, wrap="word", height=26)
        self.context_text.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.context_text.configure(font=("Consolas", 10))

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
        self.preferred_var = tk.StringVar()
        self.rejected_var = tk.StringVar()

        ttk.Label(edit, text="Decision").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            edit,
            textvariable=self.decision_var,
            values=DECISION_OPTIONS,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="Time window JSON").grid(row=1, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.time_window_var).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="Approved event").grid(row=2, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.event_var).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="Preferred subgoals").grid(row=3, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.preferred_var).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="Rejected subgoals").grid(row=4, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.rejected_var).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(
            edit,
            text="Use for Hu training",
            variable=self.use_for_training_var,
        ).grid(row=5, column=1, sticky="w", pady=2)

        ttk.Label(edit, text="Schema action").grid(row=6, column=0, sticky="w")
        ttk.Combobox(
            edit,
            textvariable=self.schema_action_var,
            values=SCHEMA_ACTION_OPTIONS,
            state="readonly",
        ).grid(row=6, column=1, sticky="ew", pady=2)

        ttk.Label(edit, text="Condition overrides JSON").grid(row=7, column=0, sticky="nw")
        self.condition_text = tk.Text(edit, height=8, wrap="none")
        self.condition_text.grid(row=7, column=1, sticky="nsew", pady=2)
        self.condition_text.configure(font=("Consolas", 9))

        ttk.Label(edit, text="Schema update JSON").grid(row=8, column=0, sticky="nw")
        self.schema_update_text = tk.Text(edit, height=8, wrap="none")
        self.schema_update_text.grid(row=8, column=1, sticky="nsew", pady=2)
        self.schema_update_text.configure(font=("Consolas", 9))

        ttk.Label(edit, text="Review notes").grid(row=9, column=0, sticky="nw")
        self.notes_text = tk.Text(edit, height=6, wrap="word")
        self.notes_text.grid(row=9, column=1, sticky="nsew", pady=2)

        save_bar = ttk.Frame(edit)
        save_bar.grid(row=10, column=1, sticky="sew", pady=(8, 0))
        ttk.Button(save_bar, text="Save", command=self.save_current).grid(row=0, column=0)
        ttk.Button(save_bar, text="Save + Next", command=self.save_and_next).grid(
            row=0,
            column=1,
            padx=4,
        )
        ttk.Button(save_bar, text="Export items", command=self.export_items).grid(row=0, column=2)

    def _item_label(self, index: int, item: dict[str, Any]) -> str:
        decision = item.get("existing_decision") or item.get("default_decision") or {}
        status = decision.get("decision") or "new"
        step = item.get("feedback_total_step")
        text = short_text(item.get("feedback_text"))
        return f"{index + 1:02d}. t={step} [{status}] {text}"

    def _refresh_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for index, item in enumerate(self.items):
            self.item_list.insert(tk.END, self._item_label(index, item))

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
        self.preferred_var.set(", ".join(preference.get("preferred_subgoals") or []))
        self.rejected_var.set(", ".join(preference.get("rejected_subgoals") or []))
        self.use_for_training_var.set(bool(decision.get("use_for_hu_training")))
        self.schema_action_var.set(decision.get("schema_action") or "none")
        set_text(self.condition_text, to_pretty_json(decision.get("approved_condition_overrides") or {}))
        set_text(self.schema_update_text, to_pretty_json(decision.get("approved_schema_update")))
        set_text(self.notes_text, decision.get("review_notes") or "")
        set_text(self.context_text, self._render_context(item))

    def _render_context(self, item: dict[str, Any]) -> str:
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

        return {
            "record_type": "review_decision",
            "feedback_event_id": item.get("feedback_event_id"),
            "feedback_total_step": item.get("feedback_total_step"),
            "feedback_text": item.get("feedback_text"),
            "decision": self.decision_var.get() or "needs_revision",
            "approved_time_window": time_window,
            "approved_event": self.event_var.get().strip() or None,
            "approved_condition_overrides": condition_overrides,
            "approved_preference": {
                "preferred_subgoals": parse_csv_field(self.preferred_var.get()),
                "rejected_subgoals": parse_csv_field(self.rejected_var.get()),
            },
            "use_for_hu_training": bool(self.use_for_training_var.get()),
            "schema_action": self.schema_action_var.get() or "none",
            "approved_schema_update": schema_update,
            "review_notes": read_text(self.notes_text),
            "default_decision_snapshot": default,
        }

    def save_current(self) -> None:
        try:
            decision = self.build_current_decision()
        except ValueError as exc:
            messagebox.showerror("Cannot save", str(exc))
            return
        upsert_review_decision(self.session_dir, decision)
        self.items[self.current_index]["existing_decision"] = decision
        self._refresh_list()
        self._select_index(self.current_index)

    def save_and_next(self) -> None:
        self.save_current()
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
