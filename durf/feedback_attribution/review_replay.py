"""Map replay window for post-session attribution review."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


TERRAIN_STYLE = {
    " ": ("#F1E4C8", ""),
    "X": ("#A77A45", "台"),
    "P": ("#5F6670", "锅"),
    "D": ("#E8E8E8", "盘"),
    "T": ("#D95C4F", "番"),
    "O": ("#D7A43B", "洋"),
    "S": ("#85B98E", "出"),
}

OBJECT_STYLE = {
    "tomato": ("#D6453D", "番"),
    "onion": ("#E3B341", "洋"),
    "dish": ("#F6F6F6", "盘"),
    "soup": ("#EE8C3A", "汤"),
}

SPEED_OPTIONS = {
    "慢速 1.0 秒/步": 1000,
    "正常 0.5 秒/步": 500,
    "快速 0.25 秒/步": 250,
}


def select_replay_steps(
    steps: list[dict[str, Any]],
    time_window: list[int] | None,
    *,
    context_steps: int = 3,
) -> list[dict[str, Any]]:
    """Select ordered replay frames around an attribution time window."""
    ordered = sorted(
        steps,
        key=lambda step: int(step.get("total_step") or -1),
    )
    if not isinstance(time_window, list) or len(time_window) != 2:
        return ordered
    try:
        start, end = (int(time_window[0]), int(time_window[1]))
    except (TypeError, ValueError):
        return ordered
    if start > end:
        start, end = end, start
    lower = start - max(0, context_steps)
    upper = end + max(0, context_steps)
    return [
        step
        for step in ordered
        if lower <= int(step.get("total_step") or -1) <= upper
    ]


def event_names_at_step(
    candidate_events: list[dict[str, Any]],
    total_step: int,
) -> list[str]:
    """Return candidate event names whose detected span covers this frame."""
    names = []
    for event in candidate_events:
        try:
            start = int(event.get("start_timestep"))
            end = int(event.get("end_timestep"))
        except (TypeError, ValueError):
            continue
        if start <= total_step <= end:
            name = str(event.get("event_type") or "")
            if name and name not in names:
                names.append(name)
    return names


def object_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("type") or "-")
    return "-" if value is None else str(value)


class ReplayWindow:
    """Interactive Tk map replay for one feedback item."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        steps: list[dict[str, Any]],
        time_window: list[int] | None,
        feedback_step: int | None,
        feedback_text: str,
        candidate_events: list[dict[str, Any]],
    ) -> None:
        self.steps = steps
        self.time_window = self._normalize_window(time_window)
        self.feedback_step = feedback_step
        self.feedback_text = feedback_text
        self.candidate_events = candidate_events
        self.index = 0
        self.playing = False
        self.after_id: str | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("DURF 近期轨迹地图回放")
        self.window.geometry("1000x760")
        self.window.minsize(760, 600)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.frame_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.event_var = tk.StringVar()
        self.speed_var = tk.StringVar(value="正常 0.5 秒/步")
        self.play_var = tk.StringVar(value="播放")

        self._build_widgets()
        self.render()

    @staticmethod
    def _normalize_window(value: list[int] | None) -> tuple[int, int] | None:
        if not isinstance(value, list) or len(value) != 2:
            return None
        try:
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
        return (min(start, end), max(start, end))

    def _build_widgets(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        ttk.Label(
            root,
            text=f'用户反馈：“{self.feedback_text}”',
            font=("Segoe UI", 12, "bold"),
            wraplength=940,
        ).grid(row=0, column=0, sticky="w")

        info = ttk.Frame(root)
        info.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        info.columnconfigure(0, weight=1)
        ttk.Label(info, textvariable=self.frame_var, foreground="#7A4E00").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(info, textvariable=self.detail_var).grid(
            row=1,
            column=0,
            sticky="w",
        )
        ttk.Label(
            info,
            textvariable=self.event_var,
            foreground="#8B2E2E",
            wraplength=940,
        ).grid(row=2, column=0, sticky="w")

        canvas_frame = ttk.Frame(root)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#202328",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(root)
        controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(4, weight=1)
        ttk.Button(controls, text="|<", width=4, command=self.first).grid(row=0, column=0)
        ttk.Button(controls, text="< 上一步", command=self.previous).grid(
            row=0,
            column=1,
            padx=(4, 0),
        )
        ttk.Button(controls, textvariable=self.play_var, command=self.toggle_play).grid(
            row=0,
            column=2,
            padx=4,
        )
        ttk.Button(controls, text="下一步 >", command=self.next).grid(row=0, column=3)
        self.timeline = ttk.Scale(
            controls,
            from_=0,
            to=max(0, len(self.steps) - 1),
            command=self._seek,
        )
        self.timeline.grid(row=0, column=4, sticky="ew", padx=8)
        ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=list(SPEED_OPTIONS),
            state="readonly",
            width=18,
        ).grid(row=0, column=5)

        ttk.Label(
            root,
            text=(
                "蓝色=AI，绿色=人类；红框=当前帧位于归因时间段；"
                "本帧地图是执行所列动作后的状态。"
            ),
            foreground="#5A5A5A",
        ).grid(row=4, column=0, sticky="w", pady=(6, 0))

    def close(self) -> None:
        self.pause()
        self.window.destroy()

    def first(self) -> None:
        self.pause()
        self.index = 0
        self.render()

    def previous(self) -> None:
        self.pause()
        self.index = max(0, self.index - 1)
        self.render()

    def next(self) -> None:
        self.pause()
        self.index = min(len(self.steps) - 1, self.index + 1)
        self.render()

    def _seek(self, value: str) -> None:
        new_index = max(0, min(len(self.steps) - 1, int(float(value))))
        if new_index != self.index:
            self.pause()
            self.index = new_index
            self.render(update_scale=False)

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
            return
        if self.index >= len(self.steps) - 1:
            self.index = 0
        self.playing = True
        self.play_var.set("暂停")
        self.render()
        self._schedule_next()

    def pause(self) -> None:
        self.playing = False
        self.play_var.set("播放")
        if self.after_id is not None:
            self.window.after_cancel(self.after_id)
            self.after_id = None

    def _schedule_next(self) -> None:
        if not self.playing:
            return
        delay = SPEED_OPTIONS.get(self.speed_var.get(), 500)
        self.after_id = self.window.after(delay, self._advance)

    def _advance(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        if self.index >= len(self.steps) - 1:
            self.pause()
            return
        self.index += 1
        self.render()
        self._schedule_next()

    def render(self, *, update_scale: bool = True) -> None:
        if not self.steps:
            return
        step = self.steps[self.index]
        facts = step.get("state_facts") or {}
        total_step = int(step.get("total_step") or 0)
        inside_window = bool(
            self.time_window
            and self.time_window[0] <= total_step <= self.time_window[1]
        )
        feedback_marker = " | 反馈发生" if total_step == self.feedback_step else ""
        window_marker = " | 归因窗口内" if inside_window else ""
        self.frame_var.set(
            f"帧 {self.index + 1}/{len(self.steps)} | t={total_step}"
            f"{window_marker}{feedback_marker}"
        )
        self.detail_var.set(
            "AI: "
            f"{step.get('ai_action') or '-'} / {step.get('ai_subgoal') or '-'} / "
            f"手持 {object_name(step.get('ai_held_object'))}    "
            "人类: "
            f"{step.get('human_action') or '-'} / "
            f"手持 {object_name(step.get('human_held_object'))}"
        )
        active_events = event_names_at_step(self.candidate_events, total_step)
        self.event_var.set(
            "本帧候选事件：" + (", ".join(active_events) if active_events else "无")
        )
        if update_scale:
            self.timeline.set(self.index)
        self._draw_map(facts, inside_window=inside_window)

    def _draw_map(self, facts: dict[str, Any], *, inside_window: bool) -> None:
        self.canvas.delete("all")
        terrain = ((facts.get("layout_features") or {}).get("terrain") or [])
        if not terrain:
            self.canvas.create_text(
                30,
                30,
                anchor="nw",
                fill="white",
                text="该帧没有保存 terrain，无法绘制地图。",
                font=("Segoe UI", 12),
            )
            return

        rows = len(terrain)
        cols = max(len(row) for row in terrain)
        width = max(self.canvas.winfo_width(), 700)
        height = max(self.canvas.winfo_height(), 440)
        tile = max(38, min(72, (width - 80) // cols, (height - 60) // rows))
        origin_x = (width - cols * tile) // 2
        origin_y = (height - rows * tile) // 2
        border = "#E24A3B" if inside_window else "#D8C69D"

        self.canvas.create_rectangle(
            origin_x - 5,
            origin_y - 5,
            origin_x + cols * tile + 5,
            origin_y + rows * tile + 5,
            outline=border,
            width=5 if inside_window else 2,
        )

        for y, row in enumerate(terrain):
            for x, terrain_code in enumerate(row):
                fill, label = TERRAIN_STYLE.get(terrain_code, ("#C7B38B", terrain_code))
                x0 = origin_x + x * tile
                y0 = origin_y + y * tile
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + tile,
                    y0 + tile,
                    fill=fill,
                    outline="#6E5C43",
                )
                if label:
                    self.canvas.create_text(
                        x0 + tile / 2,
                        y0 + tile / 2,
                        text=label,
                        fill="#252525",
                        font=("Microsoft YaHei UI", max(10, tile // 4), "bold"),
                    )

        for record in facts.get("objects") or []:
            position = record.get("position")
            obj = record.get("object") or {}
            self._draw_object(origin_x, origin_y, tile, position, obj)

        players = facts.get("players") or []
        if not players:
            players = [
                {
                    "position": facts.get("ai_pos"),
                    "orientation": None,
                    "held_object": facts.get("ai_held_object"),
                },
                {
                    "position": facts.get("human_pos"),
                    "orientation": None,
                    "held_object": facts.get("human_held_object"),
                },
            ]
        for index, player in enumerate(players[:2]):
            self._draw_player(origin_x, origin_y, tile, index, player)

    def _draw_object(
        self,
        origin_x: int,
        origin_y: int,
        tile: int,
        position: Any,
        obj: dict[str, Any],
    ) -> None:
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return
        x, y = int(position[0]), int(position[1])
        center_x = origin_x + x * tile + tile / 2
        center_y = origin_y + y * tile + tile / 2
        name = object_name(obj)
        color, label = OBJECT_STYLE.get(name, ("#B56EDC", name[:1].upper()))
        radius = max(9, tile // 6)
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="#303030",
            width=2,
        )
        self.canvas.create_text(
            center_x,
            center_y,
            text=label,
            fill="#202020",
            font=("Microsoft YaHei UI", max(8, tile // 6), "bold"),
        )

    def _draw_player(
        self,
        origin_x: int,
        origin_y: int,
        tile: int,
        index: int,
        player: dict[str, Any],
    ) -> None:
        position = player.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return
        x, y = int(position[0]), int(position[1])
        center_x = origin_x + x * tile + tile / 2
        center_y = origin_y + y * tile + tile / 2
        radius = max(14, tile // 3)
        color = "#3182CE" if index == 0 else "#2FA66D"
        label = "AI" if index == 0 else "人"
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="white",
            width=3,
        )
        self.canvas.create_text(
            center_x,
            center_y,
            text=label,
            fill="white",
            font=("Microsoft YaHei UI", max(10, tile // 5), "bold"),
        )

        orientation = player.get("orientation")
        if isinstance(orientation, (list, tuple)) and len(orientation) == 2:
            dx, dy = float(orientation[0]), float(orientation[1])
            self.canvas.create_line(
                center_x,
                center_y,
                center_x + dx * radius * 1.45,
                center_y + dy * radius * 1.45,
                fill="#FFF176",
                width=3,
                arrow=tk.LAST,
            )

        held = object_name(player.get("held_object"))
        if held != "-":
            self.canvas.create_text(
                center_x,
                center_y - radius - 10,
                text=f"持:{held}",
                fill="#111111",
                font=("Microsoft YaHei UI", max(8, tile // 7), "bold"),
            )
