"""Export recovery examples into a compact human-review packet.

The recovery dataset can contain hundreds of near-duplicate timesteps.  This
tool samples a smaller, diverse set that is easier to review manually.  The
reviewer can then decide whether the scaffold label is reasonable, or write a
better correction.  Those reviewed labels can later become higher-quality
recovery data for PPO/Hu experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=60)
    parser.add_argument(
        "--per-episode",
        type=int,
        default=6,
        help="Maximum examples exported from one episode.",
    )
    parser.add_argument(
        "--min-step-gap",
        type=int,
        default=8,
        help="Skip examples too close to an already selected timestep in the same episode.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def action_text(action) -> str:
    if action == "interact":
        return "interact"
    if isinstance(action, list):
        mapping = {
            (0, -1): "north",
            (0, 1): "south",
            (1, 0): "east",
            (-1, 0): "west",
            (0, 0): "stay",
        }
        return mapping.get(tuple(action), str(action))
    return str(action)


def held_text(player: dict) -> str:
    return player.get("held_object") or "empty"


def summarize_state(state: dict) -> str:
    players = state.get("players", [])
    objects = state.get("objects", [])
    p0 = players[0] if players else {}
    p1 = players[1] if len(players) > 1 else {}
    object_bits = []
    for obj in objects[:4]:
        bit = f"{obj.get('name')}@{obj.get('position')}"
        if obj.get("is_ready"):
            bit += ":ready"
        elif obj.get("is_cooking"):
            bit += ":cooking"
        object_bits.append(bit)
    return (
        f"AI pos={p0.get('position')} facing={p0.get('orientation')} "
        f"holding={held_text(p0)}; partner pos={p1.get('position')} "
        f"holding={held_text(p1)}; objects={', '.join(object_bits)}"
    )


def select_records(records: list[dict], max_examples: int, per_episode: int, min_step_gap: int) -> list[dict]:
    by_episode: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (int(record["episode_index"]), int(record["seed"]))
        by_episode[key].append(record)

    selected: list[dict] = []
    # Prefer early failures first, then spread across later timesteps.
    for key in sorted(by_episode, key=lambda item: (item[1], item[0])):
        episode_records = sorted(by_episode[key], key=lambda rec: int(rec["step"]))
        chosen_steps: list[int] = []
        action_counter: Counter[str] = Counter()
        for record in episode_records:
            if len(chosen_steps) >= per_episode:
                break
            step = int(record["step"])
            if any(abs(step - chosen) < min_step_gap for chosen in chosen_steps):
                continue
            teacher_action = action_text(record["teacher_action0"])
            # Avoid one episode being filled with the same correction.
            if action_counter[teacher_action] >= 2:
                continue
            chosen_steps.append(step)
            action_counter[teacher_action] += 1
            selected.append(record)
            if len(selected) >= max_examples:
                return selected
    return selected


def make_review_record(index: int, record: dict) -> dict:
    return {
        "review_id": f"R{index:03d}",
        "episode_index": record["episode_index"],
        "seed": record["seed"],
        "step": record["step"],
        "episode_reward": record["episode_reward"],
        "student_action0": action_text(record["student_action0"]),
        "student_action1": action_text(record["student_action1"]),
        "scaffold_action0": action_text(record["teacher_action0"]),
        "state_summary": summarize_state(record["state"]),
        "human_review": {
            "is_scaffold_label_reasonable": None,
            "preferred_ai_action": None,
            "what_should_ai_do": "",
            "event_label": "",
            "condition_notes": "",
        },
    }


def write_markdown(records: list[dict], output_path: Path) -> None:
    lines = [
        "# Recovery Review Packet",
        "",
        "Fill the `human_review` fields in the JSONL, or use this file as a reading guide.",
        "",
        "Suggested labels:",
        "- `is_scaffold_label_reasonable`: true / false / unsure",
        "- `preferred_ai_action`: north / south / east / west / stay / interact",
        "- `what_should_ai_do`: short natural-language correction",
        "- `event_label`: e.g. missed_onion_pickup, wrong_direction, repeated_interact, blocking, should_wait",
        "- `condition_notes`: key context, e.g. AI empty-handed, onion available, pot has two tomatoes",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['review_id']} seed={record['seed']} step={record['step']}",
                "",
                f"- episode reward: `{record['episode_reward']}`",
                f"- student AI action: `{record['student_action0']}`",
                f"- partner action: `{record['student_action1']}`",
                f"- scaffold suggested action: `{record['scaffold_action0']}`",
                f"- state: {record['state_summary']}",
                "",
                "Your judgment:",
                "- reasonable? ",
                "- preferred action: ",
                "- what should AI do: ",
                "- event label: ",
                "- condition notes: ",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.dataset_jsonl)
    if not records:
        raise RuntimeError(f"No records found: {args.dataset_jsonl}")
    selected = select_records(
        records,
        args.max_examples,
        args.per_episode,
        args.min_step_gap,
    )
    review_records = [
        make_review_record(index + 1, record)
        for index, record in enumerate(selected)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "human_review_candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in review_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    md_path = args.output_dir / "human_review_packet.md"
    write_markdown(review_records, md_path)
    summary = {
        "source": str(args.dataset_jsonl),
        "total_source_records": len(records),
        "exported_records": len(review_records),
        "output_jsonl": str(jsonl_path),
        "output_markdown": str(md_path),
        "scaffold_action_counts": dict(
            Counter(record["scaffold_action0"] for record in review_records)
        ),
    }
    (args.output_dir / "review_packet_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
