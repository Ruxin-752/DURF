"""Keyboard feedback listener for TAMER-style training.

Keys:
    J     -> +1 positive feedback
    K     -> -1 negative feedback
    Space ->  0 neutral feedback
    Q     -> quit

The latest feedback is written to reward_signal.txt, and all feedback events
are appended to feedback_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


KEY_TO_FEEDBACK = {
    "j": "+1",
    "k": "-1",
    "space": "0",
}


def atomic_write(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def append_log(path: Path, key: str, feedback: str, start_time: float) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "key", "feedback"])
        writer.writerow([f"{time.time() - start_time:.3f}", key, feedback])


def record_feedback(signal_path: Path, log_path: Path, key: str, start_time: float) -> None:
    feedback = KEY_TO_FEEDBACK[key]
    atomic_write(signal_path, feedback + "\n")
    append_log(log_path, key, feedback, start_time)
    print(f"{key!r} -> {feedback}")


def run_with_pynput(signal_path: Path, log_path: Path) -> None:
    from pynput import keyboard

    start_time = time.time()
    atomic_write(signal_path, "0\n")

    def on_press(key: keyboard.Key | keyboard.KeyCode):
        try:
            key_name = key.char.lower() if key.char else ""
        except AttributeError:
            key_name = "space" if key == keyboard.Key.space else ""

        if key_name in KEY_TO_FEEDBACK:
            record_feedback(signal_path, log_path, key_name, start_time)
        elif key_name == "q":
            print("quit")
            return False

        return None

    print("Listening globally: J=+1, K=-1, Space=0, Q=quit")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def run_with_msvcrt(signal_path: Path, log_path: Path) -> None:
    import msvcrt

    start_time = time.time()
    atomic_write(signal_path, "0\n")
    print("Listening in this terminal: J=+1, K=-1, Space=0, Q=quit")
    print("Tip: install pynput for global listening while another window is focused.")

    while True:
        ch = msvcrt.getwch()
        key_name = "space" if ch == " " else ch.lower()
        if key_name in KEY_TO_FEEDBACK:
            record_feedback(signal_path, log_path, key_name, start_time)
        elif key_name == "q":
            print("quit")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for human feedback keys.")
    parser.add_argument(
        "--signal-file",
        default="reward_signal.txt",
        help="Path to write the latest feedback signal.",
    )
    parser.add_argument(
        "--log-file",
        default="feedback_log.csv",
        help="Path to append timestamped feedback events.",
    )
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Use terminal-only input instead of global keyboard listening.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal_path = Path(args.signal_file).resolve()
    log_path = Path(args.log_file).resolve()
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.terminal_only:
        run_with_msvcrt(signal_path, log_path)
        return 0

    try:
        run_with_pynput(signal_path, log_path)
    except ModuleNotFoundError:
        print("pynput is not installed; falling back to terminal-only listening.")
        run_with_msvcrt(signal_path, log_path)
    except KeyboardInterrupt:
        print("quit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
