"""Split Hu_general training data into train/val/test (70/15/15).

Splits each domain (task, coordination) independently so the rare
coordination samples don't disappear into one split.
"""
import json
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--coord-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_val = []
    all_test = []

    for name, path in [("task", args.task_file), ("coordination", args.coord_file)]:
        lines = [l.strip() for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
        n = len(lines)
        order = rng.permutation(n)

        # Ensure at least 1 sample in each split, up to domain limits
        n_train = max(1, min(int(n * args.train_frac), n - 2))
        n_val = max(1, min(int(n * args.val_frac), n - n_train - 1))
        n_test = n - n_train - n_val

        train_idx = set(int(i) for i in order[:n_train])
        val_idx = set(int(i) for i in order[n_train:n_train + n_val])

        train_lines = [lines[i] for i in range(n) if i in train_idx]
        val_lines = [lines[i] for i in range(n) if i in val_idx]
        test_lines = [lines[i] for i in range(n) if i not in train_idx and i not in val_idx]

        print(f"{name}: {n} total -> train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)}")

        all_train.extend(train_lines)
        all_val.extend(val_lines)
        all_test.extend(test_lines)

    # Write combined files
    for label, lines in [("train", all_train), ("val", all_val), ("test", all_test)]:
        p = args.output_dir / f"hu_general_{label}.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} samples to {p}")

    print(f"\nTotal: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")


if __name__ == "__main__":
    main()
