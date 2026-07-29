"""Route 2: neural feedback-to-reward inference network (structure only).

This mirrors the paper's ``TrajectoryFeedbackRewardPredictor`` /
``aaai_inference_network_training.ipynb`` architecture:

    EmbeddingBag(vocab, 30)  ->  concat 15-dim trajectory feature counts
    -> Linear(30 + n_features, 128) -> ReLU -> Linear(128, n_features)

with two Overcooked adaptations:

- the trajectory feature-count input and the regression output use the shared
  53-dim Overcooked feature schema (the paper used 15 inputs / 9 conjunction
  reward outputs);
- there is deliberately **no training loop here** -- this module only provides
  the model and the (torch-free) dataset helpers used to assemble inputs. The
  ``train_route2_inference_network.py`` script stops before any optimizer step.

We provide this scaffold to reach the paper's "just before Model Training"
state; actually fitting it needs a large human teacher-learner corpus we do not
yet have (see DIFFERENCES_FROM_PAPER.md).
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
from torch import nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EMBEDDING_DIM = 30
HIDDEN_DIM = 128


class TrajectoryFeedbackRewardPredictor(nn.Module):
    """EmbeddingBag + trajectory-feature MLP, output dim = n_features."""

    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        *,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        tokens: torch.Tensor,
        offsets: torch.Tensor,
        feature_counts: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(tokens, offsets)
        combined = torch.cat([embedded, feature_counts], dim=1)
        hidden = self.relu(self.fc1(combined))
        return self.fc2(hidden)


def build_vocab(
    token_lists: list[list[str]],
    *,
    min_freq: int = 1,
) -> dict[str, int]:
    """Word -> index vocabulary with reserved pad/unk, mirroring the paper."""

    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(token for token in tokens if token)

    itos = [PAD_TOKEN, UNK_TOKEN]
    itos.extend(
        word
        for word, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_freq and word not in (PAD_TOKEN, UNK_TOKEN)
    )
    return {word: index for index, word in enumerate(itos)}


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab.get(UNK_TOKEN, 0)
    encoded = [vocab.get(token, unk) for token in tokens if token != ""]
    return encoded or [unk]


def make_folds(
    group_ids: list[str | None],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> list[dict[str, list[int]]]:
    """Grouped CV folds (held-out analog of teachers / reward configs).

    Examples are grouped by ``group_id`` (probe id here). Whole groups are held
    out per fold. Examples with ``group_id is None`` (ungrouped seeds) always go
    into the training split.
    """

    grouped: dict[str, list[int]] = {}
    ungrouped: list[int] = []
    for index, group in enumerate(group_ids):
        if group is None:
            ungrouped.append(index)
        else:
            grouped.setdefault(group, []).append(index)

    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    n_folds = max(1, min(n_folds, len(groups))) if groups else 1

    folds = []
    for fold in range(n_folds):
        test_groups = groups[fold::n_folds]
        test_indices = [idx for group in test_groups for idx in grouped[group]]
        test_set = set(test_indices)
        train_indices = [
            index for index in range(len(group_ids)) if index not in test_set
        ]
        folds.append(
            {
                "fold": fold,
                "test_groups": test_groups,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )
    return folds


def make_train_dev_test_split(
    group_ids: list[str | None],
    *,
    seed: int = 0,
    dev_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[int] | list[str]]:
    """Create a deterministic, group-disjoint train/dev/test partition.

    Hyperparameters and early stopping may use ``dev`` only. ``test`` is an
    untouched final estimate. Ungrouped examples stay in train because they
    cannot be audited for cross-split leakage.
    """

    if not 0 <= dev_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("dev_fraction and test_fraction must be in [0, 1)")
    if dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be < 1")

    grouped: dict[str, list[int]] = {}
    ungrouped: list[int] = []
    for index, group in enumerate(group_ids):
        if group is None:
            ungrouped.append(index)
        else:
            grouped.setdefault(str(group), []).append(index)

    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    n_groups = len(groups)
    n_test = min(n_groups, max(1, round(n_groups * test_fraction))) if test_fraction else 0
    remaining = n_groups - n_test
    n_dev = min(remaining, max(1, round(n_groups * dev_fraction))) if dev_fraction else 0
    test_groups = groups[:n_test]
    dev_groups = groups[n_test : n_test + n_dev]
    train_groups = groups[n_test + n_dev :]

    def indices_for(selected: list[str]) -> list[int]:
        return [index for group in selected for index in grouped[group]]

    return {
        "train_groups": train_groups,
        "dev_groups": dev_groups,
        "test_groups": test_groups,
        "train_indices": sorted([*ungrouped, *indices_for(train_groups)]),
        "dev_indices": sorted(indices_for(dev_groups)),
        "test_indices": sorted(indices_for(test_groups)),
    }


def collate_batch(
    examples: list[dict],
    vocab: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Build EmbeddingBag tensors (flattened tokens + offsets) for a batch."""

    token_id_lists = [encode_tokens(example["tokens"], vocab) for example in examples]
    offsets = [0]
    for ids in token_id_lists[:-1]:
        offsets.append(offsets[-1] + len(ids))
    flat_tokens = [token_id for ids in token_id_lists for token_id in ids]

    feature_counts = np.asarray(
        [example["feature_counts"] for example in examples], dtype=np.float32
    )
    targets = np.asarray(
        [example["target_reward"] for example in examples], dtype=np.float32
    )
    return {
        "tokens": torch.tensor(flat_tokens, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "feature_counts": torch.from_numpy(feature_counts),
        "targets": torch.from_numpy(targets),
    }


# --------------------------------------------------------------------------- #
# Checkpoint + runtime inference (H0/subgoal extension)
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path,
    model: "TrajectoryFeedbackRewardPredictor",
    vocab: dict[str, int],
    features: list[str],
    *,
    use_feature_counts: bool = False,
    extra: dict | None = None,
) -> None:
    """Persist the trained model with everything needed to run it later."""

    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "features": list(features),
            "config": {
                "vocab_size": model.vocab_size,
                "n_features": model.n_features,
                "use_feature_counts": bool(use_feature_counts),
            },
            "extra": extra or {},
        },
        path,
    )
    # Also drop a plain-JSON vocab next to the checkpoint for easy inspection.
    (path.parent / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_checkpoint(path):
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(model, vocab, features, use_feature_counts)``.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TrajectoryFeedbackRewardPredictor(
        vocab_size=config["vocab_size"], n_features=config["n_features"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        checkpoint["vocab"],
        checkpoint["features"],
        bool(config.get("use_feature_counts", False)),
    )


def predict_reward_vector(
    model: "TrajectoryFeedbackRewardPredictor",
    vocab: dict[str, int],
    features: list[str],
    text: str | None,
    *,
    use_feature_counts: bool = False,
    feature_counts: list[float] | None = None,
) -> dict[str, float]:
    """Text -> predicted reward-weight vector ``w_hat`` (feature -> weight).

    By default the trajectory-feature input is zeroed, so this is a pure
    language -> reward mapping (the honest runtime setting: we do not have the
    grounded reference vector for raw human text). Pass ``use_feature_counts``
    with a matching ``feature_counts`` list to also condition on a grounding.
    """

    from .text_analysis import nn_tokenize

    tokens = nn_tokenize(text)
    encoded = encode_tokens(tokens, vocab)
    tokens_tensor = torch.tensor(encoded, dtype=torch.long)
    offsets_tensor = torch.tensor([0], dtype=torch.long)

    if use_feature_counts and feature_counts is not None:
        counts = np.asarray([feature_counts], dtype=np.float32)
    else:
        counts = np.zeros((1, model.n_features), dtype=np.float32)
    counts_tensor = torch.from_numpy(counts)

    with torch.no_grad():
        prediction = model(tokens_tensor, offsets_tensor, counts_tensor)[0]
    return {feature: float(value) for feature, value in zip(features, prediction.tolist())}
