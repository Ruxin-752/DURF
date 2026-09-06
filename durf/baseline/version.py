"""Version stamps for session metadata.

Every recorded session used to carry one version field, a hard-coded constant
that did not change across eight mechanism commits.  The evaluation matrix's
fairness gate is supposed to compare "candidate generator version" across
conditions; with nothing emitting it, the gate would compare nothing and pass
forever -- worse than no gate.  So: the git commit the code was run from, plus
an explicit generator version that is bumped by hand when the candidate menu
changes shape.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Bump when generate_candidate_subgoals changes WHICH candidates exist or
# their relative scores.  History:
#   rule-cascade-v1          -- through commit 9d136cc
#   enumerating-v1 (2026-09-05) -- both missing ingredients, prep alternatives,
#                                GET_DISH scored (-5) instead of deleted
#   enumerating-v2 (2026-09-06) -- GET_DISH (40) offered when the partner carries
#                                the last ingredient the pot needs; H0 unchanged
#                                (0/13 009 shadow-corpus mismatches)
CANDIDATE_GENERATOR_VERSION = "enumerating-v2"
STANDARD_HORIZON = 800
# The experiment runs on exactly one map.  play_with_baseline used to default to
# cramped_room (the demo map) while sim_session defaulted to the ring -- the
# first live-learning smoke test was played on the wrong map without anyone
# noticing until afterwards.  Both runtimes now default here and refuse
# anything else unless told the run is a smoke test.
STANDARD_LAYOUT = "ring_tomato_onion_10x6_h0_full_task"


def git_commit(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def git_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "durf", "src"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def version_stamp() -> dict:
    return {
        "git_commit": git_commit(),
        "git_dirty_durf_or_src": git_dirty(),
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
    }
