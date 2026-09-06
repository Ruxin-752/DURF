"""Item 2: H0 re-measurement + choice availability per episode.

For every H0 session (hu off, sim persona as partner): the episode reward, and
on the SAME trajectory the number of decision points where a preference would
have had a real choice -- acceptable set at step_tolerance=1 containing >= 2
non-idle actions -- plus how many distinct action pairs those choices span.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

from durf.baseline.collect_rule_teacher_dataset import (
    acceptable_candidates,
    make_motion_planner,
    rule_teacher_candidates,
)
from durf.baseline.task_cost import WAITING_SUBGOALS, attach_step_costs, build_feature_map
from durf.evaluation.matched_replay import reconstruct_state

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
IDLE = WAITING_SUBGOALS | {"PUT_DOWN_OBJECT"}


def main(root: str, out_path: str) -> int:
    planner = make_motion_planner(LAYOUT, seed=0, horizon=800)
    mdp = planner.mdp
    features = build_feature_map(mdp, planner)
    rows_out = []
    for session_dir in sorted(glob.glob(f"{root}/*/*/")):
        traj = Path(session_dir) / "trajectory.csv"
        if not traj.exists():
            continue
        name = Path(session_dir).parent.name  # persona_sN
        persona, seed = name.rsplit("_s", 1)
        rows = list(csv.DictReader(traj.open(encoding="utf-8")))
        reward = float(rows[-1]["episode_reward"])
        choice_points = 0
        pairs = collections.Counter()
        sizes = collections.Counter()
        for row in rows:
            facts = json.loads(row.get("state_before_json") or "{}")
            if not facts:
                continue
            state = reconstruct_state(facts, mdp=mdp)
            cands = rule_teacher_candidates(state, planner, 0)
            attach_step_costs(
                cands, features=features, motion_planner=planner,
                player=state.players[0], state=state, mdp=mdp,
            )
            band = acceptable_candidates(cands, step_tolerance=1.0)
            sizes[band.size] += 1
            real = sorted({c.subgoal for c in band.acceptable if c.subgoal not in IDLE})
            if len(real) >= 2:
                choice_points += 1
                pairs[" vs ".join(real)] += 1
        rows_out.append(
            {
                "persona": persona,
                "seed": int(seed),
                "steps": len(rows),
                "episode_reward": reward,
                "choice_points_tol1": choice_points,
                "distinct_pairs": len(pairs),
                "pairs": dict(pairs),
                "acceptable_sizes": dict(sizes),
            }
        )
    Path(out_path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n", encoding="utf-8"
    )

    print(f"sessions analysed: {len(rows_out)}\n")
    print(f"{'persona':12} {'n':>3} {'reward mean':>12} {'sd':>7} {'min':>6} {'max':>6}  {'choice pts mean':>15} {'min':>5} {'max':>5}  {'pairs mean':>10}")
    for persona in sorted({r["persona"] for r in rows_out}):
        sub = [r for r in rows_out if r["persona"] == persona]
        rw = [r["episode_reward"] for r in sub]
        cp = [r["choice_points_tol1"] for r in sub]
        pr = [r["distinct_pairs"] for r in sub]
        print(f"{persona:12} {len(sub):3d} {statistics.mean(rw):12.1f} {statistics.pstdev(rw):7.1f} {min(rw):6.0f} {max(rw):6.0f}  "
              f"{statistics.mean(cp):15.1f} {min(cp):5d} {max(cp):5d}  {statistics.mean(pr):10.2f}")
    rw = [r["episode_reward"] for r in rows_out]
    cp = [r["choice_points_tol1"] for r in rows_out]
    print(f"\n{'ALL':12} {len(rows_out):3d} {statistics.mean(rw):12.1f} {statistics.pstdev(rw):7.1f} {min(rw):6.0f} {max(rw):6.0f}  "
          f"{statistics.mean(cp):15.1f} {min(cp):5d} {max(cp):5d}")
    agg = collections.Counter()
    for r in rows_out:
        agg.update(r["pairs"])
    total = sum(agg.values())
    print("\npairs across all sessions:")
    for k, v in agg.most_common():
        print(f"   {v:6d} ({100*v/total:5.1f}%)  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
