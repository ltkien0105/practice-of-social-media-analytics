"""
Consensus Louvain.

Run Louvain N times with different random seeds, then for each test pair
count fraction of runs where both nodes land in the same community. Vote
threshold T: predict 1 if fraction >= T else 0.

Reduces stochastic noise from Louvain's randomized refinement phase.
Writes one submission per threshold.
"""

from __future__ import annotations

import time
from pathlib import Path

import community as community_louvain
import networkx as nx

from graph_utils import (ROOT, component_map, load_graph, load_test_pairs,
                         write_submission)

N_RUNS = 20
RESOLUTION = 1.0
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def main() -> None:
    t0 = time.time()
    g = load_graph()
    comp_of = component_map(g)
    test_pairs = load_test_pairs()
    print(f"Graph loaded: nodes={g.number_of_nodes()} "
          f"edges={g.number_of_edges()} ({time.time() - t0:.1f}s)")

    # Filter test pairs that can possibly be positive (same component, both known)
    candidate_pairs: list[tuple[int, int, int]] = []
    for idx, n1, n2 in test_pairs:
        if (n1 in comp_of and n2 in comp_of
                and comp_of[n1] == comp_of[n2]):
            candidate_pairs.append((idx, n1, n2))
    print(f"Candidate pairs (same component): {len(candidate_pairs)}/{len(test_pairs)}")

    # Co-occurrence counter: pair -> number of runs they share a community
    co: dict[int, int] = {p[0]: 0 for p in candidate_pairs}

    for run_i in range(N_RUNS):
        t0 = time.time()
        comm_of = community_louvain.best_partition(
            g, resolution=RESOLUTION, random_state=run_i,
        )
        for idx, n1, n2 in candidate_pairs:
            if comm_of[n1] == comm_of[n2]:
                co[idx] += 1
        print(f"  run {run_i + 1}/{N_RUNS} done ({time.time() - t0:.1f}s)")

    # Emit submissions at each threshold
    for thr in THRESHOLDS:
        min_votes = int(round(thr * N_RUNS))
        preds: list[tuple[int, int]] = []
        for idx, _n1, _n2 in test_pairs:
            if idx in co and co[idx] >= min_votes:
                preds.append((idx, 1))
            else:
                preds.append((idx, 0))
        pos = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_consensus_t{thr:g}.csv"
        write_submission(preds, out)
        print(f"  threshold={thr} (min_votes={min_votes}) positives={pos} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
