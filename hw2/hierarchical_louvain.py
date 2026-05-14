"""
Hierarchical Louvain.

Louvain builds a dendrogram: level 0 is finest (many small communities),
each subsequent level merges. python-louvain exposes this via
generate_dendrogram + partition_at_level. Coarser levels may fix
over-fragmentation that costs us positives at the default resolution.
"""

from __future__ import annotations

import time
from pathlib import Path

import community as community_louvain
import networkx as nx

from graph_utils import (ROOT, component_map, load_graph, load_test_pairs,
                         predict_same_community, write_submission)

SEED = 42


def main() -> None:
    t0 = time.time()
    g = load_graph()
    comp_of = component_map(g)
    test_pairs = load_test_pairs()
    print(f"Graph loaded: nodes={g.number_of_nodes()} "
          f"edges={g.number_of_edges()} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    dendro = community_louvain.generate_dendrogram(g, random_state=SEED)
    print(f"Dendrogram levels: {len(dendro)} ({time.time() - t0:.1f}s)")

    for level in range(len(dendro)):
        t0 = time.time()
        comm_of = community_louvain.partition_at_level(dendro, level)
        n_comm = len(set(comm_of.values()))
        mod = community_louvain.modularity(comm_of, g)
        preds = predict_same_community(test_pairs, comp_of, comm_of)
        pos = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_hier_level{level}.csv"
        write_submission(preds, out)
        print(f"  level={level} communities={n_comm:<6} modularity={mod:.4f} "
              f"positives={pos:<4} -> {out.name} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
