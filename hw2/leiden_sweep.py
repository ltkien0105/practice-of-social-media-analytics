"""
Leiden community detection sweep.

Leiden refines Louvain — guarantees connected communities and finds higher
modularity in practice. We use the standard ModularityVertexPartition (no
resolution param) and the RBConfigurationVertexPartition variant for
resolution control.
"""

from __future__ import annotations

import time
from pathlib import Path

import igraph as ig
import leidenalg as la

from graph_utils import (ROOT, component_map, load_edges, load_graph,
                         load_test_pairs, predict_same_community,
                         write_submission)

SEED = 42
RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def build_igraph(edges: list[tuple[int, int]]) -> tuple[ig.Graph, dict[int, int]]:
    """Build undirected igraph; return (graph, node_id -> vertex_index)."""
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_edges = [(idx_of[u], idx_of[v]) for u, v in edges]
    g = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
    g.vs["name"] = nodes
    return g, idx_of


def main() -> None:
    t0 = time.time()
    print("Loading graph...")
    edges = load_edges()
    nx_g = load_graph()  # for component map only
    comp_of = component_map(nx_g)
    print(f"  edges={len(edges)} components={len(set(comp_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    ig_g, idx_of = build_igraph(edges)
    rev_idx = {i: n for n, i in idx_of.items()}
    print(f"Built igraph (vertices={ig_g.vcount()} edges={ig_g.ecount()}) "
          f"({time.time() - t0:.1f}s)")

    test_pairs = load_test_pairs()

    # Run 1: standard modularity (no resolution parameter)
    t0 = time.time()
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=SEED)
    comm_of = {rev_idx[i]: part.membership[i] for i in range(ig_g.vcount())}
    n_comm = len(set(comm_of.values()))
    preds = predict_same_community(test_pairs, comp_of, comm_of)
    pos = sum(1 for _, c in preds if c == 1)
    out = ROOT / "submission_leiden_mod.csv"
    write_submission(preds, out)
    print(f"  modularity-leiden communities={n_comm} Q={part.modularity:.4f} "
          f"positives={pos} -> {out.name} ({time.time() - t0:.1f}s)")

    # Run 2: RBConfiguration with resolution sweep
    for res in RESOLUTIONS:
        t0 = time.time()
        part = la.find_partition(
            ig_g, la.RBConfigurationVertexPartition,
            resolution_parameter=res, seed=SEED,
        )
        comm_of = {rev_idx[i]: part.membership[i] for i in range(ig_g.vcount())}
        n_comm = len(set(comm_of.values()))
        preds = predict_same_community(test_pairs, comp_of, comm_of)
        pos = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_leiden_rb{res:g}.csv"
        write_submission(preds, out)
        print(f"  leiden-rb res={res:<5} communities={n_comm:<5} "
              f"Q={part.modularity:.4f} positives={pos:<4} "
              f"-> {out.name} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
