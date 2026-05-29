"""Graph-native quality metrics for a community partition.

Silhouette Score needs a feature-space distance and does not transfer to graph
communities. These are the standard graph-native analogs:

- coverage: fraction of edges that stay inside communities (higher = better).
- conductance (per community): boundary edges / min(volume, 2m - volume); lower
  means a community is better separated from the rest of the graph.
- modularity vs a degree-preserving configuration-model null: shows whether the
  modularity is real structure or just an artefact of the degree distribution.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

TOP_N = 10
NULL_RUNS = 3


def assess_quality(
    graph: ig.Graph, membership: np.ndarray, modularity_q: float, fig_dir: Path
) -> dict:
    membership = np.asarray(membership)
    m = graph.ecount()
    degree = np.asarray(graph.degree())
    edges = np.array(graph.get_edgelist())
    same = membership[edges[:, 0]] == membership[edges[:, 1]]

    coverage = float(same.sum()) / m
    conductance = _per_community_conductance(membership, degree, edges, same, m)
    null_q = _null_modularity(degree)

    sizes = np.bincount(membership, minlength=membership.max() + 1)
    _plot_conductance(conductance, sizes, fig_dir)

    return {
        "modularity_q": round(float(modularity_q), 4),
        "coverage": round(coverage, 4),
        "mean_conductance": round(float(conductance.mean()), 4),
        "median_conductance": round(float(np.median(conductance)), 4),
        "q_null_mean": round(float(null_q.mean()), 4),
        "q_real_over_null": round(float(modularity_q / null_q.mean()), 1),
    }


def _per_community_conductance(membership, degree, edges, same, m) -> np.ndarray:
    """Conductance of each community: boundary / min(volume, 2m - volume)."""
    ncomm = int(membership.max()) + 1
    volume = np.bincount(membership, weights=degree, minlength=ncomm)
    internal = np.bincount(membership[edges[:, 0]][same], minlength=ncomm)
    cut = volume - 2 * internal
    denom = np.minimum(volume, 2 * m - volume)
    # guard against divide-by-zero for a degenerate single-community partition
    return np.where(denom > 0, cut / denom, 0.0)


def _null_modularity(degree) -> np.ndarray:
    """Best Louvain modularity on degree-preserving random graphs (the null)."""
    qs = []
    for _ in range(NULL_RUNS):
        rnd = ig.Graph.Degree_Sequence(list(degree), method="configuration")
        rnd.simplify()
        qs.append(rnd.community_multilevel().modularity)
    return np.array(qs)


def _plot_conductance(conductance: np.ndarray, sizes: np.ndarray, fig_dir: Path) -> None:
    order = np.argsort(-sizes)[:TOP_N]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(c) for c in order], conductance[order], color="purple")
    ax.set_xlabel("community id (largest first)")
    ax.set_ylabel("conductance (lower = better separated)")
    ax.set_title(f"Conductance of the {TOP_N} largest communities")
    fig.tight_layout()
    fig.savefig(fig_dir / "p2_conductance.png", dpi=150)
    plt.close(fig)
