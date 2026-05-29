"""Global transitivity and average local clustering coefficient.

Global transitivity is the fraction of closed triplets; average local clustering
averages each node's neighbourhood density. Both quantify how tightly the network
forms local groups; on a sparse 6.8M-edge graph both are expected to be small.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


def analyze_clustering(graph: ig.Graph, local_clustering: np.ndarray, fig_dir: Path) -> dict:
    global_transitivity = graph.transitivity_undirected()
    # mode='zero' counts degree<2 nodes as 0 so the mean covers all vertices.
    avg_local = float(np.mean(local_clustering))

    # Reference point: an Erdos-Renyi random graph of the same density has
    # expected clustering ~= density. The ratio shows how much local structure
    # exceeds chance.
    density = float(graph.density())
    clustering_vs_random = float(global_transitivity / density) if density else float("nan")

    _plot_local_clustering_hist(local_clustering, fig_dir)

    return {
        "global_transitivity": round(float(global_transitivity), 4),
        "avg_local_clustering": round(avg_local, 4),
        "density": round(density, 6),
        "random_baseline_clustering": round(density, 6),
        "clustering_vs_random_ratio": round(clustering_vs_random, 1),
    }


def _plot_local_clustering_hist(local_clustering: np.ndarray, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(local_clustering, bins=50, color="seagreen", edgecolor="white")
    ax.set_xlabel("local clustering coefficient")
    ax.set_ylabel("number of nodes")
    ax.set_yscale("log")
    ax.set_title("Distribution of local clustering coefficients")
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_local_clustering_hist.png", dpi=150)
    plt.close(fig)
