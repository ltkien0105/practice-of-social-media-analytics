"""Louvain community detection.

Uses igraph's multilevel (Louvain) algorithm to maximise modularity, then
reports the modularity score Q and the community size distribution.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


STABILITY_RUNS = 3


def detect_communities(graph: ig.Graph, fig_dir: Path) -> dict:
    partition = graph.community_multilevel()
    membership = np.array(partition.membership, dtype=np.int32)
    sizes = np.array(partition.sizes(), dtype=np.int64)

    # Louvain processes nodes in a randomised order, so Q can vary run-to-run.
    # Repeat to confirm the modularity score is stable, not a lucky single run.
    q_runs = [round(float(partition.modularity), 4)]
    k_runs = [int(len(sizes))]
    for _ in range(STABILITY_RUNS - 1):
        p = graph.community_multilevel()
        q_runs.append(round(float(p.modularity), 4))
        k_runs.append(len(p.sizes()))

    _plot_size_distribution(sizes, fig_dir)

    stats = {
        "modularity_q": round(float(partition.modularity), 4),
        "n_communities": int(len(sizes)),
        "largest_community": int(sizes.max()),
        "median_community_size": int(np.median(sizes)),
        "n_singletons": int(np.sum(sizes == 1)),
        "q_runs": q_runs,
        "q_std": round(float(np.std(q_runs)), 5),
        "n_communities_runs": k_runs,
    }
    return {"membership": membership, "sizes": sizes, "stats": stats}


def _plot_size_distribution(sizes: np.ndarray, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sizes, bins=60, color="indianred", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("community size (number of nodes)")
    ax.set_ylabel("number of communities")
    ax.set_title("Louvain community size distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "p2_community_sizes.png", dpi=150)
    plt.close(fig)
