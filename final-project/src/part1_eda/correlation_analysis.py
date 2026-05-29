"""Correlation between graph-structural features and node attributes.

Structural: degree, PageRank, local clustering coefficient.
Attributes: views (log), affiliate, account lifetime.

Spearman (rank) correlation is the headline because views/degree/PageRank are
heavy-tailed; Pearson is also emitted for completeness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CORR_COLUMNS = [
    "degree",
    "pagerank",
    "local_clustering",
    "log_views",
    "affiliate",
    "life_time",
]


def analyze_correlation(metrics: pd.DataFrame, fig_dir: Path, table_dir: Path) -> dict:
    frame = metrics[CORR_COLUMNS]
    spearman = frame.corr(method="spearman")
    pearson = frame.corr(method="pearson")

    _plot_heatmap(spearman, "Spearman correlation", "p1_correlation_spearman.png", fig_dir)
    _plot_heatmap(pearson, "Pearson correlation", "p1_correlation_pearson.png", fig_dir)
    spearman.to_csv(table_dir / "correlation_spearman.csv")
    pearson.to_csv(table_dir / "correlation_pearson.csv")

    # Strongest structural<->attribute Spearman pair (exclude the diagonal/self block).
    structural = ["degree", "pagerank", "local_clustering"]
    attributes = ["log_views", "affiliate", "life_time"]
    cross = spearman.loc[structural, attributes].abs()
    flat = cross.stack()
    top_pair = flat.idxmax()
    return {
        "strongest_pair": f"{top_pair[0]} ~ {top_pair[1]}",
        "strongest_spearman": round(float(spearman.loc[top_pair[0], top_pair[1]]), 4),
        "degree_logviews_spearman": round(float(spearman.loc["degree", "log_views"]), 4),
        "pagerank_affiliate_spearman": round(float(spearman.loc["pagerank", "affiliate"]), 4),
    }


def _plot_heatmap(corr: pd.DataFrame, title: str, fname: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
