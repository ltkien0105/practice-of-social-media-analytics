"""Profile the major communities.

For the top-N communities by size, report: size, dominant language, language
purity (share of the dominant language within the community), affiliate rate,
and median view count. Community labels align positionally with the features
rows via the Phase 1 numeric_id ordering invariant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_N = 10


def profile_communities(
    df: pd.DataFrame, membership: np.ndarray, fig_dir: Path, table_dir: Path
) -> dict:
    profiled = df.copy()
    profiled["community"] = membership

    sizes = profiled["community"].value_counts()
    top_ids = sizes.head(TOP_N).index.tolist()

    rows = []
    for cid in top_ids:
        members = profiled[profiled["community"] == cid]
        lang_counts = members["language"].value_counts()
        dominant_lang = lang_counts.index[0]
        purity = float(lang_counts.iloc[0] / len(members))
        rows.append({
            "community": int(cid),
            "size": int(len(members)),
            "dominant_language": str(dominant_lang),
            "language_purity": round(purity, 4),
            "affiliate_rate": round(float(members["affiliate"].mean()), 4),
            "median_views": int(members["views"].median()),
        })

    profile = pd.DataFrame(rows)
    profile.to_csv(table_dir / "community_profile.csv", index=False)
    _plot_language_composition(profiled, top_ids, fig_dir)

    coverage = float(sizes.head(TOP_N).sum() / len(df))
    return {
        "profile": profile,
        "top_n_coverage": round(coverage, 4),
        "mean_purity_topn": round(float(profile["language_purity"].mean()), 4),
    }


def _plot_language_composition(profiled: pd.DataFrame, top_ids: list, fig_dir: Path) -> None:
    # Stacked bar of language composition within each top community.
    subset = profiled[profiled["community"].isin(top_ids)]
    comp = (
        subset.groupby(["community", "language"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(top_ids)
    )
    comp = comp.div(comp.sum(axis=1), axis=0)  # normalise to shares

    fig, ax = plt.subplots(figsize=(10, 6))
    comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85)
    ax.set_xlabel("community id")
    ax.set_ylabel("language share")
    ax.set_title(f"Language composition of the {len(top_ids)} largest communities")
    ax.legend(title="language", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p2_language_composition.png", dpi=150)
    plt.close(fig)
