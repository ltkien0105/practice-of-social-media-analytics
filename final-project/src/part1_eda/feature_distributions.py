"""Node feature distributions.

Summarises the categorical and numeric node attributes: language mix, affiliate
rate, churn (dead account) rate, view count (heavy-tailed → log scale), and
account lifetime. Emits a summary-stats table alongside the figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_LANGUAGES = 10


def analyze_feature_distributions(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> dict:
    affiliate_rate = float(df["affiliate"].mean())
    churn_rate = float(df["dead_account"].mean())
    mature_rate = float(df["mature"].mean())

    lang_counts = df["language"].value_counts()
    top_lang = lang_counts.index[0]
    top_lang_share = float(lang_counts.iloc[0] / len(df))

    _plot_language_bar(lang_counts, fig_dir)
    _plot_views_hist(df["views"].to_numpy(), fig_dir)
    _plot_lifetime_hist(df["life_time"].to_numpy(), fig_dir)
    _write_summary_table(df, lang_counts, table_dir)

    return {
        "affiliate_rate": round(affiliate_rate, 4),
        "churn_rate": round(churn_rate, 4),
        "mature_rate": round(mature_rate, 4),
        "n_languages": int(lang_counts.size),
        "dominant_language": str(top_lang),
        "dominant_language_share": round(top_lang_share, 4),
        "median_views": int(df["views"].median()),
        "median_life_time": int(df["life_time"].median()),
    }


def _plot_language_bar(lang_counts: pd.Series, fig_dir: Path) -> None:
    top = lang_counts.head(TOP_LANGUAGES)
    other = lang_counts.iloc[TOP_LANGUAGES:].sum()
    plot_data = pd.concat([top, pd.Series({"(other)": other})]) if other else top

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_data.index.astype(str), plot_data.to_numpy(), color="slateblue")
    ax.set_ylabel("number of streamers")
    ax.set_title(f"Broadcast language distribution (top {TOP_LANGUAGES} + other)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_language_distribution.png", dpi=150)
    plt.close(fig)


def _plot_views_hist(views: np.ndarray, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    positive = views[views > 0]
    ax.hist(np.log10(positive), bins=60, color="darkorange", edgecolor="white")
    ax.set_xlabel("log10(views)")
    ax.set_ylabel("number of streamers")
    ax.set_title("View count distribution (log scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_views_hist.png", dpi=150)
    plt.close(fig)


def _plot_lifetime_hist(life_time: np.ndarray, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(life_time, bins=60, color="teal", edgecolor="white")
    ax.set_xlabel("account lifetime (days)")
    ax.set_ylabel("number of streamers")
    ax.set_title("Account lifetime distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_lifetime_hist.png", dpi=150)
    plt.close(fig)


def _write_summary_table(df: pd.DataFrame, lang_counts: pd.Series, table_dir: Path) -> None:
    numeric = df[["views", "life_time", "affiliate", "dead_account", "mature"]].describe()
    numeric.to_csv(table_dir / "feature_summary.csv")
    lang_counts.head(TOP_LANGUAGES).to_csv(table_dir / "language_counts_top.csv")
