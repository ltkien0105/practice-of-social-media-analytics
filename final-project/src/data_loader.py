"""Shared data layer for the Twitch Gamers analysis.

One loader, one cached graph build, imported by both Part 1 and Part 2.

Invariant relied on throughout the project: ``numeric_id`` values are the
contiguous range ``0..n-1``, so the features DataFrame (sorted by ``numeric_id``)
aligns row-for-row with igraph vertex indices. ``build_graph`` asserts this.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

# Resolve paths relative to the project root (parent of this file's dir).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "twitch_gamers"
FEATURES_CSV = DATA_DIR / "large_twitch_features.csv"
EDGES_CSV = DATA_DIR / "large_twitch_edges.csv"
CACHE_DIR = PROJECT_ROOT / "results" / "cache"
GRAPH_CACHE = CACHE_DIR / "graph.pkl"

EXPECTED_NODES = 168_114
EXPECTED_TRANSITIVITY = 0.0184  # README checkpoint


def set_seed(seed: int = 42) -> None:
    """Make igraph's randomised algorithms reproducible.

    Louvain visits vertices in a randomised order, so the community count and
    exact modularity Q drift run-to-run. Routing igraph through Python's RNG and
    seeding it once makes the whole Part 2 pipeline (detection, stability runs,
    and the configuration-model null) identical across runs and machines.
    """
    ig.set_random_number_generator(random)
    random.seed(seed)


def load_features() -> pd.DataFrame:
    """Load node features sorted by ``numeric_id`` with explicit dtypes."""
    df = pd.read_csv(
        FEATURES_CSV,
        dtype={
            "views": np.int64,
            "mature": np.int8,
            "life_time": np.int32,
            "numeric_id": np.int32,
            "dead_account": np.int8,
            "language": "string",
            "affiliate": np.int8,
        },
        parse_dates=["created_at", "updated_at"],
    )
    df = df.sort_values("numeric_id").reset_index(drop=True)
    # Heavy-tailed view counts: keep a log1p transform for plots/correlation.
    df["log_views"] = np.log1p(df["views"].to_numpy())
    return df


def build_graph(features: pd.DataFrame) -> ig.Graph:
    """Build an undirected, simplified graph with vertices in numeric_id order."""
    ids = features["numeric_id"].to_numpy()
    n = len(ids)
    # Positional vertex<->row mapping only holds if ids are the contiguous 0..n-1.
    assert np.array_equal(ids, np.arange(n)), "numeric_id is not contiguous 0..n-1"

    edges = pd.read_csv(EDGES_CSV, dtype=np.int32)
    edge_list = list(zip(edges["numeric_id_1"], edges["numeric_id_2"]))

    graph = ig.Graph(n=n, edges=edge_list, directed=False)
    graph.simplify()  # drop duplicate edges and self-loops

    _verify_graph(graph)
    return graph


def _verify_graph(graph: ig.Graph) -> None:
    """Fail fast if the build diverges from the documented dataset."""
    assert graph.vcount() == EXPECTED_NODES, (
        f"expected {EXPECTED_NODES} vertices, got {graph.vcount()}"
    )
    transitivity = graph.transitivity_undirected()
    assert 0.017 < transitivity < 0.020, (
        f"global transitivity {transitivity:.4f} off README checkpoint "
        f"{EXPECTED_TRANSITIVITY}"
    )


def load_graph_cached(rebuild: bool = False) -> ig.Graph:
    """Return the graph, building and pickling it on first use."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GRAPH_CACHE.exists() and not rebuild:
        with GRAPH_CACHE.open("rb") as fh:
            return pickle.load(fh)

    graph = build_graph(load_features())
    with GRAPH_CACHE.open("wb") as fh:
        pickle.dump(graph, fh)
    return graph


if __name__ == "__main__":
    g = load_graph_cached()
    print(g.summary())
    print(f"global transitivity = {g.transitivity_undirected():.4f}")
