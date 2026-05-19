"""
HW2 — Community membership prediction (M11415803).

Single-file end-to-end pipeline. Produces M11415803_Le_Trung_Kien.csv
(Kaggle score 0.948).

Run:
    uv run python main.py      (or: python main.py with deps installed)

Pipeline:
1. Load train.csv (188K nodes, 400K edges) and test.csv (1000 pairs).
2. Compute Leiden modularity partition + 8 Louvain partitions
   (different seeds, used for consensus feature).
3. Train two model families on the same edge-vs-cross-Leiden labels:

   a) "Tuned" — 8-seed Gradient Boosting on 7 features
      (common_neighbors, jaccard, AA, log_pa, same_leiden,
       same_louvain, same_component). Best single-model: 0.932.

   b) "Hidden-edge" — 6-seed Gradient Boosting on 12 features
      (adds resource_allocation, shortest_path, louvain_consensus,
       log_deg_u, log_deg_v). Trick: half of positive training pairs
      have their direct (u,v) edge hidden from feature computation,
      so shortest_path doesn't trivially separate edge vs non-edge.
      Best single-model: 0.946.

4. Final = 0.3 * tuned_probs + 0.7 * hidden_probs, take top-460 by
   averaged probability. Score 0.948 on Kaggle.

Dependencies: networkx, python-louvain, igraph, leidenalg, numpy,
              scikit-learn.
"""

from __future__ import annotations

import csv
import random
import time
from math import log
from pathlib import Path

import community as community_louvain  # python-louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
OUT_CSV = ROOT / "M11415803_Le_Trung_Kien.csv"

SEED = 42
N_LOUVAIN_RUNS = 8     # for the "louvain_consensus" feature
SP_CAP = 4             # shortest-path BFS depth cap

# Tuned ensemble
TUNED_N_ENSEMBLE = 8
TUNED_N_POS = 5000
TUNED_N_NEG = 5000

# Hidden-edge ensemble
HIDDEN_N_ENSEMBLE = 6
HIDDEN_N_POS_VISIBLE = 3000
HIDDEN_N_POS_HIDDEN = 3000
HIDDEN_N_NEG = 6000

# Final blend
WEIGHT_TUNED = 0.3
WEIGHT_HIDDEN = 0.7
TOP_N = 460


# ============================================================
# I/O
# ============================================================

def load_edges(path: Path = TRAIN_CSV) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            if u != v:
                edges.append((u, v))
    return edges


def load_test_pairs(path: Path = TEST_CSV) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pairs.append((int(row[0]), int(row[1]), int(row[2])))
    return pairs


def write_submission(preds: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        w.writerows(preds)


def component_map(g: nx.Graph) -> dict[int, int]:
    comp_of: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = cid
    return comp_of


# ============================================================
# Community detection
# ============================================================

def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    """Leiden modularity partition; returns node -> community id."""
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=SEED)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def run_louvain_n(g: nx.Graph, n: int) -> list[dict[int, int]]:
    """Run Louvain n times with seeds SEED, SEED+1, ..., SEED+n-1."""
    parts: list[dict[int, int]] = []
    for i in range(n):
        t0 = time.time()
        parts.append(
            community_louvain.best_partition(g, random_state=SEED + i)
        )
        print(f"  Louvain run {i + 1}/{n} ({time.time() - t0:.1f}s)")
    return parts


# ============================================================
# Features
# ============================================================

def short_path(u: int, v: int, g: nx.Graph,
               hidden: tuple[int, int] | None = None,
               cap: int = SP_CAP) -> int:
    """Capped BFS distance; if `hidden` set, the (a,b) edge is ignored."""
    if u == v:
        return 0
    skip = frozenset(hidden) if hidden is not None else None

    def neighbors(w: int):
        for x in g._adj[w]:
            if skip is not None and w in skip and x in skip:
                continue
            yield x

    nu = set(neighbors(u))
    if v in nu:
        return 1
    nv = set(neighbors(v))
    if nu & nv:
        return 2
    if cap < 3:
        return cap + 1
    visited = {u} | nu
    frontier = nu
    for d in range(2, cap):
        nxt: set[int] = set()
        for w in frontier:
            for x in neighbors(w):
                if x == v:
                    return d + 1
                if x not in visited:
                    nxt.add(x)
        if not nxt:
            return cap + 1
        visited |= nxt
        frontier = nxt
    return cap + 1


def features_tuned(u: int, v: int, g: nx.Graph,
                   leiden_of: dict[int, int],
                   louvain_of: dict[int, int],
                   comp_of: dict[int, int]) -> list[float]:
    """7-feature vector for the tuned classifier."""
    if u not in g or v not in g:
        return [0.0, 0.0, 0.0, 0.0, 0, 0, 0]
    nu, nv = set(g.neighbors(u)), set(g.neighbors(v))
    inter = nu & nv
    union = nu | nv
    cn = len(inter)
    jacc = cn / len(union) if union else 0.0
    aa = sum(1.0 / log(g.degree(w)) for w in inter if g.degree(w) > 1)
    pa = log(max(g.degree(u) * g.degree(v), 1))
    same_leid = int(leiden_of.get(u, -1) == leiden_of.get(v, -2))
    same_louv = int(louvain_of.get(u, -1) == louvain_of.get(v, -2))
    same_comp = int(comp_of.get(u, -1) == comp_of.get(v, -2)) \
        if u in comp_of and v in comp_of else 0
    return [cn, jacc, aa, pa, same_leid, same_louv, same_comp]


def features_hidden(u: int, v: int, g: nx.Graph,
                    leiden_of: dict[int, int],
                    louvain_of: dict[int, int],
                    louvain_parts: list[dict[int, int]],
                    comp_of: dict[int, int],
                    hide_edge: bool = False) -> list[float]:
    """12-feature vector for the hidden-edge classifier."""
    if u not in g or v not in g:
        return [0.0] * 12

    if hide_edge and g.has_edge(u, v):
        nu = set(g.neighbors(u)) - {v}
        nv = set(g.neighbors(v)) - {u}
        deg_u = g.degree(u) - 1
        deg_v = g.degree(v) - 1
        hidden_pair = (u, v)
    else:
        nu = set(g.neighbors(u))
        nv = set(g.neighbors(v))
        deg_u = g.degree(u)
        deg_v = g.degree(v)
        hidden_pair = None

    inter = nu & nv
    union = nu | nv
    cn = len(inter)
    jacc = cn / len(union) if union else 0.0
    aa = sum(1.0 / log(g.degree(w)) for w in inter if g.degree(w) > 1)
    ra = sum(1.0 / g.degree(w) for w in inter if g.degree(w) > 0)
    pa = log(max(deg_u * deg_v, 1))
    sp = short_path(u, v, g, hidden=hidden_pair, cap=SP_CAP)
    same_leid = int(leiden_of.get(u, -1) == leiden_of.get(v, -2))
    same_louv = int(louvain_of.get(u, -1) == louvain_of.get(v, -2))
    same_comp = int(comp_of.get(u, -1) == comp_of.get(v, -2)) \
        if u in comp_of and v in comp_of else 0
    lou_cons = sum(1 for p in louvain_parts
                   if p[u] == p[v]) / len(louvain_parts)
    return [cn, jacc, aa, ra, pa, sp,
            same_leid, same_louv, same_comp,
            lou_cons, log(deg_u + 1), log(deg_v + 1)]


# ============================================================
# Training-pair sampling
# ============================================================

def _negatives_cross_leiden(edges: list[tuple[int, int]],
                            leiden_of: dict[int, int],
                            comp_of: dict[int, int],
                            n_neg: int,
                            rng: random.Random) -> list[tuple[int, int]]:
    """Same-component, non-edge, cross-Leiden pairs."""
    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < n_neg and attempts < n_neg * 30:
        attempts += 1
        c = rng.choice(big)
        ns = by_comp[c]
        u, v = rng.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if leiden_of[u] == leiden_of[v]:
            continue
        neg.append((u, v))
    return neg


def sample_tuned(edges: list[tuple[int, int]],
                 leiden_of: dict[int, int], comp_of: dict[int, int],
                 rng: random.Random) -> tuple[list[tuple[int, int]],
                                                list[tuple[int, int]]]:
    pos = rng.sample(edges, TUNED_N_POS)
    neg = _negatives_cross_leiden(edges, leiden_of, comp_of,
                                  TUNED_N_NEG, rng)
    return pos, neg


def sample_hidden(edges: list[tuple[int, int]],
                  leiden_of: dict[int, int], comp_of: dict[int, int],
                  rng: random.Random) -> tuple[list[tuple[int, int]],
                                                 list[tuple[int, int]],
                                                 list[tuple[int, int]]]:
    visible = rng.sample(edges, HIDDEN_N_POS_VISIBLE)
    visible_set = set(visible)
    remaining = [e for e in edges if e not in visible_set]
    hidden = rng.sample(remaining, HIDDEN_N_POS_HIDDEN)
    neg = _negatives_cross_leiden(edges, leiden_of, comp_of,
                                  HIDDEN_N_NEG, rng)
    return visible, hidden, neg


# ============================================================
# Ensemble training
# ============================================================

def train_tuned(g: nx.Graph, edges: list[tuple[int, int]],
                leiden_of: dict[int, int],
                louvain_of: dict[int, int],
                comp_of: dict[int, int],
                test_pairs: list[tuple[int, int, int]]) -> np.ndarray:
    Xtest = np.array(
        [features_tuned(n1, n2, g, leiden_of, louvain_of, comp_of)
         for _, n1, n2 in test_pairs], dtype=float,
    )
    probs = np.zeros(len(test_pairs))
    for k in range(TUNED_N_ENSEMBLE):
        t0 = time.time()
        seed = SEED + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)
        pos, neg = sample_tuned(edges, leiden_of, comp_of, rng)
        X = np.array(
            [features_tuned(u, v, g, leiden_of, louvain_of, comp_of)
             for u, v in pos + neg], dtype=float,
        )
        y = np.array([1] * len(pos) + [0] * len(neg))
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs += gb.predict_proba(Xtest)[:, 1]
        print(f"  tuned member {k + 1}/{TUNED_N_ENSEMBLE} "
              f"({time.time() - t0:.1f}s)")
    return probs / TUNED_N_ENSEMBLE


def train_hidden(g: nx.Graph, edges: list[tuple[int, int]],
                 leiden_of: dict[int, int],
                 louvain_of: dict[int, int],
                 louvain_parts: list[dict[int, int]],
                 comp_of: dict[int, int],
                 test_pairs: list[tuple[int, int, int]]) -> np.ndarray:
    Xtest = np.array(
        [features_hidden(n1, n2, g, leiden_of, louvain_of, louvain_parts,
                         comp_of, hide_edge=False)
         for _, n1, n2 in test_pairs], dtype=float,
    )
    probs = np.zeros(len(test_pairs))
    for k in range(HIDDEN_N_ENSEMBLE):
        t0 = time.time()
        seed = SEED + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)
        visible, hidden, neg = sample_hidden(edges, leiden_of, comp_of, rng)
        feats: list[list[float]] = []
        for u, v in visible:
            feats.append(features_hidden(u, v, g, leiden_of, louvain_of,
                                         louvain_parts, comp_of,
                                         hide_edge=False))
        for u, v in hidden:
            feats.append(features_hidden(u, v, g, leiden_of, louvain_of,
                                         louvain_parts, comp_of,
                                         hide_edge=True))
        for u, v in neg:
            feats.append(features_hidden(u, v, g, leiden_of, louvain_of,
                                         louvain_parts, comp_of,
                                         hide_edge=False))
        X = np.array(feats, dtype=float)
        y = np.array([1] * (len(visible) + len(hidden))
                     + [0] * len(neg))
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs += gb.predict_proba(Xtest)[:, 1]
        print(f"  hidden member {k + 1}/{HIDDEN_N_ENSEMBLE} "
              f"({time.time() - t0:.1f}s)")
    return probs / HIDDEN_N_ENSEMBLE


# ============================================================
# Main
# ============================================================

def main() -> None:
    t_total = time.time()
    print("Loading graph...")
    edges = load_edges()
    g = nx.Graph()
    g.add_edges_from(edges)
    comp_of = component_map(g)
    print(f"  nodes={g.number_of_nodes()} edges={len(edges)} "
          f"components={len(set(comp_of.values()))}")

    print("\nLeiden modularity partition...")
    t0 = time.time()
    leiden_of = build_leiden(edges)
    print(f"  {len(set(leiden_of.values()))} communities "
          f"({time.time() - t0:.1f}s)")

    print(f"\nRunning {N_LOUVAIN_RUNS} Louvain partitions...")
    louvain_parts = run_louvain_n(g, N_LOUVAIN_RUNS)
    louvain_of = louvain_parts[0]  # default partition (seed=42)
    print(f"  default partition: {len(set(louvain_of.values()))} communities")

    test_pairs = load_test_pairs()
    print(f"\nTest pairs to predict: {len(test_pairs)}")

    print("\nTraining tuned ensemble (8 seeds, 7 features)...")
    t0 = time.time()
    tuned_probs = train_tuned(g, edges, leiden_of, louvain_of, comp_of,
                              test_pairs)
    np.save(ROOT / "tuned_probs.npy", tuned_probs)
    print(f"  tuned done ({time.time() - t0:.1f}s) -> tuned_probs.npy")

    print("\nTraining hidden-edge ensemble (6 seeds, 12 features)...")
    t0 = time.time()
    hidden_probs = train_hidden(g, edges, leiden_of, louvain_of,
                                louvain_parts, comp_of, test_pairs)
    np.save(ROOT / "hidden_probs.npy", hidden_probs)
    print(f"  hidden-edge done ({time.time() - t0:.1f}s) -> hidden_probs.npy")

    print(f"\nBlending {WEIGHT_TUNED} * tuned + {WEIGHT_HIDDEN} * hidden, "
          f"top-{TOP_N} by averaged probability...")
    ens = WEIGHT_TUNED * tuned_probs + WEIGHT_HIDDEN * hidden_probs
    order = np.argsort(-ens)
    positive_idx = set(order[:TOP_N].tolist())
    preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
             for i in range(len(test_pairs))]
    pos_count = sum(1 for _, c in preds if c == 1)
    write_submission(preds, OUT_CSV)
    print(f"  positives={pos_count} -> {OUT_CSV.name}")
    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
