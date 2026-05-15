"""
Hidden-edge training trick.

Problem: classifier_pipeline labels are (edges, cross-Leiden non-edges).
Edges trivially have shortest_path=1, non-edges have sp>=2. Adding sp as
a feature trivially separates labels -> the model learns "is it an edge?"
not "same community?".

Fix: for half of positive training pairs, compute features WITH THE EDGE
HIDDEN. These positives still have label=1 (we know they're same
community because they were edges) but their features (sp, cn, jaccard,
AA, RA, log_pa) are computed as if (u,v) is not directly connected.
Their sp becomes 2+ (must go through neighbors), matching the
distribution of test-time non-edge pairs.

Result: model learns to use sp / link features for the "same community
but no direct edge" case — which is exactly what we need to recover the
FN that classifier_pipeline misses.

Additional features over classifier_pipeline:
- resource_allocation
- shortest_path (BFS cap=4)
- louvain_consensus (8 runs)
- log_deg_u, log_deg_v

Same labels otherwise: positives = 6000 edges (half hidden), negatives =
6000 same-component cross-Leiden non-edges.

Ensemble: 6 seeds, threshold and rank-based submissions.
"""

from __future__ import annotations

import random
import time
from math import log
from pathlib import Path

import community as community_louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from graph_utils import (ROOT, component_map, load_edges, load_graph,
                         load_test_pairs, write_submission)

SEED_BASE = 42
N_ENSEMBLE = 6
N_POS_VISIBLE = 3000      # edges, features include direct edge (sp=1)
N_POS_HIDDEN = 3000       # edges, features computed with edge HIDDEN (sp>=2)
N_NEG = 6000              # same-component cross-Leiden non-edges
N_LOUVAIN_RUNS = 8
SP_CAP = 4
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
TARGET_COUNTS = [448, 478, 500, 520, 540]


def short_path_excluding(u: int, v: int, g: nx.Graph,
                         hidden: tuple[int, int] | None = None,
                         cap: int = SP_CAP) -> int:
    """BFS with optional hidden edge (u_h, v_h) excluded from traversal."""
    if u == v:
        return 0
    skip = None
    if hidden is not None:
        a, b = hidden
        skip = frozenset((a, b))

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


def pair_features(u: int, v: int, g: nx.Graph,
                  leiden_of: dict[int, int],
                  louvain_of: dict[int, int],
                  louvain_parts: list[dict[int, int]],
                  comp_of: dict[int, int],
                  hide_edge: bool = False) -> list[float]:
    """Compute features for pair (u, v). If hide_edge=True, ignore the
    direct (u,v) edge in cn/jaccard/aa/ra/log_pa/sp calculations
    (used for hidden-edge positive samples in training)."""
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
    sp = short_path_excluding(u, v, g, hidden=hidden_pair, cap=SP_CAP)

    same_leid = int(leiden_of.get(u, -1) == leiden_of.get(v, -2))
    same_louv = int(louvain_of.get(u, -1) == louvain_of.get(v, -2))
    same_comp = int(comp_of.get(u, -1) == comp_of.get(v, -2)) \
        if u in comp_of and v in comp_of else 0
    lou_cons = sum(1 for p in louvain_parts if p[u] == p[v]) / len(louvain_parts)

    return [cn, jacc, aa, ra, pa, sp,
            same_leid, same_louv, same_comp,
            lou_cons, log(deg_u + 1), log(deg_v + 1)]


def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition,
                             seed=SEED_BASE)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def sample_training(g: nx.Graph, edges: list[tuple[int, int]],
                    leiden_of: dict[int, int], comp_of: dict[int, int],
                    rng: random.Random) -> tuple[list[tuple[int, int]],
                                                  list[tuple[int, int]],
                                                  list[tuple[int, int]]]:
    visible_pos = rng.sample(edges, N_POS_VISIBLE)
    # use disjoint set for hidden so identical edges don't appear twice
    remaining = [e for e in edges if e not in set(visible_pos)]
    hidden_pos = rng.sample(remaining, N_POS_HIDDEN)

    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big_comps = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}

    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < N_NEG and attempts < N_NEG * 30:
        attempts += 1
        c = rng.choice(big_comps)
        ns = by_comp[c]
        u, v = rng.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if leiden_of[u] == leiden_of[v]:
            continue
        neg.append((u, v))
    return visible_pos, hidden_pos, neg


def main() -> None:
    t0 = time.time()
    print("Loading graph...")
    edges = load_edges()
    g = load_graph()
    comp_of = component_map(g)
    print(f"  nodes={g.number_of_nodes()} edges={len(edges)} "
          f"components={len(set(comp_of.values()))} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    leiden_of = build_leiden(edges)
    print(f"Leiden communities={len(set(leiden_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    louvain_of = community_louvain.best_partition(g, random_state=SEED_BASE)
    print(f"Louvain communities={len(set(louvain_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    print(f"Running {N_LOUVAIN_RUNS} Louvain runs for consensus...")
    louvain_parts: list[dict[int, int]] = []
    for i in range(N_LOUVAIN_RUNS):
        t0 = time.time()
        louvain_parts.append(
            community_louvain.best_partition(g, random_state=SEED_BASE + i)
        )
        print(f"  run {i + 1}/{N_LOUVAIN_RUNS} ({time.time() - t0:.1f}s)")

    test_pairs = load_test_pairs()
    t0 = time.time()
    Xtest = np.array(
        [pair_features(n1, n2, g, leiden_of, louvain_of, louvain_parts,
                       comp_of, hide_edge=False)
         for _, n1, n2 in test_pairs], dtype=float,
    )
    print(f"Test features {Xtest.shape} ({time.time() - t0:.1f}s)")

    ensemble_probs = np.zeros(len(test_pairs))
    importances = np.zeros(Xtest.shape[1])

    for k in range(N_ENSEMBLE):
        seed = SEED_BASE + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)

        t0 = time.time()
        visible_pos, hidden_pos, neg = sample_training(
            g, edges, leiden_of, comp_of, rng,
        )
        feats: list[list[float]] = []
        for u, v in visible_pos:
            feats.append(pair_features(u, v, g, leiden_of, louvain_of,
                                       louvain_parts, comp_of,
                                       hide_edge=False))
        for u, v in hidden_pos:
            feats.append(pair_features(u, v, g, leiden_of, louvain_of,
                                       louvain_parts, comp_of,
                                       hide_edge=True))
        for u, v in neg:
            feats.append(pair_features(u, v, g, leiden_of, louvain_of,
                                       louvain_parts, comp_of,
                                       hide_edge=False))
        X = np.array(feats, dtype=float)
        y = np.array([1] * (len(visible_pos) + len(hidden_pos))
                     + [0] * len(neg))

        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs = gb.predict_proba(Xtest)[:, 1]
        ensemble_probs += probs
        importances += gb.feature_importances_
        print(f"  member {k + 1}/{N_ENSEMBLE} done ({time.time() - t0:.1f}s)")

    ensemble_probs /= N_ENSEMBLE
    importances /= N_ENSEMBLE
    np.save(ROOT / "hidden_edge_probs.npy", ensemble_probs)

    feat_names = ["cn", "jacc", "aa", "ra", "log_pa", "sp",
                  "same_leid", "same_louv", "same_comp",
                  "lou_cons", "log_deg_u", "log_deg_v"]
    print("\nFeature importances (averaged over ensemble):")
    for n, imp in zip(feat_names, importances):
        print(f"  {n:<12} {imp:.4f}")
    print(f"\nProb stats: min={ensemble_probs.min():.3f} "
          f"max={ensemble_probs.max():.3f} "
          f"median={np.median(ensemble_probs):.3f}")

    for thr in THRESHOLDS:
        preds = [(test_pairs[i][0], int(ensemble_probs[i] >= thr))
                 for i in range(len(test_pairs))]
        pos_count = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_hidden_t{thr:g}.csv"
        write_submission(preds, out)
        print(f"  thr={thr:<5} positives={pos_count:<4} -> {out.name}")

    order = np.argsort(-ensemble_probs)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        pos_count = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_hidden_top{n_target}.csv"
        write_submission(preds, out)
        thr_at = ensemble_probs[order[n_target - 1]] if n_target > 0 else 1.0
        print(f"  top {n_target:<4} (thr~{thr_at:.3f}) positives={pos_count:<4} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
