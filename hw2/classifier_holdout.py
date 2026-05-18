"""
Held-out edge classifier.

Stronger version of the hidden-edge trick: actually remove 10% of train
edges before running community detection. Held-out edges then have a
"truly different" Leiden community label (since Leiden ran without
seeing them) — providing fundamentally different signal than hidden_edge
where Leiden saw all edges.

Training:
- Positives: 3000 visible edges from G_90 (sp=1) + 3000 held-out edges
  (sp >= 2 in G_90, but ground-truth same community)
- Negatives: 6000 same-component cross-Leiden non-edges in G_90
- Features computed on G_90 (consistent feature distribution)

Test: features on G_full (full graph for max info at predict time).

If this still plateaus, the bottleneck is the training-label model
itself (edges = same community), and we need GNN-style learned
representations to push higher.
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

from graph_utils import (ROOT, component_map, load_edges, load_test_pairs,
                         write_submission)

SEED = 42
N_ENSEMBLE = 5
HOLDOUT_FRAC = 0.10
N_POS_VISIBLE = 3000
N_POS_HELDOUT = 3000
N_NEG = 6000
N_LOUVAIN_RUNS = 8
SP_CAP = 4
THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6]
TARGET_COUNTS = [450, 455, 460, 462, 465, 470]


def short_path(u: int, v: int, g: nx.Graph, cap: int = SP_CAP) -> int:
    if u == v:
        return 0
    nu = set(g.neighbors(u))
    if v in nu:
        return 1
    nv = set(g.neighbors(v))
    if nu & nv:
        return 2
    if cap < 3:
        return cap + 1
    visited = {u} | nu
    frontier = nu
    for d in range(2, cap):
        nxt: set[int] = set()
        for w in frontier:
            for x in g._adj[w]:
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
                  comp_of: dict[int, int]) -> list[float]:
    if u not in g or v not in g:
        return [0.0] * 12
    nu = set(g.neighbors(u))
    nv = set(g.neighbors(v))
    inter = nu & nv
    union = nu | nv
    cn = len(inter)
    jacc = cn / len(union) if union else 0.0
    aa = sum(1.0 / log(g.degree(w)) for w in inter if g.degree(w) > 1)
    ra = sum(1.0 / g.degree(w) for w in inter if g.degree(w) > 0)
    deg_u, deg_v = g.degree(u), g.degree(v)
    pa = log(max(deg_u * deg_v, 1))
    sp = short_path(u, v, g, cap=SP_CAP)
    same_leid = int(leiden_of.get(u, -1) == leiden_of.get(v, -2))
    same_louv = int(louvain_of.get(u, -1) == louvain_of.get(v, -2))
    same_comp = int(comp_of.get(u, -1) == comp_of.get(v, -2)) \
        if u in comp_of and v in comp_of else 0
    lou_cons = sum(1 for p in louvain_parts if p[u] == p[v]) / len(louvain_parts)
    return [cn, jacc, aa, ra, pa, sp,
            same_leid, same_louv, same_comp,
            lou_cons, log(deg_u + 1), log(deg_v + 1)]


def build_leiden(edges: list[tuple[int, int]],
                 seed: int = SEED) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=seed)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def sample_negatives(g_90: nx.Graph, edges_90: list[tuple[int, int]],
                     leiden_of: dict[int, int],
                     comp_of: dict[int, int],
                     n_neg: int,
                     rng: random.Random) -> list[tuple[int, int]]:
    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big_comps = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges_90}

    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < n_neg and attempts < n_neg * 30:
        attempts += 1
        c = rng.choice(big_comps)
        ns = by_comp[c]
        u, v = rng.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if u not in leiden_of or v not in leiden_of:
            continue
        if leiden_of[u] == leiden_of[v]:
            continue
        neg.append((u, v))
    return neg


def main() -> None:
    t0 = time.time()
    edges_full = load_edges()
    g_full = nx.Graph()
    g_full.add_edges_from(edges_full)
    print(f"Full graph: nodes={g_full.number_of_nodes()} "
          f"edges={len(edges_full)} ({time.time() - t0:.1f}s)")

    # Hold out 10% of edges
    rng_split = random.Random(SEED)
    edges_shuffled = edges_full[:]
    rng_split.shuffle(edges_shuffled)
    n_holdout = int(HOLDOUT_FRAC * len(edges_shuffled))
    heldout = edges_shuffled[:n_holdout]
    visible = edges_shuffled[n_holdout:]
    print(f"  visible={len(visible)} heldout={len(heldout)}")

    t0 = time.time()
    g_90 = nx.Graph()
    g_90.add_edges_from(visible)
    comp_of_90 = component_map(g_90)
    print(f"G_90: nodes={g_90.number_of_nodes()} edges={g_90.number_of_edges()} "
          f"components={len(set(comp_of_90.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    leiden_of = build_leiden(visible)
    print(f"Leiden G_90: {len(set(leiden_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    louvain_of = community_louvain.best_partition(g_90, random_state=SEED)
    print(f"Louvain G_90: {len(set(louvain_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    print(f"Running {N_LOUVAIN_RUNS} Louvain runs on G_90...")
    louvain_parts: list[dict[int, int]] = []
    for i in range(N_LOUVAIN_RUNS):
        t0 = time.time()
        louvain_parts.append(
            community_louvain.best_partition(g_90, random_state=SEED + i)
        )
        print(f"  run {i + 1}/{N_LOUVAIN_RUNS} ({time.time() - t0:.1f}s)")

    # Filter held-out edges to those whose endpoints survived in G_90
    # (so leiden_of / comp_of_90 have entries for them)
    heldout_in_g90 = [(u, v) for u, v in heldout
                      if u in leiden_of and v in leiden_of]
    print(f"Held-out edges with both endpoints in G_90: {len(heldout_in_g90)}")

    test_pairs = load_test_pairs()
    t0 = time.time()
    Xtest = np.array(
        [pair_features(n1, n2, g_90, leiden_of, louvain_of, louvain_parts,
                       comp_of_90)
         for _, n1, n2 in test_pairs], dtype=float,
    )
    print(f"Test features {Xtest.shape} ({time.time() - t0:.1f}s)")

    ensemble_probs = np.zeros(len(test_pairs))
    importances = np.zeros(Xtest.shape[1])

    for k in range(N_ENSEMBLE):
        seed = SEED + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)

        t0 = time.time()
        # Positives: visible edges (sp=1) + held-out edges (sp>=2)
        n_v = min(N_POS_VISIBLE, len(visible))
        n_h = min(N_POS_HELDOUT, len(heldout_in_g90))
        visible_pos = rng.sample(visible, n_v)
        heldout_pos = rng.sample(heldout_in_g90, n_h)
        # Filter visible_pos for endpoints in leiden_of (should be all)
        visible_pos = [(u, v) for u, v in visible_pos
                       if u in leiden_of and v in leiden_of]

        neg = sample_negatives(g_90, visible, leiden_of, comp_of_90,
                               N_NEG, rng)

        feats: list[list[float]] = []
        for u, v in visible_pos + heldout_pos:
            feats.append(pair_features(u, v, g_90, leiden_of, louvain_of,
                                       louvain_parts, comp_of_90))
        for u, v in neg:
            feats.append(pair_features(u, v, g_90, leiden_of, louvain_of,
                                       louvain_parts, comp_of_90))
        X = np.array(feats, dtype=float)
        y = np.array([1] * (len(visible_pos) + len(heldout_pos))
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
    np.save(ROOT / "holdout_probs.npy", ensemble_probs)

    feat_names = ["cn", "jacc", "aa", "ra", "log_pa", "sp",
                  "same_leid", "same_louv", "same_comp",
                  "lou_cons", "log_deg_u", "log_deg_v"]
    print("\nFeature importances (avg):")
    for n, imp in zip(feat_names, importances):
        print(f"  {n:<12} {imp:.4f}")
    print(f"\nProb stats: min={ensemble_probs.min():.3f} "
          f"max={ensemble_probs.max():.3f} "
          f"median={np.median(ensemble_probs):.3f}")

    for thr in THRESHOLDS:
        preds = [(test_pairs[i][0], int(ensemble_probs[i] >= thr))
                 for i in range(len(test_pairs))]
        pos = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_holdout_t{thr:g}.csv"
        write_submission(preds, out)
        print(f"  thr={thr:<5} positives={pos:<4} -> {out.name}")

    order = np.argsort(-ensemble_probs)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        pos = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_holdout_top{n_target}.csv"
        write_submission(preds, out)
        thr_at = ensemble_probs[order[n_target - 1]]
        print(f"  top {n_target:<4} (thr~{thr_at:.3f}) positives={pos:<4} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
