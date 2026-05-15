"""
Tuned classifier: ensemble of N seeds, threshold sweep over averaged probs.

Builds on the labels & features of classifier_pipeline.py that scored
0.922 on Kaggle. Hypothesis: probability ordering is near-optimal, so
multi-seed averaging cuts noise and threshold tuning extracts more lift.

Differences vs classifier_pipeline:
- Ensemble across SEEDS (different sampling + GB random state) instead of
  one fit. Cuts variance.
- Saves predict_proba per pair (averaged across ensemble members).
- Writes submission at multiple thresholds + count targets.
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
N_ENSEMBLE = 8
N_POS = 5000
N_NEG = 5000
THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
# Also emit submissions targeting specific positive counts via rank threshold.
TARGET_COUNTS = [420, 448, 460, 470, 478, 490, 500]


def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition,
                             seed=SEED_BASE)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def pair_features(u: int, v: int, g: nx.Graph,
                  leiden_of: dict[int, int],
                  louvain_of: dict[int, int],
                  comp_of: dict[int, int]) -> list[float]:
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


def sample_training(g: nx.Graph, edges: list[tuple[int, int]],
                    leiden_of: dict[int, int],
                    comp_of: dict[int, int],
                    rng: random.Random) -> tuple[list[tuple[int, int]],
                                                  list[tuple[int, int]]]:
    pos = rng.sample(edges, N_POS)
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
    return pos, neg


def main() -> None:
    t0 = time.time()
    print("Loading graph...")
    edges = load_edges()
    g = load_graph()
    comp_of = component_map(g)
    print(f"  nodes={g.number_of_nodes()} edges={len(edges)} "
          f"components={len(set(comp_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    leiden_of = build_leiden(edges)
    print(f"Leiden communities={len(set(leiden_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    louvain_of = community_louvain.best_partition(g, random_state=SEED_BASE)
    print(f"Louvain communities={len(set(louvain_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    test_pairs = load_test_pairs()
    Xtest = np.array(
        [pair_features(n1, n2, g, leiden_of, louvain_of, comp_of)
         for _, n1, n2 in test_pairs], dtype=float,
    )
    print(f"Test features: {Xtest.shape}")

    ensemble_probs = np.zeros(len(test_pairs))
    for k in range(N_ENSEMBLE):
        seed = SEED_BASE + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)

        t0 = time.time()
        pos, neg = sample_training(g, edges, leiden_of, comp_of, rng)
        feats_pos = [pair_features(u, v, g, leiden_of, louvain_of, comp_of)
                     for u, v in pos]
        feats_neg = [pair_features(u, v, g, leiden_of, louvain_of, comp_of)
                     for u, v in neg]
        X = np.array(feats_pos + feats_neg, dtype=float)
        y = np.array([1] * len(pos) + [0] * len(neg))

        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs = gb.predict_proba(Xtest)[:, 1]
        ensemble_probs += probs
        print(f"  member {k + 1}/{N_ENSEMBLE} done ({time.time() - t0:.1f}s)")

    ensemble_probs /= N_ENSEMBLE
    np.save(ROOT / "tuned_probs.npy", ensemble_probs)
    print(f"\nProb stats: min={ensemble_probs.min():.3f} "
          f"max={ensemble_probs.max():.3f} median={np.median(ensemble_probs):.3f}")
    print(f"Saved to {ROOT / 'tuned_probs.npy'}")

    # Threshold-based submissions
    for thr in THRESHOLDS:
        preds = [(test_pairs[i][0], int(ensemble_probs[i] >= thr))
                 for i in range(len(test_pairs))]
        pos_count = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_tuned_t{thr:g}.csv"
        write_submission(preds, out)
        print(f"  thr={thr:<5} positives={pos_count:<4} -> {out.name}")

    # Count-target submissions (rank-thresholded)
    order = np.argsort(-ensemble_probs)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        pos_count = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_tuned_top{n_target}.csv"
        write_submission(preds, out)
        thr_at = ensemble_probs[order[n_target - 1]] if n_target > 0 else 1.0
        print(f"  top {n_target:<4} (thr~{thr_at:.3f}) positives={pos_count:<4} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
