"""
Edge-feature classifier for same-community prediction.

Self-supervised training:
- positives: random EDGES from train.csv (very likely same community)
- negatives: random non-edge pairs sampled from the same connected
  component but assigned to different Leiden communities
  (very likely different community)

Features per pair (u, v):
- jaccard, adamic_adar, common_neighbors, log_preferential_attachment
- same_leiden_community, same_louvain_community
- same_component

Classifier: GradientBoostingClassifier (handles non-linear feature
interactions, no scaling needed).

Idea: use Leiden labels as a strong baseline feature, then let the
boosted trees correct borderline cases using link-prediction features.
"""

from __future__ import annotations

import math
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graph_utils import (ROOT, component_map, load_edges, load_graph,
                         load_test_pairs, write_submission)

SEED = 42
N_POS = 5000
N_NEG = 5000
random.seed(SEED)
np.random.seed(SEED)


def pair_features(u: int, v: int, g: nx.Graph,
                  leiden_of: dict[int, int],
                  louvain_of: dict[int, int],
                  comp_of: dict[int, int]) -> list[float]:
    """Return feature vector for the pair (u, v).

    Unknown nodes -> zero-ish features so model sees structural absence.
    """
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


def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=SEED)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def sample_training(g: nx.Graph, edges: list[tuple[int, int]],
                    leiden_of: dict[int, int],
                    comp_of: dict[int, int]) -> tuple[list[tuple[int, int]],
                                                       list[tuple[int, int]]]:
    # Positives: random edges
    pos = random.sample(edges, N_POS)

    # Group nodes by component for efficient negative sampling
    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big_comps = [c for c, ns in by_comp.items() if len(ns) >= 50]

    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < N_NEG and attempts < N_NEG * 20:
        c = random.choice(big_comps)
        ns = by_comp[c]
        u, v = random.sample(ns, 2)
        attempts += 1
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
    print("Running Leiden...")
    leiden_of = build_leiden(edges)
    print(f"  leiden communities={len(set(leiden_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Running Louvain...")
    louvain_of = community_louvain.best_partition(g, random_state=SEED)
    print(f"  louvain communities={len(set(louvain_of.values()))} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Sampling training pairs...")
    pos, neg = sample_training(g, edges, leiden_of, comp_of)
    print(f"  pos={len(pos)} neg={len(neg)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Computing features...")
    feats_pos = [pair_features(u, v, g, leiden_of, louvain_of, comp_of)
                 for u, v in pos]
    feats_neg = [pair_features(u, v, g, leiden_of, louvain_of, comp_of)
                 for u, v in neg]
    X = np.array(feats_pos + feats_neg, dtype=float)
    y = np.array([1] * len(pos) + [0] * len(neg))
    print(f"  X.shape={X.shape} ({time.time() - t0:.1f}s)")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)

    t0 = time.time()
    print("Training GradientBoostingClassifier...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    random_state=SEED)
    gb.fit(Xtr, ytr)
    val_auc = roc_auc_score(yva, gb.predict_proba(Xva)[:, 1])
    val_acc = gb.score(Xva, yva)
    print(f"  GB val AUC={val_auc:.4f} acc={val_acc:.4f} "
          f"({time.time() - t0:.1f}s)")
    print("  feature importances "
          "(cn,jacc,aa,pa,same_leid,same_louv,same_comp): "
          f"{gb.feature_importances_.round(3).tolist()}")

    t0 = time.time()
    print("Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(Xtr, ytr)
    lr_auc = roc_auc_score(yva, lr.predict_proba(Xva)[:, 1])
    lr_acc = lr.score(Xva, yva)
    print(f"  LR val AUC={lr_auc:.4f} acc={lr_acc:.4f} "
          f"({time.time() - t0:.1f}s)")

    # Predict on test set
    test_pairs = load_test_pairs()
    Xtest = np.array(
        [pair_features(n1, n2, g, leiden_of, louvain_of, comp_of)
         for _, n1, n2 in test_pairs], dtype=float,
    )

    for name, model in [("gb", gb), ("lr", lr)]:
        probs = model.predict_proba(Xtest)[:, 1]
        preds = [(test_pairs[i][0], int(probs[i] >= 0.5))
                 for i in range(len(test_pairs))]
        pos_count = sum(1 for _, c in preds if c == 1)
        out = ROOT / f"submission_clf_{name}.csv"
        write_submission(preds, out)
        print(f"  {name} test positives={pos_count} -> {out.name}")


if __name__ == "__main__":
    main()
