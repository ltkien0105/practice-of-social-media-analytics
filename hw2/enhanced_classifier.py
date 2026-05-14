"""
Enhanced classifier with richer features and multiple models.

Features added vs classifier_pipeline.py:
- resource_allocation
- shortest_path (BFS cap=4)
- louvain_consensus (10 runs)
- leiden_consensus (5 resolutions)
- log_deg_u, log_deg_v

Models: sklearn GradientBoosting, XGBoost, LightGBM.
Threshold sweep at predict time (0.3..0.7) writes one submission per
(model, threshold) combo.
"""

from __future__ import annotations

import random
import time
from math import log
from pathlib import Path

import community as community_louvain
import igraph as ig
import leidenalg as la
import lightgbm as lgb
import networkx as nx
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graph_utils import (ROOT, component_map, load_edges, load_graph,
                         load_test_pairs, write_submission)

SEED = 42
# Training pairs are split to mix shortest-path distances across both classes
# so the classifier can't trivially predict "directly connected" instead of
# "same community".
N_POS_EDGE = 3000          # edges (sp=1), label=1
N_POS_SAME_LEIDEN = 3000   # non-edge same-Leiden pairs (sp>=2), label=1
N_NEG_CROSS_LEIDEN = 6000  # same-component but cross-Leiden, label=0
LOUVAIN_RUNS = 10
LEIDEN_RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
SP_CAP = 4

random.seed(SEED)
np.random.seed(SEED)

FEATURE_NAMES = [
    "common_neighbors", "jaccard", "adamic_adar", "resource_allocation",
    "log_pref_attach", "shortest_path",
    "louvain_consensus", "leiden_consensus",
    "same_leiden_default", "same_louvain_default",
    "same_component", "log_deg_u", "log_deg_v",
]


# ---------- shortest-path BFS with depth cap ----------

def short_path(u: int, v: int, g: nx.Graph,
               nu: set[int] | None = None,
               nv: set[int] | None = None,
               cap: int = SP_CAP) -> int:
    """Capped BFS shortest path; returns cap+1 if > cap."""
    if u == v:
        return 0
    if nu is None:
        nu = set(g.neighbors(u))
    if v in nu:
        return 1
    if nv is None:
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


# ---------- per-pair feature computation ----------

def features_for_pair(u: int, v: int, g: nx.Graph,
                      louvain_parts: list[dict[int, int]],
                      leiden_parts: list[dict[int, int]],
                      leiden_default: dict[int, int],
                      louvain_default: dict[int, int],
                      comp_of: dict[int, int]) -> list[float]:
    if u not in g or v not in g:
        return [0.0] * len(FEATURE_NAMES)
    nu = set(g.neighbors(u))
    nv = set(g.neighbors(v))
    inter = nu & nv
    union = nu | nv
    cn = len(inter)
    jacc = cn / len(union) if union else 0.0
    aa = sum(1.0 / log(g.degree(w)) for w in inter if g.degree(w) > 1)
    ra = sum(1.0 / g.degree(w) for w in inter if g.degree(w) > 0)
    pa = log(max(g.degree(u) * g.degree(v), 1))
    sp = short_path(u, v, g, nu=nu, nv=nv, cap=SP_CAP)

    lou_agree = sum(1 for p in louvain_parts if p[u] == p[v])
    lou_cons = lou_agree / len(louvain_parts)
    leid_agree = sum(1 for p in leiden_parts if p[u] == p[v])
    leid_cons = leid_agree / len(leiden_parts)

    same_leid = int(leiden_default[u] == leiden_default[v])
    same_louv = int(louvain_default[u] == louvain_default[v])
    same_comp = int(comp_of.get(u, -1) == comp_of.get(v, -2)) \
        if u in comp_of and v in comp_of else 0

    return [cn, jacc, aa, ra, pa, sp, lou_cons, leid_cons,
            same_leid, same_louv, same_comp,
            log(g.degree(u) + 1), log(g.degree(v) + 1)]


# ---------- community runs ----------

def run_louvain_multi(g: nx.Graph, n: int) -> list[dict[int, int]]:
    out: list[dict[int, int]] = []
    for i in range(n):
        t0 = time.time()
        out.append(community_louvain.best_partition(g, random_state=i))
        print(f"  louvain run {i + 1}/{n} ({time.time() - t0:.1f}s)")
    return out


def run_leiden_multi(edges: list[tuple[int, int]],
                     resolutions: list[float]) -> tuple[list[dict[int, int]],
                                                          dict[int, int]]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    parts: list[dict[int, int]] = []
    default: dict[int, int] = {}
    for r in resolutions:
        t0 = time.time()
        p = la.find_partition(ig_g, la.RBConfigurationVertexPartition,
                              resolution_parameter=r, seed=SEED)
        d = {nodes[i]: p.membership[i] for i in range(len(nodes))}
        parts.append(d)
        if r == 1.0:
            default = d
        print(f"  leiden res={r} communities={len(set(d.values()))} "
              f"Q={p.modularity:.4f} ({time.time() - t0:.1f}s)")
    if not default:
        # fallback: use first resolution as "default"
        default = parts[0]
    return parts, default


# ---------- training-pair sampling ----------

def sample_training(g: nx.Graph, edges: list[tuple[int, int]],
                    leiden_default: dict[int, int],
                    comp_of: dict[int, int]) -> tuple[list[tuple[int, int]],
                                                       list[tuple[int, int]]]:
    """Sample training pairs with mixed shortest-path distances.

    Positives: edges (sp=1) + same-Leiden non-edges (sp>=2)
    Negatives: same-component cross-Leiden pairs (sp varies)
    """
    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big_comps = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}

    pos: list[tuple[int, int]] = list(random.sample(edges, N_POS_EDGE))

    # Same-Leiden non-edge positives
    attempts = 0
    while (len(pos) < N_POS_EDGE + N_POS_SAME_LEIDEN
           and attempts < N_POS_SAME_LEIDEN * 50):
        attempts += 1
        c = random.choice(big_comps)
        ns = by_comp[c]
        u, v = random.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if leiden_default[u] != leiden_default[v]:
            continue
        pos.append((u, v))

    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < N_NEG_CROSS_LEIDEN and attempts < N_NEG_CROSS_LEIDEN * 30:
        attempts += 1
        c = random.choice(big_comps)
        ns = by_comp[c]
        u, v = random.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if leiden_default[u] == leiden_default[v]:
            continue
        neg.append((u, v))
    return pos, neg


# ---------- main ----------

def main() -> None:
    t0 = time.time()
    print("Loading graph...")
    edges = load_edges()
    g = load_graph()
    comp_of = component_map(g)
    print(f"  nodes={g.number_of_nodes()} edges={len(edges)} "
          f"components={len(set(comp_of.values()))} ({time.time() - t0:.1f}s)")

    print(f"Running {LOUVAIN_RUNS} Louvain runs...")
    louvain_parts = run_louvain_multi(g, LOUVAIN_RUNS)
    louvain_default = louvain_parts[0]

    print(f"Running Leiden at {LEIDEN_RESOLUTIONS}...")
    leiden_parts, leiden_default = run_leiden_multi(edges, LEIDEN_RESOLUTIONS)

    t0 = time.time()
    print("Sampling training pairs...")
    pos, neg = sample_training(g, edges, leiden_default, comp_of)
    print(f"  pos={len(pos)} neg={len(neg)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Computing training features...")
    X_list = []
    y_list = []
    for pair_list, label in ((pos, 1), (neg, 0)):
        for u, v in pair_list:
            X_list.append(features_for_pair(
                u, v, g, louvain_parts, leiden_parts,
                leiden_default, louvain_default, comp_of,
            ))
            y_list.append(label)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    print(f"  X.shape={X.shape} ({time.time() - t0:.1f}s)")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)

    # Train models
    models = {}
    t0 = time.time()
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                    random_state=SEED)
    gb.fit(Xtr, ytr)
    print(f"  sklearn-GB val AUC={roc_auc_score(yva, gb.predict_proba(Xva)[:, 1]):.4f} "
          f"acc={gb.score(Xva, yva):.4f} ({time.time() - t0:.1f}s)")
    models["sgb"] = gb

    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=SEED, eval_metric="logloss",
        tree_method="hist",
    )
    xgbm.fit(Xtr, ytr)
    print(f"  XGBoost val AUC={roc_auc_score(yva, xgbm.predict_proba(Xva)[:, 1]):.4f} "
          f"acc={xgbm.score(Xva, yva):.4f} ({time.time() - t0:.1f}s)")
    models["xgb"] = xgbm

    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=500, max_depth=-1, learning_rate=0.05,
        random_state=SEED, verbosity=-1,
    )
    lgbm.fit(Xtr, ytr)
    print(f"  LightGBM val AUC={roc_auc_score(yva, lgbm.predict_proba(Xva)[:, 1]):.4f} "
          f"acc={lgbm.score(Xva, yva):.4f} ({time.time() - t0:.1f}s)")
    models["lgbm"] = lgbm

    print("Feature importances (xgb):")
    for name, imp in zip(FEATURE_NAMES, xgbm.feature_importances_):
        print(f"  {name:<22} {imp:.3f}")

    # Test predictions
    test_pairs = load_test_pairs()
    t0 = time.time()
    Xtest = np.array([
        features_for_pair(n1, n2, g, louvain_parts, leiden_parts,
                          leiden_default, louvain_default, comp_of)
        for _, n1, n2 in test_pairs
    ], dtype=float)
    print(f"Test features computed ({time.time() - t0:.1f}s)")

    for mname, model in models.items():
        probs = model.predict_proba(Xtest)[:, 1]
        for thr in THRESHOLDS:
            preds = [(test_pairs[i][0], int(probs[i] >= thr))
                     for i in range(len(test_pairs))]
            pos_count = sum(1 for _, c in preds if c == 1)
            out = ROOT / f"submission_{mname}_t{thr:g}.csv"
            write_submission(preds, out)
            print(f"  {mname} thr={thr} positives={pos_count} -> {out.name}")


if __name__ == "__main__":
    main()
