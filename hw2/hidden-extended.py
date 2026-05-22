"""
Extend the hidden-edge classifier with embedding-cosine features.

Idea: instead of blending the 0.948 baseline with separate n2v / spec
probability vectors (all tested, all <= 0.948), let the GBM classifier
itself decide how to use the embedding signals as features.

New features (added to the 12-dim hidden feature set):
- n2v_cos: cosine of biased Node2Vec embeddings (p=1, q=0.5)
- spec_cos: cosine of spectral embeddings (k=64)

Then retrain the 6-seed hidden ensemble on the extended 14-dim space,
blend with the existing tuned probs (0.3/0.7), and produce a top-460
submission.

Run:
    uv run python hidden-extended.py
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from math import log
from pathlib import Path

import community as community_louvain
import igraph as ig
import leidenalg as la_pkg
import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

SEED = 42
SP_CAP = 4
N_LOUVAIN_RUNS = 8
HIDDEN_N_ENSEMBLE = 6
HIDDEN_N_POS_VISIBLE = 3000
HIDDEN_N_POS_HIDDEN = 3000
HIDDEN_N_NEG = 6000
TOP_N = 460


def load_edges():
    edges = []
    with TRAIN_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            u, v = int(row[0]), int(row[1])
            if u != v:
                edges.append((u, v))
    return edges


def load_test_pairs():
    pairs = []
    with TEST_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            pairs.append((int(row[0]), int(row[1]), int(row[2])))
    return pairs


def component_map(g):
    cm = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            cm[n] = cid
    return cm


def build_leiden(edges):
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la_pkg.find_partition(ig_g, la_pkg.ModularityVertexPartition,
                                 seed=SEED)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def run_louvain_n(g, n):
    parts = []
    for i in range(n):
        t0 = time.time()
        parts.append(community_louvain.best_partition(g, random_state=SEED + i))
        print(f"  Louvain {i + 1}/{n} ({time.time() - t0:.1f}s)")
    return parts


def short_path(u, v, g, hidden=None, cap=SP_CAP):
    if u == v:
        return 0
    skip = frozenset(hidden) if hidden is not None else None

    def neighbors(w):
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
        nxt = set()
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


def emb_cos(u, v, emb, idx, norms):
    iu, iv = idx.get(u), idx.get(v)
    if iu is None or iv is None:
        return 0.0
    nu, nv = norms[iu], norms[iv]
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(emb[iu], emb[iv]) / (nu * nv))


def features_ext(u, v, g, leiden_of, louvain_of, louvain_parts, comp_of,
                 n2v_emb, n2v_idx, n2v_norms,
                 spec_emb, spec_idx, spec_norms,
                 hide_edge=False):
    if u not in g or v not in g:
        return [0.0] * 14

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
    same_comp = (int(comp_of.get(u, -1) == comp_of.get(v, -2))
                 if u in comp_of and v in comp_of else 0)
    lou_cons = sum(1 for p in louvain_parts if p[u] == p[v]) / len(louvain_parts)
    n2v_c = emb_cos(u, v, n2v_emb, n2v_idx, n2v_norms)
    spec_c = emb_cos(u, v, spec_emb, spec_idx, spec_norms)
    return [cn, jacc, aa, ra, pa, sp, same_leid, same_louv, same_comp,
            lou_cons, log(deg_u + 1), log(deg_v + 1), n2v_c, spec_c]


def cross_leiden_negatives(edges, leiden_of, comp_of, n_neg, rng):
    by_comp = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    neg = []
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


def main():
    t_total = time.time()
    edges = load_edges()
    test_pairs = load_test_pairs()
    g = nx.Graph(); g.add_edges_from(edges)
    comp_of = component_map(g)
    print(f"graph: nodes={g.number_of_nodes()} edges={len(edges)}")

    print("Leiden...")
    leiden_of = build_leiden(edges)
    print(f"  {len(set(leiden_of.values()))} communities")

    print(f"Louvain x{N_LOUVAIN_RUNS}...")
    louvain_parts = run_louvain_n(g, N_LOUVAIN_RUNS)
    louvain_of = louvain_parts[0]

    print("Loading cached embeddings...")
    n2v_emb = np.load(ROOT / "n2v_embeddings_p10_q05.npy")
    with (ROOT / "n2v_node_index_p10_q05.pkl").open("rb") as f:
        n2v_idx = pickle.load(f)
    n2v_norms = np.linalg.norm(n2v_emb, axis=1)
    spec_emb = np.load(ROOT / "spec_embeddings_k64.npy")
    with (ROOT / "spec_node_index_k64.pkl").open("rb") as f:
        spec_idx = pickle.load(f)
    spec_norms = np.linalg.norm(spec_emb, axis=1)
    print(f"  n2v: {n2v_emb.shape}  spec: {spec_emb.shape}")

    print("Computing test features...")
    Xtest = np.array([
        features_ext(n1, n2, g, leiden_of, louvain_of, louvain_parts, comp_of,
                     n2v_emb, n2v_idx, n2v_norms,
                     spec_emb, spec_idx, spec_norms, hide_edge=False)
        for _, n1, n2 in test_pairs
    ], dtype=float)

    probs = np.zeros(len(test_pairs))
    for k in range(HIDDEN_N_ENSEMBLE):
        t0 = time.time()
        seed = SEED + k * 17
        rng = random.Random(seed)
        np.random.seed(seed)
        visible = rng.sample(edges, HIDDEN_N_POS_VISIBLE)
        visible_set = set(visible)
        remaining = [e for e in edges if e not in visible_set]
        hidden = rng.sample(remaining, HIDDEN_N_POS_HIDDEN)
        neg = cross_leiden_negatives(edges, leiden_of, comp_of, HIDDEN_N_NEG, rng)

        feats = []
        for u, v in visible:
            feats.append(features_ext(u, v, g, leiden_of, louvain_of,
                                      louvain_parts, comp_of,
                                      n2v_emb, n2v_idx, n2v_norms,
                                      spec_emb, spec_idx, spec_norms,
                                      hide_edge=False))
        for u, v in hidden:
            feats.append(features_ext(u, v, g, leiden_of, louvain_of,
                                      louvain_parts, comp_of,
                                      n2v_emb, n2v_idx, n2v_norms,
                                      spec_emb, spec_idx, spec_norms,
                                      hide_edge=True))
        for u, v in neg:
            feats.append(features_ext(u, v, g, leiden_of, louvain_of,
                                      louvain_parts, comp_of,
                                      n2v_emb, n2v_idx, n2v_norms,
                                      spec_emb, spec_idx, spec_norms,
                                      hide_edge=False))
        X = np.array(feats, dtype=float)
        y = np.array([1] * (len(visible) + len(hidden)) + [0] * len(neg))
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs += gb.predict_proba(Xtest)[:, 1]
        if k == 0:
            fnames = ["cn", "jacc", "aa", "ra", "log_pa", "sp",
                     "same_leid", "same_louv", "same_comp", "lou_cons",
                     "log_deg_u", "log_deg_v", "n2v_cos", "spec_cos"]
            imp = sorted(zip(fnames, gb.feature_importances_),
                         key=lambda x: -x[1])
            print("  feature importance (seed 0):")
            for n, v in imp:
                print(f"    {n:>10}: {v:.4f}")
        print(f"  ext-hidden member {k + 1}/{HIDDEN_N_ENSEMBLE} "
              f"({time.time() - t0:.1f}s)")
    probs /= HIDDEN_N_ENSEMBLE
    np.save(ROOT / "hidden_ext_probs.npy", probs)

    # Blend with cached tuned probs (0.3/0.7 like baseline)
    tuned = np.load(ROOT / "tuned_probs.npy")
    ens = 0.3 * tuned + 0.7 * probs
    order = np.argsort(-ens)
    pos = set(order[:TOP_N].tolist())
    out = ROOT / f"submission_ext_t3h7_top{TOP_N}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for i, (tid, _, _) in enumerate(test_pairs):
            w.writerow([tid, 1 if i in pos else 0])

    # Compare to baseline
    hidden_base = np.load(ROOT / "hidden_probs.npy")
    base_ens = 0.3 * tuned + 0.7 * hidden_base
    base_top = set(np.argsort(-base_ens)[:TOP_N].tolist())
    print(f"\nOutput: {out.name}")
    print(f"  overlap with baseline (0.948) top-{TOP_N}: "
          f"{len(pos & base_top)}/{TOP_N}  flips={TOP_N - len(pos & base_top)}")

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
