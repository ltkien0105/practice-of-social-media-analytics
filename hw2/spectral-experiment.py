"""
Spectral embedding experiment — linear, global structure.

For each node, learn a k-dim embedding from the top-k eigenvectors of the
normalized adjacency matrix (Â = D^(-1/2) A D^(-1/2)). Captures global
community structure in a way Node2Vec's local random walks cannot.

Predictors:
- cosine similarity → top-N submission
- hadamard + GBM ensemble (matching node2vec-experiment recipe)
- linear / rank blend with baseline (tuned + hidden) probs

Run:
    uv run python spectral-experiment.py
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from pathlib import Path

import community as community_louvain  # noqa: F401  (kept parallel to n2v script)
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

SEED = 42
K = 256                    # number of spectral eigenvectors
EMB_PATH = ROOT / f"spec_embeddings_k{K}.npy"
NODE_INDEX_PATH = ROOT / f"spec_node_index_k{K}.pkl"
TAG = f"speck{K}"


def load_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with TRAIN_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            u, v = int(row[0]), int(row[1])
            if u != v:
                edges.append((u, v))
    return edges


def load_test_pairs() -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    with TEST_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            pairs.append((int(row[0]), int(row[1]), int(row[2])))
    return pairs


def write_submission(preds: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        w.writerows(preds)


def build_spectral(edges: list[tuple[int, int]]) -> tuple[np.ndarray, dict[int, int]]:
    if EMB_PATH.exists() and NODE_INDEX_PATH.exists():
        print(f"Loading cached embeddings from {EMB_PATH.name}...")
        emb = np.load(EMB_PATH)
        with NODE_INDEX_PATH.open("rb") as f:
            idx = pickle.load(f)
        return emb, idx

    print("Building sparse adjacency...")
    nodes_set: set[int] = set()
    for u, v in edges:
        nodes_set.add(u); nodes_set.add(v)
    nodes = sorted(nodes_set)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows = np.array([idx[u] for u, _ in edges] + [idx[v] for _, v in edges])
    cols = np.array([idx[v] for _, v in edges] + [idx[u] for u, _ in edges])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T).sign()  # ensure symmetric, binary
    print(f"  nodes={n} nnz={A.nnz // 2}")

    print("Computing degree-normalized adjacency...")
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    print(f"Computing top-{K} eigenvectors of normalized adjacency "
          f"(largest magnitude)...")
    t0 = time.time()
    vals, vecs = spla.eigsh(A_norm, k=K, which="LA", tol=1e-4, maxiter=2000)
    print(f"  eigsh done in {time.time() - t0:.1f}s; "
          f"vals[0]={vals[0]:.3f} vals[-1]={vals[-1]:.3f}")

    emb = vecs.astype(np.float32)
    np.save(EMB_PATH, emb)
    with NODE_INDEX_PATH.open("wb") as f:
        pickle.dump(idx, f)
    print(f"  saved -> {EMB_PATH.name} ({emb.shape})")
    return emb, idx


def cosine_sims(emb: np.ndarray, idx: dict[int, int],
                pairs: list[tuple[int, int, int]]) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1)
    sims = np.zeros(len(pairs))
    for i, (_, u, v) in enumerate(pairs):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        nu, nv = norms[iu], norms[iv]
        if nu == 0 or nv == 0:
            continue
        sims[i] = float(np.dot(emb[iu], emb[iv]) / (nu * nv))
    return sims


def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=SEED)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def component_map(g: nx.Graph) -> dict[int, int]:
    cm: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            cm[n] = cid
    return cm


def cross_leiden_negatives(edges, leiden_of, comp_of, n_neg, rng):
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


def hadamard(emb, idx, pairs):
    out = np.zeros((len(pairs), emb.shape[1]), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        out[i] = emb[iu] * emb[iv]
    return out


def train_classifier(emb, idx, edges, leiden_of, comp_of, test_pairs,
                     n_ensemble=6):
    test_pp = [(n1, n2) for _, n1, n2 in test_pairs]
    Xtest = hadamard(emb, idx, test_pp)
    probs = np.zeros(len(test_pairs))
    for k in range(n_ensemble):
        t0 = time.time()
        seed = SEED + k * 17
        rng = random.Random(seed)
        pos = rng.sample(edges, 5000)
        neg = cross_leiden_negatives(edges, leiden_of, comp_of, 5000, rng)
        X = hadamard(emb, idx, pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs += gb.predict_proba(Xtest)[:, 1]
        print(f"  spec-clf member {k + 1}/{n_ensemble} "
              f"({time.time() - t0:.1f}s)")
    return probs / n_ensemble


def topn_submission(scores, test_pairs, top_n, name):
    order = np.argsort(-scores)
    pos = set(order[:top_n].tolist())
    path = ROOT / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for i, (tid, _, _) in enumerate(test_pairs):
            w.writerow([tid, 1 if i in pos else 0])
    print(f"  positives={len(pos)} -> {path.name}")


def main():
    t_total = time.time()
    edges = load_edges()
    test_pairs = load_test_pairs()
    print(f"edges={len(edges)} test_pairs={len(test_pairs)}")

    emb, idx = build_spectral(edges)
    print(f"embeddings: {emb.shape}")

    g = nx.Graph(); g.add_edges_from(edges)
    comp_of = component_map(g)
    leiden_of = build_leiden(edges)
    print(f"leiden communities={len(set(leiden_of.values()))}")

    print("\nCosine similarity scores...")
    sims = cosine_sims(emb, idx, test_pairs)
    print(f"  sim stats: min={sims.min():.3f} max={sims.max():.3f} "
          f"mean={sims.mean():.3f}")
    for top in (440, 460, 480, 500):
        topn_submission(sims, test_pairs, top,
                        f"submission_{TAG}_cos_top{top}.csv")

    print("\nHadamard + GBM classifier ensemble...")
    spec_probs = train_classifier(emb, idx, edges, leiden_of, comp_of,
                                  test_pairs, n_ensemble=6)
    np.save(ROOT / f"{TAG}_probs.npy", spec_probs)
    for top in (440, 460, 480, 500):
        topn_submission(spec_probs, test_pairs, top,
                        f"submission_{TAG}_clf_top{top}.csv")

    tuned_path = ROOT / "tuned_probs.npy"
    hidden_path = ROOT / "hidden_probs.npy"
    if tuned_path.exists() and hidden_path.exists():
        print("\nBlending baseline + spectral cosine at multiple weights...")
        tuned = np.load(tuned_path)
        hidden = np.load(hidden_path)
        base = 0.3 * tuned + 0.7 * hidden
        rank_base = np.argsort(np.argsort(base)).astype(float) / (len(base) - 1)
        rank_cos = np.argsort(np.argsort(sims)).astype(float) / (len(sims) - 1)
        for w in (0.10, 0.20, 0.30, 0.50):
            ens = (1 - w) * base + w * sims
            topn_submission(ens, test_pairs, 460,
                            f"submission_blend_{TAG}cos{int(w * 100):02d}_top460.csv")
        for w in (0.30, 0.50):
            ens = (1 - w) * rank_base + w * rank_cos
            topn_submission(ens, test_pairs, 460,
                            f"submission_rankblend_{TAG}cos{int(w * 100):02d}_top460.csv")
    else:
        print("\nSkip blend; run main.py to cache baseline probs first.")

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
