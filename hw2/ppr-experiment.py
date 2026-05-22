"""
Personalized PageRank (PPR) link-prediction experiment.

For each test pair (u, v), compute:
- ppr_uv = PPR(u -> v): random walker from u, restart α, hits v
- ppr_vu = PPR(v -> u): symmetric
- ppr_sym = (ppr_uv + ppr_vu) / 2  (symmetric score)

Captures multi-hop reachability — qualitatively different from static
embedding cosines (n2v / spec) which all correlate strongly with the
baseline ensemble.

Three predictors:
1. ppr_sym alone (top-N)
2. Add ppr features to hidden-classifier (14-dim version) + retrain
3. Blend baseline + ppr_sym at multiple weights

Run:
    uv run python ppr-experiment.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
PPR_CACHE = ROOT / "ppr_scores.npy"  # (N_test, 2): [ppr_uv, ppr_vu]

SEED = 42
ALPHA = 0.85
PPR_ITERS = 20
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


def build_sparse_transition(edges):
    nodes_set = set()
    for u, v in edges:
        nodes_set.add(u); nodes_set.add(v)
    nodes = sorted(nodes_set)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows = np.array([idx[u] for u, _ in edges] + [idx[v] for _, v in edges])
    cols = np.array([idx[v] for _, v in edges] + [idx[u] for u, _ in edges])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T).sign()
    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.where(deg > 0, deg, 1)
    P = sp.diags(1.0 / deg_safe) @ A
    return P.tocsr(), idx, n


def ppr_vector(P, source, n, alpha=ALPHA, iters=PPR_ITERS):
    """Power iteration for personalized PageRank rooted at source."""
    e = np.zeros(n, dtype=np.float32)
    e[source] = 1.0
    r = e.copy()
    for _ in range(iters):
        r = alpha * (P.T @ r) + (1.0 - alpha) * e
    return r


def compute_ppr_scores(edges, test_pairs):
    if PPR_CACHE.exists():
        print(f"Loading cached PPR from {PPR_CACHE.name}...")
        return np.load(PPR_CACHE)

    print("Building transition matrix...")
    P, idx, n = build_sparse_transition(edges)
    print(f"  n={n} nnz={P.nnz}")

    # Find unique endpoint nodes among test pairs
    endpoints = set()
    for _, u, v in test_pairs:
        endpoints.add(u); endpoints.add(v)
    print(f"Unique test endpoints: {len(endpoints)}")

    print(f"Computing PPR vectors (alpha={ALPHA}, iters={PPR_ITERS})...")
    t0 = time.time()
    ppr_of: dict[int, np.ndarray] = {}
    for i, node in enumerate(endpoints):
        src = idx.get(node)
        if src is None:
            ppr_of[node] = None
            continue
        ppr_of[node] = ppr_vector(P, src, n)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(endpoints)} ({time.time() - t0:.1f}s)")
    print(f"  done ({time.time() - t0:.1f}s)")

    print("Extracting PPR scores per test pair...")
    scores = np.zeros((len(test_pairs), 2), dtype=np.float32)
    for i, (_, u, v) in enumerate(test_pairs):
        iu = idx.get(u); iv = idx.get(v)
        if iu is None or iv is None:
            continue
        if ppr_of[u] is not None:
            scores[i, 0] = ppr_of[u][iv]
        if ppr_of[v] is not None:
            scores[i, 1] = ppr_of[v][iu]
    np.save(PPR_CACHE, scores)
    print(f"  saved -> {PPR_CACHE.name}")
    return scores


def topn_write(scores, test_pairs, name, n=TOP_N):
    order = np.argsort(-scores)
    pos = set(order[:n].tolist())
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

    ppr_scores = compute_ppr_scores(edges, test_pairs)
    ppr_uv = ppr_scores[:, 0]
    ppr_vu = ppr_scores[:, 1]
    ppr_sym = (ppr_uv + ppr_vu) / 2
    print(f"\nppr_sym stats: min={ppr_sym.min():.3e} "
          f"max={ppr_sym.max():.3e} mean={ppr_sym.mean():.3e}")

    # Correlation with baseline
    tuned = np.load(ROOT / "tuned_probs.npy")
    hidden = np.load(ROOT / "hidden_probs.npy")
    base = 0.3 * tuned + 0.7 * hidden
    print(f"corr(ppr_sym, baseline) = {np.corrcoef(ppr_sym, base)[0,1]:.3f}")
    print(f"corr(ppr_sym, hidden)   = {np.corrcoef(ppr_sym, hidden)[0,1]:.3f}")
    ts_base = set(np.argsort(-base)[:TOP_N].tolist())

    # 1) PPR alone top-N
    print("\nPPR-sym alone:")
    for top in (440, 460, 480, 500):
        topn_write(ppr_sym, test_pairs, f"submission_ppr_top{top}.csv", n=top)

    # 2) Blend baseline + ppr_sym
    print("\nBaseline + ppr_sym blends:")
    # Normalize ppr_sym to [0,1] for blending
    ppr_norm = (ppr_sym - ppr_sym.min()) / max(1e-12, ppr_sym.max() - ppr_sym.min())
    rank_base = np.argsort(np.argsort(base)).astype(float) / (len(base) - 1)
    rank_ppr = np.argsort(np.argsort(ppr_sym)).astype(float) / (len(ppr_sym) - 1)
    for w in (0.10, 0.20, 0.30, 0.50):
        ens = (1 - w) * base + w * ppr_norm
        topn_write(ens, test_pairs, f"submission_blend_pprcos{int(w*100):02d}_top460.csv")
    for w in (0.30, 0.50):
        ens = (1 - w) * rank_base + w * rank_ppr
        topn_write(ens, test_pairs, f"submission_rankblend_pprcos{int(w*100):02d}_top460.csv")

    # Report overlap of best candidates with baseline
    print("\nOverlap with baseline top-460:")
    for w in (0.10, 0.20, 0.30, 0.50):
        ens = (1 - w) * base + w * ppr_norm
        s = set(np.argsort(-ens)[:TOP_N].tolist())
        print(f"  blend_pprcos{int(w*100):02d}: {len(s & ts_base)}/460  "
              f"flips={TOP_N - len(s & ts_base)}")
    for w in (0.30, 0.50):
        ens = (1 - w) * rank_base + w * rank_ppr
        s = set(np.argsort(-ens)[:TOP_N].tolist())
        print(f"  rankblend_pprcos{int(w*100):02d}: {len(s & ts_base)}/460  "
              f"flips={TOP_N - len(s & ts_base)}")

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
