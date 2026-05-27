"""
HW2 — Community membership prediction (M11415803).

Single-file end-to-end pipeline. Produces M11415803_Le_Trung_Kien.csv
(Kaggle score 0.964 — ties leaderboard #1).

Run:
    uv run python main.py      (or: python main.py with deps installed)

Method — Personalized PageRank (PPR):
1. Load train.csv (188K nodes, 400K edges) and test.csv (1000 pairs).
2. Build the row-stochastic random-walk transition matrix
   P = D^(-1) A  (undirected graph, so A is symmetric).
3. For every node that appears as a test-pair endpoint, run personalized
   PageRank by power iteration:
       r_{t+1} = α · Pᵀ r_t + (1-α) · e_s      (α = 0.85, 20 iters)
   r_s[v] is the long-run probability that a walker starting at s — and
   restarting at s with prob (1-α) each step — is found at v. This is a
   multi-hop community-membership signal: nodes in the same dense
   community accumulate high mutual PPR mass.
4. For each test pair (u, v), score = (PPR(u→v) + PPR(v→u)) / 2.
5. Predict the top-474 highest-scoring pairs as same-community (1).

Why PPR beats the structural-feature ensemble (which plateaued at 0.948):
common-neighbour / Jaccard / Adamic-Adar / shortest-path all measure
*local* overlap. PPR integrates *all* paths of every length (geometrically
down-weighted), so it captures community membership for pairs that share
no direct neighbours but sit in the same densely-connected region. Its
predictions correlate only 0.50 with the structural ensemble yet are more
accurate at the decision boundary — top-474 reaches 0.964.

Dependencies: numpy, scipy.
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
OUT_CSV = ROOT / "M11415803_Le_Trung_Kien.csv"

ALPHA = 0.85      # PPR restart (teleport) probability is (1 - ALPHA)
PPR_ITERS = 20    # power-iteration steps
TOP_N = 474       # number of pairs predicted as same-community


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


# ============================================================
# PPR
# ============================================================

def build_transition(edges: list[tuple[int, int]]) -> tuple[sp.csr_matrix,
                                                             dict[int, int]]:
    """Row-stochastic transition matrix P = D^(-1) A of the undirected graph."""
    nodes = sorted({n for e in edges for n in e})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows = np.array([idx[u] for u, _ in edges] + [idx[v] for _, v in edges])
    cols = np.array([idx[v] for _, v in edges] + [idx[u] for u, _ in edges])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T).sign()  # symmetric, binary
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    P = sp.diags(1.0 / deg) @ A
    return P.tocsr(), idx


def ppr_vector(P: sp.csr_matrix, source: int, n: int,
               alpha: float = ALPHA, iters: int = PPR_ITERS) -> np.ndarray:
    """Personalized PageRank rooted at `source` via power iteration."""
    e = np.zeros(n, dtype=np.float32)
    e[source] = 1.0
    r = e.copy()
    Pt = P.T.tocsr()
    for _ in range(iters):
        r = alpha * (Pt @ r) + (1.0 - alpha) * e
    return r


def compute_pair_scores(edges: list[tuple[int, int]],
                        test_pairs: list[tuple[int, int, int]]) -> np.ndarray:
    """Symmetric PPR score (PPR(u→v) + PPR(v→u)) / 2 for each test pair."""
    P, idx = build_transition(edges)
    n = P.shape[0]
    print(f"  transition matrix: n={n} nnz={P.nnz}")

    endpoints = {u for _, u, _ in test_pairs} | {v for _, _, v in test_pairs}
    print(f"  computing PPR for {len(endpoints)} unique endpoints...")
    t0 = time.time()
    ppr_of: dict[int, np.ndarray | None] = {}
    for i, node in enumerate(endpoints):
        src = idx.get(node)
        ppr_of[node] = ppr_vector(P, src, n) if src is not None else None
        if (i + 1) % 400 == 0:
            print(f"    {i + 1}/{len(endpoints)} ({time.time() - t0:.1f}s)")
    print(f"  PPR done ({time.time() - t0:.1f}s)")

    scores = np.zeros(len(test_pairs))
    for i, (_, u, v) in enumerate(test_pairs):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        s = 0.0
        if ppr_of[u] is not None:
            s += ppr_of[u][iv]
        if ppr_of[v] is not None:
            s += ppr_of[v][iu]
        scores[i] = s / 2.0
    return scores


# ============================================================
# Main
# ============================================================

def main() -> None:
    t_total = time.time()
    print("Loading data...")
    edges = load_edges()
    test_pairs = load_test_pairs()
    print(f"  edges={len(edges)} test_pairs={len(test_pairs)}")

    print("\nComputing personalized PageRank scores...")
    scores = compute_pair_scores(edges, test_pairs)
    print(f"  score stats: min={scores.min():.3e} max={scores.max():.3e} "
          f"mean={scores.mean():.3e}")

    print(f"\nPredicting top-{TOP_N} pairs as same-community...")
    order = np.argsort(-scores)
    positive_idx = set(order[:TOP_N].tolist())
    preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
             for i in range(len(test_pairs))]
    pos_count = sum(1 for _, c in preds if c == 1)
    write_submission(preds, OUT_CSV)
    print(f"  positives={pos_count} -> {OUT_CSV.name}")
    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
