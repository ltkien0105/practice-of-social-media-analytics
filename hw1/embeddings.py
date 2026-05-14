"""SVD matrix-factorisation embeddings for directed link prediction.

Builds a low-rank (k=64) decomposition of the adjacency matrix A ≈ U Σ V^T.
For pair (u, v):
  - source embedding of u  = U[u] * sqrt(Σ)   → captures "what u follows"
  - target embedding of v  = V[v] * sqrt(Σ)   → captures "who v is followed by"
  - dot product ≈ A_hat[u, v]   (predicted edge probability)
"""

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def build_svd_embeddings(
    G: nx.DiGraph, k: int = 64
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Return (src_emb, dst_emb, zero_vec) dicts keyed by node id."""
    nodes = sorted(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    src_list, dst_list = zip(*G.edges())
    row_idx = [node2idx[u] for u in src_list]
    col_idx = [node2idx[v] for v in dst_list]
    A = sp.csr_matrix(
        (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
        shape=(N, N),
    )

    k_actual = min(k, N - 2)
    # svds returns singular values in ascending order — flip to descending
    U, S, Vt = svds(A, k=k_actual)
    order = np.argsort(S)[::-1]
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    sqrt_S   = np.sqrt(S)
    src_emb  = U * sqrt_S        # N × k  (source perspective)
    dst_emb  = Vt.T * sqrt_S     # N × k  (target perspective)

    zero_vec = np.zeros(k_actual, dtype=np.float32)
    src_dict = {n: src_emb[i].astype(np.float32) for n, i in node2idx.items()}
    dst_dict = {n: dst_emb[i].astype(np.float32) for n, i in node2idx.items()}
    return src_dict, dst_dict, zero_vec


def compute_svd_features(
    pairs: list[tuple[int, int]],
    src_dict: dict,
    dst_dict: dict,
    zero_vec: np.ndarray,
    prefix: str = "svd",
) -> pd.DataFrame:
    """Return DataFrame with dot, l2, cosine, summary stats + Hadamard features."""
    k = len(zero_vec)
    rows = []
    for u, v in pairs:
        eu = src_dict.get(u, zero_vec)
        ev = dst_dict.get(v, zero_vec)
        hadamard = eu * ev
        dot      = float(np.dot(eu, ev))
        diff     = eu - ev
        l2       = float(np.linalg.norm(diff))
        norm_u   = float(np.linalg.norm(eu))
        norm_v   = float(np.linalg.norm(ev))
        cosine   = dot / (norm_u * norm_v + 1e-10)

        had_abs = np.abs(hadamard)
        had_mean     = float(np.mean(hadamard))
        had_std      = float(np.std(hadamard))
        had_abs_mean = float(np.mean(had_abs))
        had_abs_max  = float(np.max(had_abs)) if k > 0 else 0.0

        rows.append([
            dot, l2, cosine, norm_u, norm_v,
            had_mean, had_std, had_abs_mean, had_abs_max,
        ] + hadamard.tolist())

    cols = [
        f"{prefix}_dot", f"{prefix}_l2", f"{prefix}_cos",
        f"{prefix}_src_norm", f"{prefix}_dst_norm",
        f"{prefix}_had_mean", f"{prefix}_had_std",
        f"{prefix}_had_abs_mean", f"{prefix}_had_abs_max",
    ] + [f"{prefix}_h{i}" for i in range(k)]
    return pd.DataFrame(rows, columns=cols, dtype=np.float32)
