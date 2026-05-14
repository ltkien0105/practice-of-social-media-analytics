"""
Link prediction — directed social graph.

Approach B: graph features (27) + SVD embeddings (k=256), random negatives, LGBM.
Synced with main_1.py: adjacency caching, capped negative sampling, and
fold-averaged test predictions.

Usage:  uv run python main_2.py
"""

import sys
import io
import math
import random

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── graph features ────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # neighbourhood
    "out_in_common", "in_out_common", "common_succ", "common_pred",
    "jaccard_out_in", "adamic_adar", "resource_alloc_dir",
    "triadic_closure", "reverse_exists", "common_all",
    # degree
    "u_out_deg", "u_in_deg", "v_out_deg", "v_in_deg",
    "u_total_deg", "v_total_deg",
    "pref_attach", "pref_attach_total", "u_follow_ratio", "v_follow_ratio",
    # global rank
    "u_pagerank", "v_pagerank",
    "u_hub", "u_auth", "v_hub", "v_auth",
    # community
    "same_community",
]


def build_global_scores(G: nx.DiGraph) -> tuple[dict, dict, dict, dict]:
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    print("  Computing HITS...")
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
    print("  Computing Louvain communities...")
    communities = nx.community.louvain_communities(G.to_undirected(), seed=42)
    node_community: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_community[n] = cid
    return pagerank, hubs, authorities, node_community


def build_adjacency_cache(G: nx.DiGraph) -> dict:
    """Pre-compute neighbour sets and degrees for O(1) lookup during feature extraction."""
    return {
        "out_nbrs": {n: set(G.successors(n))   for n in G.nodes()},
        "in_nbrs":  {n: set(G.predecessors(n)) for n in G.nodes()},
        "out_deg":  dict(G.out_degree()),
        "in_deg":   dict(G.in_degree()),
        "edge_set": set(G.edges()),
    }


def compute_graph_features(
    pairs: list[tuple[int, int]],
    cache: dict,
    pagerank: dict, hubs: dict, authorities: dict, node_community: dict,
    desc: str = "graph features",
) -> pd.DataFrame:
    out_nbrs = cache["out_nbrs"]
    in_nbrs  = cache["in_nbrs"]
    out_deg  = cache["out_deg"]
    in_deg   = cache["in_deg"]
    edge_set = cache["edge_set"]
    empty    = set()

    rows = []
    for u, v in tqdm(pairs, desc=desc, miniters=2000):
        out_u = out_nbrs.get(u, empty)
        in_u  = in_nbrs.get(u,  empty)
        out_v = out_nbrs.get(v, empty)
        in_v  = in_nbrs.get(v,  empty)

        out_in      = out_u & in_v
        in_out      = in_u & out_v
        common_succ = out_u & out_v
        common_pred = in_u & in_v

        union_out_in = out_u | in_v
        jaccard = len(out_in) / len(union_out_in) if union_out_in else 0.0

        aa = ra = 0.0
        for w in out_in:
            deg = out_deg.get(w, 0) + in_deg.get(w, 0)
            if deg > 1:
                aa += 1.0 / math.log(deg)
            out_w = out_deg.get(w, 0)
            if out_w > 0:
                ra += 1.0 / out_w

        u_out = out_deg.get(u, 0)
        u_in  = in_deg.get(u,  0)
        v_out = out_deg.get(v, 0)
        v_in  = in_deg.get(v,  0)
        u_total = u_out + u_in
        v_total = v_out + v_in

        common_all   = len(out_in) + len(in_out) + len(common_succ) + len(common_pred)
        triadic      = len(out_in) / u_out if u_out > 0 else 0.0
        u_follow_rat = u_out / (u_in + 1)
        v_follow_rat = v_out / (v_in + 1)

        cu = node_community.get(u, -1)
        cv = node_community.get(v, -2)

        rows.append([
            len(out_in), len(in_out), len(common_succ), len(common_pred),
            jaccard, aa, ra, triadic, int((v, u) in edge_set), common_all,
            u_out, u_in, v_out, v_in, u_total, v_total,
            u_out * v_in, u_total * v_total, u_follow_rat, v_follow_rat,
            pagerank.get(u, 0.0), pagerank.get(v, 0.0),
            hubs.get(u, 0.0), authorities.get(u, 0.0),
            hubs.get(v, 0.0), authorities.get(v, 0.0),
            int(cu == cv and cu != -1),
        ])
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ── SVD embeddings ────────────────────────────────────────────────────────────

def build_svd_embeddings(
    G: nx.DiGraph, k: int = 256, seed: int = 42
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
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
    U, S, Vt = svds(A, k=k_actual)
    order = np.argsort(S)[::-1]
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    sqrt_S  = np.sqrt(S)
    src_emb = U * sqrt_S
    dst_emb = Vt.T * sqrt_S

    zero_vec = np.zeros(k_actual, dtype=np.float32)
    src_dict = {n: src_emb[i].astype(np.float32) for n, i in node2idx.items()}
    dst_dict = {n: dst_emb[i].astype(np.float32) for n, i in node2idx.items()}
    return src_dict, dst_dict, zero_vec


def compute_svd_features(
    pairs: list[tuple[int, int]],
    src_dict: dict, dst_dict: dict, zero_vec: np.ndarray,
    prefix: str = "svd",
) -> pd.DataFrame:
    k = len(zero_vec)
    rows = []
    for u, v in pairs:
        eu = src_dict.get(u, zero_vec)
        ev = dst_dict.get(v, zero_vec)
        hadamard = eu * ev
        dot    = float(np.dot(eu, ev))
        l2     = float(np.linalg.norm(eu - ev))
        norm_u = float(np.linalg.norm(eu))
        norm_v = float(np.linalg.norm(ev))
        cosine = dot / (norm_u * norm_v + 1e-10)

        had_abs = np.abs(hadamard)
        rows.append([
            dot, l2, cosine, norm_u, norm_v,
            float(np.mean(hadamard)), float(np.std(hadamard)),
            float(np.mean(had_abs)), float(np.max(had_abs)) if k > 0 else 0.0,
        ] + hadamard.tolist())

    cols = [
        f"{prefix}_dot", f"{prefix}_l2", f"{prefix}_cos",
        f"{prefix}_src_norm", f"{prefix}_dst_norm",
        f"{prefix}_had_mean", f"{prefix}_had_std",
        f"{prefix}_had_abs_mean", f"{prefix}_had_abs_max",
    ] + [f"{prefix}_h{i}" for i in range(k)]
    return pd.DataFrame(rows, columns=cols, dtype=np.float32)


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_random_negatives(G: nx.DiGraph, n: int, seed: int = 42) -> list[tuple]:
    """Random non-edges. Capped at 20× attempts to avoid pathological infinite loops."""
    nodes    = list(G.nodes())
    edge_set = set(G.edges())
    rng      = random.Random(seed)
    negs: list[tuple] = []
    attempts = 0
    max_attempts = n * 20
    while len(negs) < n and attempts < max_attempts:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and (u, v) not in edge_set:
            negs.append((u, v))
        attempts += 1
    return negs


# ── model ─────────────────────────────────────────────────────────────────────

def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=255,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Loading data ===")
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")
    G = nx.DiGraph()
    G.add_edges_from(zip(train_df["Node1"], train_df["Node2"]))
    pos_pairs  = list(G.edges())
    test_pairs = list(zip(test_df["Node1"], test_df["Node2"]))
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Test pairs: {len(test_df)}")

    print("\n=== Building adjacency cache ===")
    cache = build_adjacency_cache(G)

    print("\n=== Pre-computing global scores ===")
    pagerank, hubs, authorities, node_community = build_global_scores(G)

    print("\n=== Building SVD embeddings (k=256) ===")
    src_emb, dst_emb, zero_vec = build_svd_embeddings(G, k=256)

    print("\n=== Sampling negatives ===")
    neg_rand = sample_random_negatives(G, len(pos_pairs))
    pairs    = pos_pairs + neg_rand
    y        = np.array([1] * len(pos_pairs) + [0] * len(neg_rand), dtype=np.int8)

    print("\n=== Building train features ===")
    X_graph = compute_graph_features(
        pairs, cache, pagerank, hubs, authorities, node_community, desc="train graph"
    )
    X_svd  = compute_svd_features(pairs, src_emb, dst_emb, zero_vec)
    X_train = pd.concat([X_graph, X_svd], axis=1)

    print("\n=== Building test features ===")
    X_test_graph = compute_graph_features(
        test_pairs, cache, pagerank, hubs, authorities, node_community, desc="test graph"
    )
    X_test_svd = compute_svd_features(test_pairs, src_emb, dst_emb, zero_vec)
    X_test     = pd.concat([X_test_graph, X_test_svd], axis=1)

    print("\n=== Training with 5-fold CV (fold-averaged predictions) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = make_lgbm()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds         += model.predict_proba(X_test)[:, 1] / skf.n_splits

        auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold+1}/5  AUC = {auc:.5f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n  OOF AUC: {overall_auc:.5f}")

    print("\n=== Saving submission ===")
    sub_df = pd.read_csv("sample_submission.csv")
    sub_df["Label"] = test_preds
    sub_df.to_csv("M11415803_Le_Trung_Kien_B.csv", index=False)
    print(f"Saved M11415803_Le_Trung_Kien_B.csv  ({len(sub_df)} rows)")
    print(sub_df.head())


if __name__ == "__main__":
    main()
