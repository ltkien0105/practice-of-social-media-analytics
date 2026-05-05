"""
Link prediction — directed social graph.

Approach B: graph features (27) + SVD embeddings (k=256), random negatives, LGBM.
Achieved 0.9994 ROC AUC (3-fold CV).

Usage:  uv run python main.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def compute_graph_features(
    pairs: list[tuple[int, int]],
    G: nx.DiGraph,
    pagerank: dict, hubs: dict, authorities: dict, node_community: dict,
    desc: str = "graph features",
) -> pd.DataFrame:
    rows = []
    for u, v in tqdm(pairs, desc=desc, miniters=2000):
        out_u = set(G.successors(u)) if G.has_node(u) else set()
        in_u  = set(G.predecessors(u)) if G.has_node(u) else set()
        out_v = set(G.successors(v)) if G.has_node(v) else set()
        in_v  = set(G.predecessors(v)) if G.has_node(v) else set()

        out_in      = out_u & in_v
        in_out      = in_u & out_v
        common_succ = out_u & out_v
        common_pred = in_u & in_v

        union_out_in = out_u | in_v
        jaccard = len(out_in) / len(union_out_in) if union_out_in else 0.0

        aa = ra = 0.0
        for w in out_in:
            deg = G.out_degree(w) + G.in_degree(w)
            if deg > 1:
                aa += 1.0 / math.log(deg)
            out_w = G.out_degree(w)
            if out_w > 0:
                ra += 1.0 / out_w

        u_out = G.out_degree(u) if G.has_node(u) else 0
        u_in  = G.in_degree(u)  if G.has_node(u) else 0
        v_out = G.out_degree(v) if G.has_node(v) else 0
        v_in  = G.in_degree(v)  if G.has_node(v) else 0
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
            jaccard, aa, ra, triadic, int(G.has_edge(v, u)), common_all,
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
    G: nx.DiGraph, k: int = 256,
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
    nodes    = list(G.nodes())
    edge_set = set(G.edges())
    rng      = random.Random(seed)
    negs: list[tuple] = []
    while len(negs) < n:
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and (u, v) not in edge_set:
            negs.append((u, v))
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

    print("\n=== Pre-computing global scores ===")
    pagerank, hubs, authorities, node_community = build_global_scores(G)

    print("\n=== Building SVD embeddings (k=256) ===")
    src_emb, dst_emb, zero_vec = build_svd_embeddings(G, k=256)

    print("\n=== Sampling negatives ===")
    neg_rand = sample_random_negatives(G, len(pos_pairs))
    pairs    = pos_pairs + neg_rand
    y        = np.array([1] * len(pos_pairs) + [0] * len(neg_rand), dtype=np.int8)

    print("\n=== Building features ===")
    X_graph = compute_graph_features(
        pairs, G, pagerank, hubs, authorities, node_community, desc="train graph"
    )
    X_svd  = compute_svd_features(pairs, src_emb, dst_emb, zero_vec)
    X_train = pd.concat([X_graph, X_svd], axis=1)

    print("\n=== Cross-validating ===")
    skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(make_lgbm(), X_train, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"   CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

    print("\n=== Building test features ===")
    X_test_graph = compute_graph_features(
        test_pairs, G, pagerank, hubs, authorities, node_community, desc="test graph"
    )
    X_test_svd = compute_svd_features(test_pairs, src_emb, dst_emb, zero_vec)
    X_test     = pd.concat([X_test_graph, X_test_svd], axis=1)

    print("\n=== Training final model ===")
    clf = make_lgbm()
    clf.fit(X_train, y)
    probs = clf.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"ID": test_df["ID"], "Label": probs})
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv  ({len(submission)} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
