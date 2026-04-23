"""
Link prediction — directed social graph.

Tests four feature/sampling combinations and one ensemble approach,
reports 3-fold CV ROC AUC for each, then saves submission.csv from
the highest-scoring approach.

Usage:  uv run python main.py
"""

import sys
import io

import numpy as np
import pandas as pd
import networkx as nx

from features   import build_global_scores, compute_graph_features
from embeddings import build_svd_embeddings, compute_svd_features
from sampling   import sample_random_negatives, sample_hard_negatives, sample_mixed_negatives
from models     import (make_lgbm, make_hgb, make_rf, cv_auc,
                         ensemble_oof_auc, train_and_predict_ensemble)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── helpers ───────────────────────────────────────────────────────────────────

def load_graph(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    G  = nx.DiGraph()
    G.add_edges_from(zip(df["Node1"], df["Node2"]))
    return G


def build_feature_matrix(
    pairs, G, pagerank, hubs, authorities, node_community,
    src_emb, dst_emb, zero_vec, include_svd: bool, desc: str = "features"
) -> pd.DataFrame:
    X_graph = compute_graph_features(
        pairs, G, pagerank, hubs, authorities, node_community, desc=desc
    )
    if not include_svd:
        return X_graph
    X_svd = compute_svd_features(pairs, src_emb, dst_emb, zero_vec)
    return pd.concat([X_graph, X_svd], axis=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Loading data ===")
    G       = load_graph("train.csv")
    test_df = pd.read_csv("test.csv")
    test_pairs = list(zip(test_df["Node1"], test_df["Node2"]))
    pos_pairs  = list(G.edges())
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Test pairs: {len(test_df)}")

    print("\n=== Pre-computing global scores ===")
    pagerank, hubs, authorities, node_community = build_global_scores(G)

    SVD_K = 256
    print(f"\n=== Building SVD embeddings (k={SVD_K}) ===")
    src_emb, dst_emb, zero_vec = build_svd_embeddings(G, k=SVD_K)

    print("\n=== Generating negative samples ===")
    neg_rand  = sample_random_negatives(G, len(pos_pairs))
    neg_mixed = sample_mixed_negatives(G, len(pos_pairs), hard_ratio=0.3)
    y_base    = np.array([1] * len(pos_pairs) + [0] * len(pos_pairs), dtype=np.int8)

    def pairs_y(negs):
        return pos_pairs + negs, y_base

    results: dict[str, tuple[float, float]] = {}

    # ── A: graph features only, random negatives ──────────────────────────────
    print("\n[A] Graph features + random negatives")
    tr_pairs_a, y_a = pairs_y(neg_rand)
    X_a = build_feature_matrix(
        tr_pairs_a, G, pagerank, hubs, authorities, node_community,
        src_emb, dst_emb, zero_vec, include_svd=False, desc="A graph"
    )
    results["A  graph-only  rand-neg"] = cv_auc(make_lgbm(), X_a, y_a)
    print(f"   CV AUC: {results['A  graph-only  rand-neg'][0]:.4f} +/- {results['A  graph-only  rand-neg'][1]:.4f}")

    # ── B: graph + SVD, random negatives ─────────────────────────────────────
    print("\n[B] Graph + SVD features + random negatives")
    tr_pairs_b, y_b = pairs_y(neg_rand)
    X_b = build_feature_matrix(
        tr_pairs_b, G, pagerank, hubs, authorities, node_community,
        src_emb, dst_emb, zero_vec, include_svd=True, desc="B graph+SVD"
    )
    results["B  graph+SVD   rand-neg"] = cv_auc(make_lgbm(), X_b, y_b)
    print(f"   CV AUC: {results['B  graph+SVD   rand-neg'][0]:.4f} +/- {results['B  graph+SVD   rand-neg'][1]:.4f}")

    # ── C: graph + SVD, mixed negatives (30% hard + 70% random) ────────────
    print("\n[C] Graph + SVD features + mixed (30% hard + 70% random) negatives")
    tr_pairs_c, y_c = pairs_y(neg_mixed)
    X_c = build_feature_matrix(
        tr_pairs_c, G, pagerank, hubs, authorities, node_community,
        src_emb, dst_emb, zero_vec, include_svd=True, desc="C graph+SVD+mixed"
    )
    results["C  graph+SVD   mixed-neg"] = cv_auc(make_lgbm(), X_c, y_c)
    print(f"   CV AUC: {results['C  graph+SVD   mixed-neg'][0]:.4f} +/- {results['C  graph+SVD   mixed-neg'][1]:.4f}")

    # ── D/E: ensembles on best single-model training set ────────────────────
    best_single = max("B  graph+SVD   rand-neg", "C  graph+SVD   mixed-neg",
                      key=lambda k: results[k][0])
    X_d  = X_b if "B" in best_single else X_c
    y_d  = y_b if "B" in best_single else y_c

    print(f"\n[D] Ensemble (LGBM+HGB+RF), proper OOF on ({best_single.strip()})")
    ens_auc_3, per_auc_3 = ensemble_oof_auc(
        [make_lgbm, make_hgb, make_rf], X_d, y_d, cv=3
    )
    auc_lgbm, auc_hgb, auc_rf = per_auc_3
    print(f"   LGBM: {auc_lgbm:.4f}  HGB: {auc_hgb:.4f}  RF: {auc_rf:.4f}")
    print(f"   ENSEMBLE-3 OOF: {ens_auc_3:.4f}")
    results["D  ensemble-3   best-neg"] = (ens_auc_3, 0.0)

    print(f"\n[E] Ensemble (LGBM+HGB only), proper OOF on ({best_single.strip()})")
    ens_auc_2, per_auc_2 = ensemble_oof_auc(
        [make_lgbm, make_hgb], X_d, y_d, cv=3
    )
    print(f"   LGBM: {per_auc_2[0]:.4f}  HGB: {per_auc_2[1]:.4f}")
    print(f"   ENSEMBLE-2 OOF: {ens_auc_2:.4f}")
    results["E  ensemble-2   best-neg"] = (ens_auc_2, 0.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== CV AUC Summary ===")
    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
    for name, (auc, std) in ranked:
        marker = " << BEST" if name == ranked[0][0] else ""
        print(f"  {name}:  {auc:.4f} +/- {std:.4f}{marker}")

    best_name = ranked[0][0]

    # ── Final training + prediction ───────────────────────────────────────────
    print(f"\n=== Training final model ({best_name.strip()}) ===")

    if best_name.startswith(("D", "E")):
        X_final_train, y_final = X_d, y_d
    elif "C" in best_name:
        X_final_train, y_final = X_c, y_c
    else:
        X_final_train, y_final = X_b, y_b

    X_test = build_feature_matrix(
        test_pairs, G, pagerank, hubs, authorities, node_community,
        src_emb, dst_emb, zero_vec, include_svd=True, desc="test"
    )

    if best_name.startswith("D"):
        probs = train_and_predict_ensemble(
            [make_lgbm(), make_hgb(), make_rf()],
            X_final_train, y_final, X_test,
            weights=[auc_lgbm, auc_hgb, auc_rf],
        )
    elif best_name.startswith("E"):
        probs = train_and_predict_ensemble(
            [make_lgbm(), make_hgb()],
            X_final_train, y_final, X_test,
            weights=[per_auc_2[0], per_auc_2[1]],
        )
    else:
        clf = make_lgbm()
        clf.fit(X_final_train, y_final)
        probs = clf.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"ID": test_df["ID"], "Label": probs})
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv  ({len(submission)} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
