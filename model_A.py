import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("=== Loading data ===")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
sub_df   = pd.read_csv("sample_submission.csv")

print(f"Train edges: {len(train_df):,}")
print(f"Test pairs:  {len(test_df):,}")

# ── Build directed graph ──────────────────────────────────────────────────────
print("\n=== Building graph ===")
G = nx.DiGraph()
G.add_edges_from(zip(train_df["Node1"], train_df["Node2"]))
print(f"Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

# Undirected view for symmetric metrics
UG = G.to_undirected()

# Pre-compute neighbour sets for speed
out_nbrs  = {n: set(G.successors(n))   for n in G.nodes()}
in_nbrs   = {n: set(G.predecessors(n)) for n in G.nodes()}
un_nbrs   = {n: set(UG.neighbors(n))   for n in UG.nodes()}

out_deg   = dict(G.out_degree())
in_deg    = dict(G.in_degree())

# ── Negative sampling for training ───────────────────────────────────────────
print("\n=== Generating negative samples ===")
pos_set  = set(zip(train_df["Node1"], train_df["Node2"]))
nodes    = list(G.nodes())
rng      = np.random.default_rng(42)

neg_edges = set()
needed    = len(pos_set)
attempts  = 0
while len(neg_edges) < needed and attempts < needed * 20:
    u = int(rng.choice(nodes))
    v = int(rng.choice(nodes))
    if u != v and (u, v) not in pos_set and (u, v) not in neg_edges:
        neg_edges.add((u, v))
    attempts += 1

neg_list = list(neg_edges)
print(f"Positive: {len(pos_set):,}  Negative: {len(neg_list):,}")

# ── Feature engineering ───────────────────────────────────────────────────────
def get_features(pairs):
    feats = []
    for u, v in pairs:
        out_u  = out_nbrs.get(u, set())
        in_u   = in_nbrs.get(u,  set())
        out_v  = out_nbrs.get(v, set())
        in_v   = in_nbrs.get(v,  set())
        un_u   = un_nbrs.get(u,  set())
        un_v   = un_nbrs.get(v,  set())

        # Degree features
        out_u_deg = out_deg.get(u, 0)
        in_u_deg  = in_deg.get(u,  0)
        out_v_deg = out_deg.get(v, 0)
        in_v_deg  = in_deg.get(v,  0)

        # Common neighbours (directed variants)
        cn_out_out   = len(out_u  & out_v)   # u→w←v (common followees)
        cn_out_in    = len(out_u  & in_v)    # u→w, v←w
        cn_in_out    = len(in_u   & out_v)   # w→u, w→v
        cn_in_in     = len(in_u   & in_v)    # w→u, w→v followers
        cn_undirected= len(un_u   & un_v)

        # Jaccard (undirected)
        union_un = len(un_u | un_v)
        jaccard   = cn_undirected / union_un if union_un else 0

        # Adamic-Adar (undirected)
        aa = sum(1 / np.log(len(un_nbrs[w]) + 1e-9)
                 for w in un_u & un_v if len(un_nbrs.get(w, set())) > 1)

        # Preferential attachment
        pa_out = out_u_deg * in_v_deg
        pa_un  = len(un_u) * len(un_v)

        # Reciprocity: does v already follow u?
        reciprocal = int(v in out_nbrs.get(u, set()) or u in out_nbrs.get(v, set()))
        v_follows_u = int(u in out_v)

        # Path features: is there a path of length 2 u→w→v?
        path2 = len(out_u & in_v)   # same as cn_out_in above

        # Hub scores (simple)
        hub_u = out_u_deg / (in_u_deg + 1)
        hub_v = out_v_deg / (in_v_deg + 1)

        # Katz-like: fraction of u's followees that v also follows
        katz_proxy = cn_out_out / (out_u_deg + 1)

        feats.append([
            out_u_deg, in_u_deg, out_v_deg, in_v_deg,
            cn_out_out, cn_out_in, cn_in_out, cn_in_in, cn_undirected,
            jaccard, aa,
            pa_out, pa_un,
            reciprocal, v_follows_u,
            path2, hub_u, hub_v, katz_proxy,
        ])
    return np.array(feats, dtype=np.float32)

print("\n=== Engineering features ===")
pos_pairs = list(zip(train_df["Node1"], train_df["Node2"]))
X_pos  = get_features(pos_pairs)
X_neg  = get_features(neg_list)
X_train = np.vstack([X_pos, X_neg])
y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))

test_pairs = list(zip(test_df["Node1"], test_df["Node2"]))
print("  Featurising test pairs …")
X_test = get_features(test_pairs)
print(f"  Train shape: {X_train.shape}  Test shape: {X_test.shape}")

# ── Model training with cross-validation ─────────────────────────────────────
print("\n=== Training LightGBM ===")
lgb_params = dict(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds         += model.predict_proba(X_test)[:, 1] / skf.n_splits

    auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"  Fold {fold+1}/5  AUC = {auc:.5f}")

overall_auc = roc_auc_score(y_train, oof_preds)
print(f"\n  OOF AUC: {overall_auc:.5f}")

# ── Save submission ───────────────────────────────────────────────────────────
print("\n=== Saving submission ===")
sub_df["Label"] = test_preds
out_path = "M11415803_Le_Trung_Kien_A.csv"
sub_df.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
print(sub_df.head(10))
print(f"\nLabel stats:  min={test_preds.min():.4f}  max={test_preds.max():.4f}  mean={test_preds.mean():.4f}")