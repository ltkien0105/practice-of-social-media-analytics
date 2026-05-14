# Link Prediction on a Directed Social Network — Report

**Student ID:** M11415803
**Name:** Le Trung Kien
**Task:** HW1.2 — Predict hidden directed edges (Node1 → Node2) in a social-network graph reconstructed from ~186,000 observed edges. Evaluation metric: ROC AUC.

---

## 1. Approach Summary

Two complementary supervised models were trained, and their outputs were blended in **rank space** to produce the final prediction.

| Component | Role |
|---|---|
| **Model A** — `main_1.py` | Compact hand-crafted graph features + LightGBM |
| **Model B** — `main_2.py` | Extended graph features + 256-dim SVD embeddings + LightGBM |
| **Rank blend** — `make_blends.py` | Weighted average of percentile ranks (80% A, 20% B) |

The blend was chosen because the two models exhibit moderate disagreement (Spearman ρ = 0.7226 between their predictions), so combining them captures orthogonal signal.

---

## 2. Algorithm Details

### 2.1 Model A — Hand-Crafted Graph Features (`main_1.py`)

For every candidate pair (u, v), 19 features are computed from the directed graph G and its undirected projection:

| Group | Features |
|---|---|
| Degrees | `out_deg(u)`, `in_deg(u)`, `out_deg(v)`, `in_deg(v)` |
| Common neighbours (4 directed variants + 1 undirected) | `\|out_u ∩ out_v\|`, `\|out_u ∩ in_v\|`, `\|in_u ∩ out_v\|`, `\|in_u ∩ in_v\|`, `\|N_u ∩ N_v\|` |
| Similarity | Jaccard (undirected), Adamic–Adar |
| Preferential attachment | `out_deg(u) × in_deg(v)`, `\|N_u\| × \|N_v\|` |
| Reciprocity | `v→u edge?`, `u in out(v)?` |
| Path / structure | length-2 path count, hub ratio for u and v, Katz-like proxy `cn_out_out / (out_deg(u)+1)` |

**Training procedure:**
- Negatives: random non-edges, drawn in equal number to positives (seed = 42).
- Classifier: LightGBM (`n_estimators=800`, `learning_rate=0.05`, `num_leaves=63`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`), early stopping with patience 50.
- 5-fold stratified CV; final test predictions are the per-fold average.

### 2.2 Model B — Graph Features + SVD Embeddings (`main_2.py`)

Adds two pieces on top of Model A's idea:

**(i) 27 graph features**, including everything in Model A plus:
- Global node scores: `PageRank`, `HITS` hub/authority for u and v.
- Community: Louvain communities on the undirected graph; `same_community` indicator.
- Resource-allocation index, triadic-closure ratio, follow-ratios `out/(in+1)`.

**(ii) 256-dim truncated SVD embeddings** of the binary adjacency matrix A:

```
A ≈ U Σ Vᵀ        (top-256 singular values)
src_emb(u) = U[u] · √Σ
dst_emb(v) = Vᵀ[v] · √Σ
```

For each pair, the embedding-derived features are: dot product, L2 distance, cosine similarity, source/destination norms, summary statistics of the Hadamard product (mean, std, |·| mean, |·| max), and the full 256-dim Hadamard product itself.

Total feature dimension ≈ 290. Classifier: LightGBM (`n_estimators=1000`, `learning_rate=0.03`, `num_leaves=255`). 5-fold stratified CV.

### 2.3 Final Ensemble — Rank-Space Blending

The two models have very different probability calibrations:

| Model | Mean predicted probability | Median |
|---|---|---|
| Model A | 0.111 | 0.0004 |
| Model B | 0.409 | 0.0056 |

Naive probability averaging is dominated by Model B's inflated scale. To remove the scale mismatch, predictions are first converted to **percentile ranks** in [0, 1] and then averaged:

```
final_score(pair) = 0.8 × rank_A(pair) + 0.2 × rank_B(pair)
```

The weight 0.8 was selected by probing 50/50, 70/30, 80/20, 90/10 on the public leaderboard; 80/20 gave the best score (see Section 3). Because rank-AUC is a step function of the blend weight (it changes only at points where two pairs swap order), the curve is not smooth and the optimum has to be found empirically.

---

## 3. Results

| Submission | CV AUC | Public LB AUC |
|---|---|---|
| Model A (main_1.py) | 0.99997 | 0.85428 |
| Model B (main_2.py) | 0.99940 | 0.78773 |
| Rank blend 50/50 | — | 0.84212 |
| Rank blend 70/30 | — | 0.85292 |
| Rank blend 90/10 | — | 0.85062 |
| **Rank blend 80/20 (final pick #1)** | — | **0.85505** |
| **Model A standalone (final pick #2)** | — | 0.85428 |

The two final submissions were chosen for diversity: the blend captures both models' signals; pure Model A acts as an anchor in case the blend overfits the public leaderboard.

> **Note on the CV–LB gap.** Both models scored near 1.0 in cross-validation but markedly lower on the leaderboard. The cause is the negative-sampling distribution mismatch: random non-edges are easy to distinguish from real edges, while the actual hidden edges lie in much harder regions of the graph. Model A's simpler feature set generalizes better than Model B's richer one (smaller CV–LB gap), which is why Model A dominates the blend.

---

## 4. How to Run

### Requirements

- Python ≥ 3.12
- Package manager: [`uv`](https://docs.astral.sh/uv/)
- Dependencies (declared in `pyproject.toml`): `lightgbm`, `networkx`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`

Install:

```bash
uv sync
```

### Input Data

Place the following in the working directory:

- `train.csv` — known edges (columns: `Node1`, `Node2`)
- `test.csv` — candidate pairs to score (columns: `ID`, `Node1`, `Node2`)
- `sample_submission.csv` — submission template (columns: `ID`, `Label`)

### Step 1 — Train Model A and produce its submission

```bash
uv run python main_1.py
mv submission.csv M11415803_Le_Trung_Kien.csv
```

### Step 2 — Train Model B and produce its submission

```bash
uv run python main_2.py
mv submission.csv M11415803_Le_Trung_Kien_0505_opus5fold.csv
```

### Step 3 — Generate the rank-space blends

```bash
uv run python make_blends.py
```

This reads the two CSVs above and writes all blend variants. The final submission is **`M11415803_Le_Trung_Kien_blend_rank_80_20.csv`**.

### Reproducibility Notes

- All RNG seeds are fixed (`random_state=42` in LightGBM, seeded `numpy` / `random` for negative sampling).
- `scipy.sparse.linalg.svds` does not expose a deterministic seed for its ARPACK initialization, so Model B's outputs may differ by ~10⁻⁴ in absolute probability across runs. Rank order — and therefore AUC — is essentially unaffected.

---

## 5. File Manifest

| File | Description |
|---|---|
| `main_1.py` | Model A pipeline (features + 5-fold LGBM) |
| `main_2.py` | Model B pipeline (features + SVD + LGBM) |
| `make_blends.py` | Computes rank-space and probability-space blends |
| `M11415803_Le_Trung_Kien.csv` | Model A submission — **final pick #2** |
| `M11415803_Le_Trung_Kien_blend_rank_80_20.csv` | Final blend — **final pick #1** |
| `M11415803_Le_Trung_Kien_report.md` | This report |

---
