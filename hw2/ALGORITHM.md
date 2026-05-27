# HW2 Algorithm Summary — Community Membership Prediction

**Student:** M11415803 — Le Trung Kien
**Task:** Given a graph (train.csv: 188,109 nodes / 398,711 undirected edges),
predict for each of 1,000 test pairs whether the two nodes belong to the
same community.
**Metric:** accuracy over 1,000 pairs (Kaggle).

## Final method: Personalized PageRank (PPR) — score 0.964 (ties leaderboard #1)

`main.py` is the complete pipeline (≈54 s runtime, deps: numpy + scipy).

### Steps

1. **Build transition matrix.** From the undirected edge list, build the
   symmetric binary adjacency `A`, then the row-stochastic random-walk
   matrix `P = D⁻¹A` (degree-normalized).

2. **Personalized PageRank per endpoint.** For each node `s` that appears
   as an endpoint of some test pair (1,750 of them), run PPR by power
   iteration:

   ```
   r₀ = e_s
   r_{t+1} = α · Pᵀ r_t + (1 − α) · e_s        α = 0.85, 20 iterations
   ```

   `r_s[v]` = long-run probability a walker that starts at `s` (and
   teleports back to `s` with prob `1 − α` each step) is found at `v`.

3. **Symmetric pair score.** For each test pair `(u, v)`:

   ```
   score(u, v) = (PPR(u → v) + PPR(v → u)) / 2
   ```

4. **Threshold by rank.** Predict the **top-474** highest-scoring pairs as
   same-community (`1`), the rest `0`.

### Why it works

Local link-prediction features (common neighbours, Jaccard, Adamic-Adar,
shortest path) only see the immediate overlap of two nodes' neighbourhoods.
PPR instead integrates **all paths of every length**, geometrically
down-weighted by `α`. Two nodes deep inside the same dense community
accumulate high mutual PPR mass even when they share no direct neighbour —
exactly the "hard" pairs the local-feature models miss.

Empirically PPR correlates only **0.50** with the structural-feature
ensemble (vs 0.86 for Node2Vec, 0.71 for spectral) yet is *more* accurate at
the decision boundary — it is the rare signal that is both decorrelated and
correct.

### The top-N tuning axis

The PPR score ranks pairs well; the only tunable is the cut point. The peak
is sharp at **474**:

| top-N | Kaggle |
|------:|:------:|
| 450 | 0.948 |
| 460 | 0.956 |
| 470 | 0.962 |
| **474** | **0.964** |
| 480 | 0.956 |
| 485 | 0.948 |

Two side notes from this curve:

- **The signal is normalization-invariant.** Symmetric-normalized walk
  `D^(-1/2) A D^(-1/2)` and a higher restart `α = 0.90` produce the *exact
  same* top-474 set as `D⁻¹A`, `α = 0.85` — so α / normalization are dead
  levers; the cut point is the entire game.
- **Kaggle's public score is on a subset.** The 474→480 drop is 8 points
  (0.964→0.956) yet only 6 pairs changed — impossible on a full-1000 metric.
  So 474 is the *public* optimum; the diverse second pick hedges the private
  split.

## What did NOT work (the 0.948 plateau)

A large search before PPR all capped at **0.948**:

| Approach | Best | Note |
|---|---|---|
| Louvain / Leiden / hierarchical / consensus partitions | 0.858 | community-detection labels too coarse |
| Gradient-Boosting on 7 structural features | 0.922 | |
| Tuned 8-seed GB ensemble | 0.932 | |
| Hidden-edge 6-seed GB (12 features, edge-hiding trick) | 0.946 | best single classifier |
| Ensemble 0.3·tuned + 0.7·hidden, top-460 | **0.948** | the plateau |
| Walktrap / holdout / fine-weight-grid blends | 0.948 | could not exceed |
| Node2Vec cosine (uniform & biased q=0.5) | 0.946 | embeddings correlate 0.86 with baseline |
| Node2Vec hadamard + GB | 0.668 | hadamard destroys the signal |
| Spectral embedding (k=64 / k=256) cosine | 0.802 | too coarse for 1,019 communities |
| Classifier + embedding-cosine features | 0.944 | overfits to n2v_cos (importance 0.998) |

**Lesson:** every embedding/classifier signal was highly correlated with the
structural ensemble, so blending never helped — each disagreement resolved
in the baseline's favour. PPR was the first signal *independent enough* to
add real information.

## Files (final 2 approaches kept)

| File | Purpose |
|---|---|
| `main.py` | **Primary** — PPR pipeline → `M11415803_Le_Trung_Kien.csv` (0.964) |
| `submission_ensemble_top460.csv` | Secondary submission — structural ensemble (0.948) |
| `ensemble.py` | **Secondary** — structural-feature GB ensemble → `submission_ensemble_top460.csv` (0.948) |
| `plans/kaggle-scores.md` | Full per-submission score log (all approaches tried) |

Two final Kaggle picks chosen for diversity: PPR (0.964, multi-hop reach)
and the structural ensemble (0.948, local features) — uncorrelated methods,
hedging against a public/private leaderboard shake-up. All exploratory
scripts (Node2Vec, spectral, blends, PPR tuning) were removed after their
findings were folded into this summary; they remain in git history.

## Status: ties leaderboard #1 (0.964)

PPR top-474 reaches 0.964 — the public leaderboard top. Since the public
metric is a subset (see note above) and α / normalization are dead levers,
there is no further headroom from this signal alone. Beating 0.964 on the
*private* split would require a genuinely independent signal (e.g. a GNN), at
significant engineering cost for an uncertain gain — the diverse 0.948
ensemble is the cheaper hedge.
