# Part 2 — Community Detection: Findings

Dataset: Twitch Gamers mutual-follow network — 168,114 nodes, 6,797,557 edges.

## 1. Louvain communities
- Modularity **Q = 0.4228** (23 communities).
- Largest community = 47,341 nodes; median size = 2005; singletons = 0.
- Stability across 3 runs: Q = [0.4228, 0.4226, 0.422] (std = 0.00034), communities = [23, 24, 22] — modularity is stable, not a single-run artefact.

## 2. Major community profiles
Top 10 communities cover 93.2% of all nodes; mean language purity among them = 90.3%.

|   community |   size | dominant_language   |   language_purity |   affiliate_rate |   median_views |
|------------:|-------:|:--------------------|------------------:|-----------------:|---------------:|
|           1 |  47341 | EN                  |            0.905  |           0.4383 |           3090 |
|           3 |  29994 | EN                  |            0.9775 |           0.5592 |           5087 |
|           7 |  18020 | EN                  |            0.9074 |           0.434  |           4617 |
|           2 |  17430 | EN                  |            0.9283 |           0.3128 |           2347 |
|           0 |  14086 | EN                  |            0.9686 |           0.768  |           3797 |
|           9 |   8829 | DE                  |            0.9494 |           0.514  |           6572 |
|           4 |   6255 | FR                  |            0.9584 |           0.5209 |           6094 |
|           5 |   5077 | ZH                  |            0.5395 |           0.3551 |          12274 |
|          10 |   5050 | ES                  |            0.955  |           0.5307 |           4817 |
|           6 |   4644 | RU                  |            0.9462 |           0.4388 |          11269 |

## 3. NMI: communities vs language
- NMI(community, language) = **0.4786** (random baseline ≈ 0.0007).
- **Verdict: partially organised by language.**

## 4. Cluster-quality metrics (graph-native Silhouette analogs)
- Modularity Q = **0.4228** (vs degree-preserving null ≈ 0.0792 → **5.3×** the null: structure is genuine).
- Coverage = 60.4% of edges stay intra-community.
- Conductance (lower = better separated): mean 0.374, median 0.3794.

Figures: `results/figures/p2_*.png`. Table: `results/tables/community_profile.csv`.
