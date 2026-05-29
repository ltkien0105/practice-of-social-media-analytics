# Part 2 — Community Detection: Findings

Dataset: Twitch Gamers mutual-follow network — 168,114 nodes, 6,797,557 edges.

## 1. Louvain communities
- Modularity **Q = 0.4225** (24 communities).
- Largest community = 43,161 nodes; median size = 1180; singletons = 0.
- Stability across 3 runs: Q = [0.4225, 0.4254, 0.4234] (std = 0.00121), communities = [24, 19, 28] — modularity is stable, not a single-run artefact.

## 2. Major community profiles
Top 10 communities cover 95.0% of all nodes; mean language purity among them = 91.8%.

|   community |   size | dominant_language   |   language_purity |   affiliate_rate |   median_views |
|------------:|-------:|:--------------------|------------------:|-----------------:|---------------:|
|           1 |  43161 | EN                  |            0.9067 |           0.4387 |           2999 |
|           3 |  31865 | EN                  |            0.9707 |           0.486  |           4081 |
|           2 |  26422 | EN                  |            0.9159 |           0.4037 |           3813 |
|           0 |  17922 | EN                  |            0.9715 |           0.7569 |           4144 |
|           6 |  11992 | EN                  |            0.6503 |           0.3761 |           4452 |
|           9 |   8915 | DE                  |            0.9456 |           0.5137 |           6562 |
|           4 |   6259 | FR                  |            0.9581 |           0.521  |           6090 |
|          10 |   5131 | ES                  |            0.9544 |           0.5295 |           4842 |
|           7 |   4661 | RU                  |            0.9399 |           0.4364 |          11150 |
|          12 |   3433 | EN                  |            0.9665 |           0.4984 |           2361 |

## 3. NMI: communities vs language
- NMI(community, language) = **0.4755** (random baseline ≈ 0.0007).
- **Verdict: partially organised by language.**

## 4. Cluster-quality metrics (graph-native Silhouette analogs)
- Modularity Q = **0.4225** (vs degree-preserving null ≈ 0.0796 → **5.3×** the null: structure is genuine).
- Coverage = 60.9% of edges stay intra-community.
- Conductance (lower = better separated): mean 0.3784, median 0.389.

Figures: `results/figures/p2_*.png`. Table: `results/tables/community_profile.csv`.
