---
phase: 2
title: Part 1 EDA
status: completed
priority: P1
effort: 3-4h
dependencies:
  - 1
---

# Phase 2: Part 1 EDA

## Overview

Examine structural and statistical properties before any ML. Four analyses, one module each,
plus a `run_part1.py` orchestrator that produces figures, tables, and a findings report.

## Requirements (maps 1:1 to proposal Part 1 bullets)

1. **Degree distribution + power-law** — fit power-law, report exponent α; decide scale-free.
2. **Clustering & transitivity** — global transitivity + average local clustering coefficient.
3. **Node feature distributions** — language, affiliate rate, view count, lifetime, churn rate.
4. **Correlation analysis** — heatmap of graph-structural features (degree, PageRank, local
   clustering) vs node attributes (views, affiliate, lifetime).

## Architecture / Modules

- `part1_eda/degree_powerlaw.py`:
  - degrees = `graph.degree()`. Plot degree distribution on log-log (CCDF preferred).
  - `powerlaw.Fit(degrees, discrete=True)` → α (`fit.power_law.alpha`), `xmin`, KS distance.
  - `fit.distribution_compare('power_law','lognormal')` → (R, p) to avoid over-claiming
    scale-free. Report both; conclusion = "scale-free in the tail iff α∈~(2,3) and PL not
    rejected vs lognormal".
- `part1_eda/clustering_transitivity.py`:
  - `transitivity_undirected()` (global, ≈0.0184 checkpoint) and
    `mean(transitivity_local_undirected(mode='zero'))` (avg local). Note sparsity → low values.
- `part1_eda/feature_distributions.py`:
  - language bar chart (top-N + OTHER), affiliate rate = `affiliate.mean()`,
    churn rate = `dead_account.mean()`, views histogram on log scale (heavy-tailed),
    life_time histogram. Emit a summary stats table (`results/tables/feature_summary.csv`).
- `part1_eda/correlation_analysis.py`:
  - per-node frame: `degree`, `pagerank` (`graph.pagerank()`), `local_clustering`,
    plus `views`/`log_views`, `affiliate`, `life_time`.
  - Compute **Spearman** correlation (skewed/heavy-tailed data → rank correlation more honest
    than Pearson; include Pearson too for completeness). Seaborn heatmap.
- `run_part1.py`: load graph once, call the four modules, save all artifacts, write report.

## Related Code Files

- Create: `src/part1_eda/degree_powerlaw.py`, `clustering_transitivity.py`,
  `feature_distributions.py`, `correlation_analysis.py`; `src/run_part1.py`;
  `reports/part1-eda-findings.md`
- Read: `src/data_loader.py`
- Output: `results/figures/p1_*.png`, `results/tables/*.csv`

## Implementation Steps

1. Build per-node metrics frame once in `run_part1.py` (degree, pagerank, local clustering) —
   reuse across correlation; do not recompute per module.
2. Implement the four modules; each takes the shared graph/frame and returns figure paths +
   a small results dict.
3. Save figures at 150 dpi, consistent style; tables as CSV.
4. Write `part1-eda-findings.md`: α value + scale-free verdict, transitivity vs avg-local,
   feature summary (EN≈74% dominance, affiliate/churn rates), strongest correlations.

## Success Criteria

- [ ] `uv run python -m src.run_part1` completes, writes ≥5 figures + summary tables
- [ ] α reported with KS + power-law-vs-lognormal comparison (not just a bare exponent)
- [ ] global transitivity in output matches README 0.0184
- [ ] correlation heatmap covers all 3 structural × 3 attribute features; findings written

## Risk Assessment

- Per-node local clustering on 6.8M edges is the slowest step → igraph C impl handles it (~tens
  of seconds); compute once and cache in the metrics frame.
- `powerlaw` may warn on discrete fits → use `discrete=True`, report `xmin`, don't suppress.
