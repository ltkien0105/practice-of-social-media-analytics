# Part 1 — Exploratory Data Analysis: Findings

Dataset: Twitch Gamers mutual-follow network — 168,114 nodes, 6,797,557 edges.

## 1. Degree distribution & power law
- Power-law exponent α = **2.5437** (xmin = 250, KS distance = 0.0051).
- Power-law vs lognormal: log-likelihood ratio R = -0.1867, p = 0.8519 (R>0 favours power law).
- Mean degree = 80.87, max degree = 35279.
- **Verdict: heavy-tailed, consistent with scale-free in the tail (α≈2.5437); note the power-law vs lognormal test is not decisive, so we avoid claiming a strictly scale-free network** (criterion: 2<α<3 and power law not rejected vs lognormal).

## 2. Clustering & transitivity
- Global transitivity = **0.0184** (matches dataset README checkpoint 0.0184).
- Average local clustering coefficient = 0.1599.
- Graph density = 0.000481.
- Transitivity is **~38.2×** the random-graph baseline (≈ density 0.000481), i.e. real local structure well above chance despite the sparse graph.

## 3. Node feature distributions
- Languages: 21 distinct; dominant = **EN** (74.0% of nodes).
- Affiliate rate = 48.5%; churn (dead account) rate = 3.1%; mature rate = 47.0%.
- Median views = 4,117; median lifetime = 1,540 days.

## 4. Correlation (structural vs attributes)
- Strongest structural↔attribute pair (Spearman): **pagerank ~ log_views** = 0.5515.
- degree ~ log_views = 0.5144; pagerank ~ affiliate = 0.239.

Figures: `results/figures/p1_*.png`. Tables: `results/tables/*.csv`.
