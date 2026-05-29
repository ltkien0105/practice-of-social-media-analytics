# Social Network Analysis — Twitch Gamers (Part 1 & 2)

EDA and community detection on the Twitch Gamers mutual-follow network
(168,114 nodes, 6,797,557 edges). See `proposal.md` for the full project scope;
this repo covers **Part 1 (EDA)** and **Part 2 (Community Detection)**.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python -m src.run_part1   # EDA: degree/power-law, clustering, feature dists, correlation
uv run python -m src.run_part2   # Louvain communities, profiling, NMI vs language
```

First run builds and caches the igraph graph to `results/cache/graph.pkl` (~17s).

## Layout

- `src/data_loader.py` — shared loader + cached graph build (vertices in `numeric_id` order)
- `src/part1_eda/` — one module per EDA analysis
- `src/part2_community/` — Louvain detection, profiling, NMI
- `results/figures/`, `results/tables/` — generated artifacts
- `reports/` — written findings (`part1-eda-findings.md`, `part2-community-findings.md`)

## Engine

python-igraph (C core) — networkx is too slow/memory-heavy at 6.8M edges.
