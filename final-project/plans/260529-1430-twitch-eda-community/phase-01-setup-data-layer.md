---
phase: 1
title: Setup & Data Layer
status: completed
priority: P1
effort: 1-2h
dependencies: []
---

# Phase 1: Setup & Data Layer

## Overview

Establish the Python environment and a single shared data layer that both parts import. One
loader, one cached graph build — no duplicated CSV/graph code (DRY).

## Requirements

- Functional: load `large_twitch_features.csv` into a pandas DataFrame indexed by `numeric_id`;
  build an undirected, simplified igraph `Graph` from `large_twitch_edges.csv` with vertices
  ordered by `numeric_id` so metric arrays align row-for-row with the DataFrame.
- Non-functional: graph build < ~60s, peak RAM < ~3GB; cache the built graph to avoid rebuilds.

## Architecture

- `data_loader.py` exposes:
  - `load_features() -> pd.DataFrame` — dtypes set explicitly; `created_at`/`updated_at` parsed
    as datetime; derived `log_views = log1p(views)`.
  - `build_graph(features) -> ig.Graph` — vertices = `range(n)` matching `numeric_id` (ids are
    contiguous 0..168113; assert this), edges from CSV, `.simplify()` to drop dup/self-loops.
  - `load_graph_cached() -> ig.Graph` — pickle to `results/cache/graph.pkl`; rebuild if missing.
- Verification hook: after build, assert `graph.vcount() == 168114` and
  `0.017 < graph.transitivity_undirected() < 0.020` (README checkpoint 0.0184).

## Related Code Files

- Create: `src/data_loader.py`, `src/__init__.py`, `src/part1_eda/__init__.py`,
  `src/part2_community/__init__.py`
- Modify: `pyproject.toml` (add deps), `README.md` (fill empty file: how to run)
- Create dirs: `results/figures/`, `results/tables/`, `results/cache/`, `reports/`

## Implementation Steps

1. Resolve interpreter: `uv venv` (try 3.14). Smoke-test wheels:
   `uv add python-igraph pandas numpy matplotlib seaborn scikit-learn powerlaw`.
   If any wheel build fails on 3.14 → `uv venv --python 3.12` and retry.
2. Write `data_loader.py` per Architecture. Read edges fast with
   `pandas.read_csv(..., dtype=np.int32)` then `Graph(n, edges.tolist())` or
   `Graph.DataFrame`. Confirm contiguous ids before assuming positional mapping.
3. Add the transitivity + vcount assertions; print them once on first build.
4. Add `results/cache/` to `.gitignore` (don't commit the pickle / large artifacts).
5. Fill `README.md`: dataset note, `uv sync`, `uv run python -m src.run_part1`.

## Success Criteria

- [ ] `uv run python -c "from src.data_loader import load_graph_cached as g; print(g().summary())"`
      prints 168114 vertices and runs without error
- [ ] global transitivity assertion passes (≈0.0184)
- [ ] DataFrame row order aligns with vertex indices (spot-check 5 ids)

## Risk Assessment

- 3.14 wheel gaps → mitigation: pin 3.12 (documented in plan Key Risk).
- id non-contiguity would break positional mapping → mitigation: explicit assert; if it fails,
  build a `numeric_id -> vertex_index` map and reindex.
