---
title: Twitch Gamers EDA + Community Detection (Part 1 & 2)
description: >-
  Structural/statistical EDA and Louvain community detection on the Twitch
  Gamers mutual-follow network (168K nodes, 6.8M edges)
status: completed
priority: P2
branch: main
tags:
  - data-analysis
  - graph
  - eda
  - community-detection
blockedBy: []
blocks: []
created: '2026-05-29T06:31:17.522Z'
createdBy: 'ck:plan'
source: skill
---

# Twitch Gamers EDA + Community Detection (Part 1 & 2)

## Overview

Implement team-assigned Part 1 (Exploratory Data Analysis) and Part 2 (Community Detection)
from `proposal.md` against local `twitch_gamers/`. Proposal verified to match dataset:
168,114 nodes, 6,797,557 edges, columns
`views, mature, life_time, created_at, updated_at, numeric_id, dead_account, language, affiliate`
(churn = `dead_account`). README global transitivity (0.0184) used as a correctness checkpoint.

**Engine decision:** use **python-igraph** (C core) as graph backbone. At 6.8M edges networkx
is too slow/memory-heavy for PageRank, per-node clustering, and Louvain. Louvain =
`igraph.Graph.community_multilevel()`. Stats/viz via pandas/numpy/matplotlib/seaborn;
power-law fit via `powerlaw`; NMI via scikit-learn.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Setup & Data Layer](./phase-01-setup-data-layer.md) | Completed |
| 2 | [Part 1 EDA](./phase-02-part-1-eda.md) | Completed |
| 3 | [Part 2 Community Detection](./phase-03-part-2-community-detection.md) | Completed |

## Deliverables

- `src/data_loader.py` — load features CSV, build/cache undirected igraph graph
- `src/part1_eda/*`, `src/part2_community/*` — focused modules (<200 lines each)
- `src/run_part1.py`, `src/run_part2.py` — orchestrators (run via `uv run`)
- `results/figures/`, `results/tables/` — generated artifacts
- `reports/part1-eda-findings.md`, `reports/part2-community-findings.md` — written findings

## Conventions

Python modules use **snake_case** (not kebab-case): hyphens break `import`. Names stay
long/descriptive to satisfy the self-documenting intent. Vertex order = `numeric_id` order so
graph metrics align row-for-row with the features DataFrame.

## Dependencies

None cross-plan. Self-contained homework analysis. Parts 3 & 4 owned by other teammates,
out of scope (no shared code required).

## Key Risk

Project pins `requires-python >=3.14`. If `python-igraph`/`powerlaw`/`scipy` lack 3.14 wheels,
fall back to `uv venv --python 3.12`. Resolved in Phase 1.
