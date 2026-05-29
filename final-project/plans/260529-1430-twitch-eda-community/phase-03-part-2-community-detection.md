---
phase: 3
title: Part 2 Community Detection
status: completed
priority: P1
effort: 2-3h
dependencies:
  - 1
---

# Phase 3: Part 2 Community Detection

## Overview

Detect communities with Louvain, profile the major ones, and test (via NMI) whether the
network is organized by language or by other factors. Depends only on Phase 1 (the shared
graph + features); independent of Phase 2.

## Requirements (maps 1:1 to proposal Part 2 bullets)

1. **Louvain** — maximize modularity; report modularity score Q.
2. **Community profiling** — per major community: dominant language, language purity,
   affiliate rate, median view count.
3. **NMI evaluation** — NMI between detected communities and language labels.

## Architecture / Modules

- `part2_community/louvain_detection.py`:
  - `graph.community_multilevel()` (igraph's Louvain). Report `partition.modularity` (Q),
    number of communities, and the size distribution (largest, median, count of singletons).
  - Determinism note: Louvain has randomness; igraph multilevel is deterministic given fixed
    input order — record Q and cluster count; optionally run 2–3 times to confirm stability.
- `part2_community/community_profiling.py`:
  - Attach `community` label to the features DataFrame (positional align via Phase 1 ordering).
  - For top-N communities by size: size, dominant language, **language purity** =
    share of the dominant language within the community, affiliate rate (`affiliate.mean()`),
    median `views`. Emit `results/tables/community_profile.csv`.
- `part2_community/nmi_evaluation.py`:
  - `sklearn.metrics.normalized_mutual_info_score(community_labels, language_codes)`.
  - Interpret against EN≈74% baseline: high NMI → language-organized; low NMI → other drivers
    (geography/topic/follow-cliques). State the verdict explicitly.
- `run_part2.py`: load graph once, run detection → profiling → NMI, save artifacts, write report.

## Related Code Files

- Create: `src/part2_community/louvain_detection.py`, `community_profiling.py`,
  `nmi_evaluation.py`; `src/run_part2.py`; `reports/part2-community-findings.md`
- Read: `src/data_loader.py`
- Output: `results/figures/p2_*.png`, `results/tables/community_profile.csv`

## Implementation Steps

1. Run Louvain on the cached graph; capture Q + partition object.
2. Map community labels back to DataFrame rows; build the top-N profile table.
3. Plot community size distribution + a stacked language-composition bar for top-N.
4. Compute NMI(community, language). Compare to a sanity baseline (e.g. NMI with a random
   permutation of labels ≈ 0) to contextualize the value.
5. Write `part2-community-findings.md`: Q, #communities, profile highlights, NMI + verdict.

## Success Criteria

- [ ] `uv run python -m src.run_part2` completes; reports Q and community count
- [ ] community profile table covers dominant language, purity, affiliate rate, median views
- [ ] NMI(community, language) computed with an explicit organized-by-language verdict
- [ ] Louvain Q reported as a positive modularity (sanity: Q clearly > 0)

## Risk Assessment

- Singleton/tiny communities can dominate the count → profile only top-N by size and report how
  much of the graph they cover.
- Positional label alignment is the main correctness hazard → reuse the Phase 1 ordering
  invariant and spot-check a few `numeric_id`s after the join.
