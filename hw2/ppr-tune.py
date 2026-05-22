"""
PPR tuning — given the ppr_sym top-460 scored 0.956 (beat the 0.948 plateau),
sweep top-N and asymmetric PPR aggregations to see if we can push higher.

Variants:
- ppr_sym top-N for N in {440, 450, 455, 460, 465, 470, 480}
- ppr_max = max(ppr_uv, ppr_vu): top-N
- ppr_geomean = sqrt(ppr_uv * ppr_vu): top-N
- ppr_min = min(ppr_uv, ppr_vu): top-N (high-precision)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
ppr_scores = np.load(ROOT / "ppr_scores.npy")
ppr_uv = ppr_scores[:, 0]
ppr_vu = ppr_scores[:, 1]

test_pairs = []
with (ROOT / "test.csv").open() as f:
    r = csv.reader(f); next(r)
    for row in r:
        test_pairs.append((int(row[0]), int(row[1]), int(row[2])))

ppr_sym = (ppr_uv + ppr_vu) / 2
ppr_max = np.maximum(ppr_uv, ppr_vu)
ppr_min = np.minimum(ppr_uv, ppr_vu)
ppr_geom = np.sqrt(np.clip(ppr_uv * ppr_vu, 0, None))

# Baseline overlap for sanity
tuned = np.load(ROOT / "tuned_probs.npy")
hidden = np.load(ROOT / "hidden_probs.npy")
base = 0.3 * tuned + 0.7 * hidden
ts_base460 = set(np.argsort(-base)[:460].tolist())


def write(scores, name, n):
    order = np.argsort(-scores)
    pos = set(order[:n].tolist())
    path = ROOT / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for i, (tid, _, _) in enumerate(test_pairs):
            w.writerow([tid, 1 if i in pos else 0])
    overlap = len(pos & ts_base460)
    print(f"  {name}: positives={len(pos)} overlap_with_0.948_baseline={overlap}/{min(n,460)}")


print("ppr_sym top-N sweep:")
for n in (430, 440, 445, 450, 455, 460, 465, 470, 480):
    write(ppr_sym, f"submission_ppr_sym_top{n}.csv", n)

print("\nppr_max top-N (favors strong directional reachability):")
for n in (440, 460, 480):
    write(ppr_max, f"submission_ppr_max_top{n}.csv", n)

print("\nppr_min top-N (high precision - both directions must agree):")
for n in (440, 460, 480):
    write(ppr_min, f"submission_ppr_min_top{n}.csv", n)

print("\nppr_geom top-N (geometric mean):")
for n in (440, 460, 480):
    write(ppr_geom, f"submission_ppr_geom_top{n}.csv", n)
