"""
Finer ensemble weight grid around the 0.3/0.7 sweet spot.

ens_t3h7_top460 scored 0.948 — best so far. Explore weights 0.15-0.45
for tuned and equivalent for hidden. Also try rank-average and
geometric mean as alternative blending strategies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from graph_utils import ROOT, load_test_pairs, write_submission

# Fine weight grid for tuned (hidden = 1 - tuned)
FINE_WEIGHTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TARGET_COUNTS = [455, 458, 460, 462, 465, 470]


def main() -> None:
    tuned = np.load(ROOT / "tuned_probs.npy")
    hidden = np.load(ROOT / "hidden_edge_probs.npy")
    test_pairs = load_test_pairs()

    # Linear-weighted blends
    for w_t in FINE_WEIGHTS:
        w_h = 1 - w_t
        ens = w_t * tuned + w_h * hidden
        order = np.argsort(-ens)
        for n_target in TARGET_COUNTS:
            positive_idx = set(order[:n_target].tolist())
            preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                     for i in range(len(test_pairs))]
            tag = f"t{int(w_t * 100):02d}h{int(w_h * 100):02d}"
            out = ROOT / f"submission_fine_{tag}_top{n_target}.csv"
            write_submission(preds, out)
        print(f"  weights tuned={w_t:.2f}/hidden={w_h:.2f}: 6 submissions")

    # Rank-average ensemble (Borda)
    rank_tuned = np.argsort(np.argsort(-tuned))  # smaller rank = higher prob
    rank_hidden = np.argsort(np.argsort(-hidden))
    rank_avg = (rank_tuned + rank_hidden) / 2.0
    order = np.argsort(rank_avg)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        out = ROOT / f"submission_rankavg_top{n_target}.csv"
        write_submission(preds, out)
    print("  rank-average: 6 submissions")

    # Geometric mean
    eps = 1e-9
    geo = np.sqrt(np.clip(tuned, eps, 1) * np.clip(hidden, eps, 1))
    order = np.argsort(-geo)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        out = ROOT / f"submission_geomean_top{n_target}.csv"
        write_submission(preds, out)
    print("  geometric-mean: 6 submissions")

    # Max (union of confident predictions)
    mx = np.maximum(tuned, hidden)
    order = np.argsort(-mx)
    for n_target in TARGET_COUNTS:
        positive_idx = set(order[:n_target].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        out = ROOT / f"submission_max_top{n_target}.csv"
        write_submission(preds, out)
    print("  max: 6 submissions")


if __name__ == "__main__":
    main()
