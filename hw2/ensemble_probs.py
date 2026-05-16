"""
Ensemble of tuned + hidden-edge classifier probabilities.

Both models have similar Kaggle scores but use different features
(tuned: same_leid dominant; hidden: lou_cons dominant), so they likely
disagree on different pairs. Averaging probabilities should improve
ranking quality.

Outputs:
- Weighted average at several weights (favor higher-scoring hidden model)
- Threshold and rank-based submissions
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from graph_utils import ROOT, load_test_pairs, write_submission

WEIGHTS = [(0.0, 1.0), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (1.0, 0.0)]
THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
TARGET_COUNTS = [440, 450, 460, 465, 470, 478]


def main() -> None:
    tuned = np.load(ROOT / "tuned_probs.npy")
    hidden = np.load(ROOT / "hidden_edge_probs.npy")
    test_pairs = load_test_pairs()
    assert len(tuned) == len(hidden) == len(test_pairs)

    print(f"Tuned : min={tuned.min():.3f} max={tuned.max():.3f} "
          f"median={np.median(tuned):.3f}")
    print(f"Hidden: min={hidden.min():.3f} max={hidden.max():.3f} "
          f"median={np.median(hidden):.3f}")

    # Rank correlation between the two models
    rank_tuned = np.argsort(np.argsort(-tuned))
    rank_hidden = np.argsort(np.argsort(-hidden))
    corr = np.corrcoef(rank_tuned, rank_hidden)[0, 1]
    print(f"Rank correlation (tuned vs hidden): {corr:.4f}")

    disagree_top500 = len(set(np.argsort(-tuned)[:500].tolist())
                          ^ set(np.argsort(-hidden)[:500].tolist())) // 2
    print(f"Top-500 disagreement: {disagree_top500} pairs")

    for w_t, w_h in WEIGHTS:
        ens = w_t * tuned + w_h * hidden
        tag = f"t{int(w_t * 10)}h{int(w_h * 10)}"
        print(f"\nWeights tuned={w_t} hidden={w_h}:")

        for thr in THRESHOLDS:
            preds = [(test_pairs[i][0], int(ens[i] >= thr))
                     for i in range(len(test_pairs))]
            pos_count = sum(1 for _, c in preds if c == 1)
            out = ROOT / f"submission_ens_{tag}_t{thr:g}.csv"
            write_submission(preds, out)
            print(f"  thr={thr:<5} positives={pos_count:<4} -> {out.name}")

        order = np.argsort(-ens)
        for n_target in TARGET_COUNTS:
            positive_idx = set(order[:n_target].tolist())
            preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                     for i in range(len(test_pairs))]
            pos_count = sum(1 for _, c in preds if c == 1)
            out = ROOT / f"submission_ens_{tag}_top{n_target}.csv"
            write_submission(preds, out)
            thr_at = ens[order[n_target - 1]] if n_target > 0 else 1.0
            print(f"  top {n_target:<4} (thr~{thr_at:.3f}) "
                  f"positives={pos_count:<4} -> {out.name}")


if __name__ == "__main__":
    main()
